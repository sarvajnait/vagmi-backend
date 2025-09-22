import json
import os
from typing import List, Optional, Dict, Any, Sequence
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_postgres import PGVector
from langgraph.graph import StateGraph, MessagesState, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
import uvicorn
from loguru import logger
from dotenv import load_dotenv
from typing_extensions import TypedDict
from sqlmodel import SQLModel, Field, create_engine, Session, select, Relationship, text
from sqlalchemy.orm import sessionmaker
from contextlib import asynccontextmanager

# Load environment variables
load_dotenv()

# Database configuration
DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://username:password@localhost:5432/education_db"
)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable not set")


# SQLModel Models
class ClassLevelBase(SQLModel):
    name: str = Field(unique=True, index=True)


class ClassLevel(ClassLevelBase, table=True):
    __tablename__ = "class_levels"

    id: Optional[int] = Field(default=None, primary_key=True)

    # Relationships
    boards: List["Board"] = Relationship(
        back_populates="class_level", cascade_delete=True
    )


class ClassLevelCreate(ClassLevelBase):
    pass


class ClassLevelRead(ClassLevelBase):
    id: int


class BoardBase(SQLModel):
    name: str
    class_level_id: int = Field(foreign_key="class_levels.id")


class Board(BoardBase, table=True):
    __tablename__ = "boards"

    id: Optional[int] = Field(default=None, primary_key=True)

    # Relationships
    class_level: ClassLevel = Relationship(back_populates="boards")
    mediums: List["Medium"] = Relationship(back_populates="board", cascade_delete=True)

    class Config:
        from_attributes = True


class BoardCreate(BoardBase):
    pass


class BoardRead(BoardBase):
    id: int
    class_level_name: Optional[str] = None


class MediumBase(SQLModel):
    name: str
    board_id: int = Field(foreign_key="boards.id")


class Medium(MediumBase, table=True):
    __tablename__ = "mediums"

    id: Optional[int] = Field(default=None, primary_key=True)

    # Relationships
    board: Board = Relationship(back_populates="mediums")
    subjects: List["Subject"] = Relationship(
        back_populates="medium", cascade_delete=True
    )


class MediumCreate(MediumBase):
    pass


class MediumRead(MediumBase):
    id: int
    board_name: Optional[str] = None


class SubjectBase(SQLModel):
    name: str
    medium_id: int = Field(foreign_key="mediums.id")


class Subject(SubjectBase, table=True):
    __tablename__ = "subjects"

    id: Optional[int] = Field(default=None, primary_key=True)

    # Relationships
    medium: Medium = Relationship(back_populates="subjects")
    chapters: List["Chapter"] = Relationship(
        back_populates="subject", cascade_delete=True
    )


class SubjectCreate(SubjectBase):
    pass


class SubjectRead(SubjectBase):
    id: int
    medium_name: Optional[str] = None


class ChapterBase(SQLModel):
    name: str
    subject_id: int = Field(foreign_key="subjects.id")


class Chapter(ChapterBase, table=True):
    __tablename__ = "chapters"

    id: Optional[int] = Field(default=None, primary_key=True)

    # Relationships
    subject: Subject = Relationship(back_populates="chapters")


class ChapterCreate(ChapterBase):
    pass


class ChapterRead(ChapterBase):
    id: int
    subject_name: Optional[str] = None


class ChatRequest(SQLModel):
    message: str
    class_level: Optional[str] = None
    board: Optional[str] = None
    medium: Optional[str] = None
    subject: Optional[str] = None
    chapter: Optional[str] = None


class DocumentUploadRequest(SQLModel):
    class_level: str
    board: str
    medium: str
    subject: str
    chapter: str


class HierarchyFilter(SQLModel):
    class_level: Optional[str] = None
    board: Optional[str] = None
    medium: Optional[str] = None
    subject: Optional[str] = None


class DeleteRequest(SQLModel):
    filename: str
    class_level: Optional[str] = None
    board: Optional[str] = None
    medium: Optional[str] = None
    subject: Optional[str] = None
    chapter: Optional[str] = None


class DocumentResponse(SQLModel):
    filename: str
    class_level: str
    board: str
    medium: str
    subject: str
    chapter: str


# LangGraph State
class RAGState(TypedDict):
    messages: List
    context: List[Document]
    query: str
    filters: Dict[str, str]


# Database setup
engine = create_engine(DATABASE_URL, echo=False)


def create_db_and_tables():
    SQLModel.metadata.create_all(engine)


def get_session():
    with Session(engine) as session:
        yield session


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    create_db_and_tables()
    logger.info("Database tables created successfully")

    yield  # This is where the app runs

    # Shutdown logic (if any)
    logger.info("Shutting down application")


app = FastAPI(lifespan=lifespan)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    verbose=False,
    temperature=0.5,
    streaming=True,
)

# Initialize PGVector store
CONNECTION_STRING = DATABASE_URL
COLLECTION_NAME = "education_documents"

# Create the vector store
vector_store = PGVector(
    embeddings=embeddings,
    collection_name=COLLECTION_NAME,
    connection=CONNECTION_STRING,
    use_jsonb=True,
)

# Memory checkpointer for conversations
memory_checkpointer = MemorySaver()


class EducationPlatform:
    """Main class handling the education platform logic"""

    def __init__(self):
        self.vector_store = vector_store
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=20, length_function=len
        )

    def get_hierarchy_options(
        self, filter_params: HierarchyFilter, session: Session
    ) -> Dict[str, List[str]]:
        """Get available options for each level of hierarchy based on filters"""
        try:
            result = {
                "class_level": [],
                "board": [],
                "medium": [],
                "subject": [],
                "chapter": [],
            }

            # Always get all class levels
            class_levels = session.exec(
                select(ClassLevel).order_by(
                    text(
                        "CASE WHEN name ~ '^[0-9]+$' THEN CAST(name AS INTEGER) ELSE 999 END, name"
                    )
                )
            ).all()
            result["class_level"] = [cl.name for cl in class_levels]

            # Get boards based on selected class level
            if filter_params.class_level:
                boards = session.exec(
                    select(Board)
                    .join(ClassLevel)
                    .where(ClassLevel.name == filter_params.class_level)
                    .order_by(Board.name)
                ).all()
                result["board"] = [b.name for b in boards]

            # Get mediums based on selected board
            if filter_params.board and filter_params.class_level:
                mediums = session.exec(
                    select(Medium)
                    .join(Board)
                    .join(ClassLevel)
                    .where(
                        ClassLevel.name == filter_params.class_level,
                        Board.name == filter_params.board,
                    )
                    .order_by(Medium.name)
                ).all()
                result["medium"] = [m.name for m in mediums]

            # Get subjects based on selected medium
            if (
                filter_params.medium
                and filter_params.board
                and filter_params.class_level
            ):
                subjects = session.exec(
                    select(Subject)
                    .join(Medium)
                    .join(Board)
                    .join(ClassLevel)
                    .where(
                        ClassLevel.name == filter_params.class_level,
                        Board.name == filter_params.board,
                        Medium.name == filter_params.medium,
                    )
                    .order_by(Subject.name)
                ).all()
                result["subject"] = [s.name for s in subjects]

            # Get chapters based on selected subject
            if (
                filter_params.subject
                and filter_params.medium
                and filter_params.board
                and filter_params.class_level
            ):
                chapters = session.exec(
                    select(Chapter)
                    .join(Subject)
                    .join(Medium)
                    .join(Board)
                    .join(ClassLevel)
                    .where(
                        ClassLevel.name == filter_params.class_level,
                        Board.name == filter_params.board,
                        Medium.name == filter_params.medium,
                        Subject.name == filter_params.subject,
                    )
                    .order_by(Chapter.name)
                ).all()
                result["chapter"] = [ch.name for ch in chapters]

            return result

        except Exception as e:
            logger.error(f"Error getting hierarchy options: {e}")
            return {
                level: []
                for level in ["class_level", "board", "medium", "subject", "chapter"]
            }

    def upload_document(self, file_path: str, metadata: Dict[str, str]) -> int:
        """Upload document with hierarchical metadata"""
        try:
            loader = PyPDFLoader(file_path)
            documents = loader.load_and_split(self.text_splitter)

            # Add metadata to each document chunk
            for doc in documents:
                doc.metadata.update(
                    {
                        "class_level": metadata["class_level"],
                        "board": metadata["board"],
                        "medium": metadata["medium"],
                        "subject": metadata["subject"],
                        "chapter": metadata["chapter"],
                        "source_file": os.path.basename(file_path),
                        "full_path": file_path,
                    }
                )

            # Add documents to vector store
            self.vector_store.add_documents(documents)

            logger.info(f"Successfully uploaded {len(documents)} document chunks")
            return len(documents)

        except Exception as e:
            logger.error(f"Error uploading document: {e}")
            raise

    def create_retrieval_tool(self, filters: Dict[str, str]):
        """Create retrieval tool with hierarchical filters"""

        @tool(response_format="content_and_artifact")
        def retrieve_filtered_content(query: str):
            """Retrieve educational content based on hierarchical filters."""
            try:
                # Build metadata filter
                metadata_filter = {}
                if filters.get("class_level"):
                    metadata_filter["class_level"] = filters["class_level"]
                if filters.get("board"):
                    metadata_filter["board"] = filters["board"]
                if filters.get("medium"):
                    metadata_filter["medium"] = filters["medium"]
                if filters.get("subject"):
                    metadata_filter["subject"] = filters["subject"]
                if filters.get("chapter"):
                    metadata_filter["chapter"] = filters["chapter"]

                # Perform similarity search with metadata filter
                retrieved_docs = self.vector_store.similarity_search(
                    query, k=3, filter=metadata_filter
                )

                if not retrieved_docs:
                    filter_desc = ", ".join(
                        [f"{k}: {v}" for k, v in metadata_filter.items()]
                    )
                    return f"No relevant content found for filters: {filter_desc}", []

                # Format retrieved content
                serialized = "\n\n".join(
                    [
                        f"Source: {doc.metadata.get('source_file', 'Unknown')}\n"
                        f"Chapter: {doc.metadata.get('chapter', 'N/A')}\n"
                        f"Content: {doc.page_content}"
                        for doc in retrieved_docs
                    ]
                )

                return serialized, retrieved_docs

            except Exception as e:
                logger.error(f"Error in retrieval: {e}")
                return f"Error retrieving content: {str(e)}", []

        return retrieve_filtered_content

    def create_rag_graph(self, filters: Dict[str, str]):
        """Create RAG graph with hierarchical filtering"""
        retrieval_tool = self.create_retrieval_tool(filters)

        def query_or_respond(state: MessagesState):
            """Generate tool call for retrieval or respond directly."""
            filter_desc = ", ".join([f"{k}: {v}" for k, v in filters.items() if v])

            system_message = SystemMessage(
                content=f"""You are an educational assistant for Indian students.
                Current context: {filter_desc}
                Use simple, age-appropriate language suitable for the class level.
                If the question is a greeting, respond directly.
                Otherwise, use the retrieval tool to get relevant educational content.
                """
            )

            messages = state.get("messages") or []
            messages = [system_message] + messages

            llm_with_tools = llm.bind_tools([retrieval_tool])
            response = llm_with_tools.invoke(messages)
            return {"messages": [response]}

        def generate_final_response(state: MessagesState):
            """Generate final response using retrieved content."""
            context = state["messages"][-1].content

            system_message_content = (
                f"You are an educational assistant. Use the following retrieved content "
                f"to answer the question. Current filters: {filters}\n\n"
                f"Question: {state.get('query', '')}\n\n"
                f"Context: {context}"
            )

            conversation_messages = [
                msg
                for msg in state["messages"]
                if msg.type in ("human", "system")
                or (msg.type == "ai" and not getattr(msg, "tool_calls", False))
            ]

            prompt_messages = [
                SystemMessage(system_message_content)
            ] + conversation_messages
            response = llm.invoke(prompt_messages)
            return {"messages": [response]}

        # Build the graph
        graph_builder = StateGraph(MessagesState)
        graph_builder.add_node("query_or_respond", query_or_respond)
        graph_builder.add_node("tools", ToolNode([retrieval_tool]))
        graph_builder.add_node("generate", generate_final_response)

        graph_builder.set_entry_point("query_or_respond")
        graph_builder.add_conditional_edges(
            "query_or_respond",
            tools_condition,
            {END: END, "tools": "tools"},
        )
        graph_builder.add_edge("tools", "generate")
        graph_builder.add_edge("generate", END)

        return graph_builder.compile(checkpointer=memory_checkpointer)


# Initialize platform
platform = EducationPlatform()

# CRUD API Endpoints for Hierarchy Management


# Class Level CRUD
@app.get("/class-levels", response_model=Dict[str, List[ClassLevelRead]])
async def get_class_levels(session: Session = Depends(get_session)):
    """Get all class levels"""
    try:
        class_levels = session.exec(
            select(ClassLevel).order_by(
                text(
                    "CASE WHEN name ~ '^[0-9]+$' THEN CAST(name AS INTEGER) ELSE 999 END, name"
                )
            )
        ).all()
        return {"data": class_levels}
    except Exception as e:
        logger.error(f"Error getting class levels: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/class-levels", response_model=Dict[str, ClassLevelRead])
async def create_class_level(
    class_level: ClassLevelCreate, session: Session = Depends(get_session)
):
    """Create a new class level"""
    try:
        # Check if class level already exists
        existing = session.exec(
            select(ClassLevel).where(ClassLevel.name == class_level.name)
        ).first()
        if existing:
            raise HTTPException(status_code=400, detail="Class level already exists")

        db_class_level = ClassLevel.model_validate(class_level)
        session.add(db_class_level)
        session.commit()
        session.refresh(db_class_level)
        return {"data": db_class_level}
    except HTTPException:
        raise
    except Exception as e:
        session.rollback()
        logger.error(f"Error creating class level: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.delete("/class-levels/{class_level_id}")
async def delete_class_level(
    class_level_id: int, session: Session = Depends(get_session)
):
    """Delete a class level"""
    try:
        class_level = session.get(ClassLevel, class_level_id)
        if not class_level:
            raise HTTPException(status_code=404, detail="Class level not found")

        session.delete(class_level)
        session.commit()
        return {"message": "Class level deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        session.rollback()
        logger.error(f"Error deleting class level: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Board CRUD
@app.get("/boards", response_model=Dict[str, List[BoardRead]])
async def get_boards(
    class_level_id: Optional[int] = Query(None), session: Session = Depends(get_session)
):
    """Get boards, optionally filtered by class level"""
    try:
        query = select(Board, ClassLevel).join(ClassLevel)

        if class_level_id:
            query = query.where(Board.class_level_id == class_level_id)

        query = query.order_by(ClassLevel.name, Board.name)
        results = session.exec(query).all()

        boards = [
            BoardRead(
                id=board.id,
                name=board.name,
                class_level_id=board.class_level_id,
                class_level_name=class_level.name,
            )
            for board, class_level in results
        ]

        return {"data": boards}
    except Exception as e:
        logger.error(f"Error getting boards: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/boards", response_model=Dict[str, BoardRead])
async def create_board(board: BoardCreate, session: Session = Depends(get_session)):
    """Create a new board"""
    try:
        # Check if class level exists
        class_level = session.get(ClassLevel, board.class_level_id)
        if not class_level:
            raise HTTPException(status_code=400, detail="Class level not found")

        # Check if board already exists for this class level
        existing = session.exec(
            select(Board).where(
                Board.name == board.name, Board.class_level_id == board.class_level_id
            )
        ).first()
        if existing:
            raise HTTPException(
                status_code=400, detail="Board already exists for this class level"
            )

        db_board = Board.model_validate(board)
        session.add(db_board)
        session.commit()
        session.refresh(db_board)

        return {
            "data": BoardRead(
                id=db_board.id,
                name=db_board.name,
                class_level_id=db_board.class_level_id,
                class_level_name=class_level.name,
            )
        }
    except HTTPException:
        raise
    except Exception as e:
        session.rollback()
        logger.error(f"Error creating board: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.delete("/boards/{board_id}")
async def delete_board(board_id: int, session: Session = Depends(get_session)):
    """Delete a board"""
    try:
        board = session.get(Board, board_id)
        if not board:
            raise HTTPException(status_code=404, detail="Board not found")

        session.delete(board)
        session.commit()
        return {"message": "Board deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        session.rollback()
        logger.error(f"Error deleting board: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Medium CRUD
@app.get("/mediums", response_model=Dict[str, List[MediumRead]])
async def get_mediums(
    board_id: Optional[int] = Query(None), session: Session = Depends(get_session)
):
    """Get mediums, optionally filtered by board"""
    try:
        query = select(Medium, Board).join(Board)

        if board_id:
            query = query.where(Medium.board_id == board_id)

        query = query.order_by(Board.name, Medium.name)
        results = session.exec(query).all()

        mediums = [
            MediumRead(
                id=medium.id,
                name=medium.name,
                board_id=medium.board_id,
                board_name=board.name,
            )
            for medium, board in results
        ]

        return {"data": mediums}
    except Exception as e:
        logger.error(f"Error getting mediums: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/mediums", response_model=Dict[str, MediumRead])
async def create_medium(medium: MediumCreate, session: Session = Depends(get_session)):
    """Create a new medium"""
    try:
        # Check if board exists
        board = session.get(Board, medium.board_id)
        if not board:
            raise HTTPException(status_code=400, detail="Board not found")

        # Check if medium already exists for this board
        existing = session.exec(
            select(Medium).where(
                Medium.name == medium.name, Medium.board_id == medium.board_id
            )
        ).first()
        if existing:
            raise HTTPException(
                status_code=400, detail="Medium already exists for this board"
            )

        db_medium = Medium.model_validate(medium)
        session.add(db_medium)
        session.commit()
        session.refresh(db_medium)

        return {
            "data": MediumRead(
                id=db_medium.id,
                name=db_medium.name,
                board_id=db_medium.board_id,
                board_name=board.name,
            )
        }
    except HTTPException:
        raise
    except Exception as e:
        session.rollback()
        logger.error(f"Error creating medium: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.delete("/mediums/{medium_id}")
async def delete_medium(medium_id: int, session: Session = Depends(get_session)):
    """Delete a medium"""
    try:
        medium = session.get(Medium, medium_id)
        if not medium:
            raise HTTPException(status_code=404, detail="Medium not found")

        session.delete(medium)
        session.commit()
        return {"message": "Medium deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        session.rollback()
        logger.error(f"Error deleting medium: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Subject CRUD
@app.get("/subjects", response_model=Dict[str, List[SubjectRead]])
async def get_subjects(
    medium_id: Optional[int] = Query(None), session: Session = Depends(get_session)
):
    """Get subjects, optionally filtered by medium"""
    try:
        query = select(Subject, Medium).join(Medium)

        if medium_id:
            query = query.where(Subject.medium_id == medium_id)

        query = query.order_by(Medium.name, Subject.name)
        results = session.exec(query).all()

        subjects = [
            SubjectRead(
                id=subject.id,
                name=subject.name,
                medium_id=subject.medium_id,
                medium_name=medium.name,
            )
            for subject, medium in results
        ]

        return {"data": subjects}
    except Exception as e:
        logger.error(f"Error getting subjects: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/subjects", response_model=Dict[str, SubjectRead])
async def create_subject(
    subject: SubjectCreate, session: Session = Depends(get_session)
):
    """Create a new subject"""
    try:
        # Check if medium exists
        medium = session.get(Medium, subject.medium_id)
        if not medium:
            raise HTTPException(status_code=400, detail="Medium not found")

        # Check if subject already exists for this medium
        existing = session.exec(
            select(Subject).where(
                Subject.name == subject.name, Subject.medium_id == subject.medium_id
            )
        ).first()
        if existing:
            raise HTTPException(
                status_code=400, detail="Subject already exists for this medium"
            )

        db_subject = Subject.model_validate(subject)
        session.add(db_subject)
        session.commit()
        session.refresh(db_subject)

        return {
            "data": SubjectRead(
                id=db_subject.id,
                name=db_subject.name,
                medium_id=db_subject.medium_id,
                medium_name=medium.name,
            )
        }
    except HTTPException:
        raise
    except Exception as e:
        session.rollback()
        logger.error(f"Error creating subject: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.delete("/subjects/{subject_id}")
async def delete_subject(subject_id: int, session: Session = Depends(get_session)):
    """Delete a subject"""
    try:
        subject = session.get(Subject, subject_id)
        if not subject:
            raise HTTPException(status_code=404, detail="Subject not found")

        session.delete(subject)
        session.commit()
        return {"message": "Subject deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        session.rollback()
        logger.error(f"Error deleting subject: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Chapter CRUD
@app.get("/chapters", response_model=Dict[str, List[ChapterRead]])
async def get_chapters(
    subject_id: Optional[int] = Query(None), session: Session = Depends(get_session)
):
    """Get chapters, optionally filtered by subject"""
    try:
        query = select(Chapter, Subject).join(Subject)

        if subject_id:
            query = query.where(Chapter.subject_id == subject_id)

        query = query.order_by(Subject.name, Chapter.name)
        results = session.exec(query).all()

        chapters = [
            ChapterRead(
                id=chapter.id,
                name=chapter.name,
                subject_id=chapter.subject_id,
                subject_name=subject.name,
            )
            for chapter, subject in results
        ]

        return {"data": chapters}
    except Exception as e:
        logger.error(f"Error getting chapters: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chapters", response_model=Dict[str, ChapterRead])
async def create_chapter(
    chapter: ChapterCreate, session: Session = Depends(get_session)
):
    """Create a new chapter"""
    try:
        # Check if subject exists
        subject = session.get(Subject, chapter.subject_id)
        if not subject:
            raise HTTPException(status_code=400, detail="Subject not found")

        # Check if chapter already exists for this subject
        existing = session.exec(
            select(Chapter).where(
                Chapter.name == chapter.name, Chapter.subject_id == chapter.subject_id
            )
        ).first()
        if existing:
            raise HTTPException(
                status_code=400, detail="Chapter already exists for this subject"
            )

        db_chapter = Chapter.model_validate(chapter)
        session.add(db_chapter)
        session.commit()
        session.refresh(db_chapter)

        return {
            "data": ChapterRead(
                id=db_chapter.id,
                name=db_chapter.name,
                subject_id=db_chapter.subject_id,
                subject_name=subject.name,
            )
        }
    except HTTPException:
        raise
    except Exception as e:
        session.rollback()
        logger.error(f"Error creating chapter: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.delete("/chapters/{chapter_id}")
async def delete_chapter(chapter_id: int, session: Session = Depends(get_session)):
    """Delete a chapter"""
    try:
        chapter = session.get(Chapter, chapter_id)
        if not chapter:
            raise HTTPException(status_code=404, detail="Chapter not found")

        session.delete(chapter)
        session.commit()
        return {"message": "Chapter deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        session.rollback()
        logger.error(f"Error deleting chapter: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Main API Endpoints


@app.get("/hierarchy-options")
async def get_hierarchy_options(
    class_level: Optional[str] = Query(None),
    board: Optional[str] = Query(None),
    medium: Optional[str] = Query(None),
    subject: Optional[str] = Query(None),
    session: Session = Depends(get_session),
):
    """Get available options for hierarchy dropdowns"""
    filter_params = HierarchyFilter(
        class_level=class_level, board=board, medium=medium, subject=subject
    )
    options = platform.get_hierarchy_options(filter_params, session)
    return {"data": options}


@app.post("/upload-document")
async def upload_document(
    file: UploadFile = File(...),
    class_level: str = Form(...),
    board: str = Form(...),
    medium: str = Form(...),
    subject: str = Form(...),
    chapter: str = Form(...),
):
    """Upload document with hierarchical classification"""
    try:
        # Create directory structure
        dir_path = f"files/{class_level}/{board}/{medium}/{subject}/{chapter}"
        os.makedirs(dir_path, exist_ok=True)

        file_path = f"{dir_path}/{file.filename}"

        # Save file
        with open(file_path, "wb") as f:
            f.write(await file.read())

        # Upload to vector store
        metadata = {
            "class_level": class_level,
            "board": board,
            "medium": medium,
            "subject": subject,
            "chapter": chapter,
        }

        doc_count = platform.upload_document(file_path, metadata)

        return {
            "message": "Document uploaded successfully",
            "metadata": metadata,
            "documents_processed": doc_count,
        }

    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/stream-chat")
async def stream_chat(chat_request: ChatRequest):
    """Stream chat responses with hierarchical filtering"""

    async def generate_response():
        try:
            # Build filters
            filters = {}
            if chat_request.class_level:
                filters["class_level"] = chat_request.class_level
            if chat_request.board:
                filters["board"] = chat_request.board
            if chat_request.medium:
                filters["medium"] = chat_request.medium
            if chat_request.subject:
                filters["subject"] = chat_request.subject
            if chat_request.chapter:
                filters["chapter"] = chat_request.chapter

            # Create RAG graph with filters
            rag_graph = platform.create_rag_graph(filters)

            # Generate thread config
            filter_key = "_".join([f"{k}_{v}" for k, v in filters.items() if v])
            config = {"configurable": {"thread_id": f"session_{filter_key}"}}

            input_messages = [{"role": "user", "content": chat_request.message}]

            # Stream response
            async for event in rag_graph.astream(
                {"messages": input_messages, "query": chat_request.message},
                config=config,
                stream_mode="messages",
            ):
                message, metadata = event
                if isinstance(message, AIMessage) and message.content:
                    for chunk in message.content:
                        if chunk:
                            yield f"data: {json.dumps({'type': 'token', 'content': str(chunk)})}\n\n"

            yield f"data: {json.dumps({'type': 'complete', 'content': ''})}\n\n"

        except Exception as e:
            logger.error(f"Error in stream chat: {e}")
            yield f"data: {json.dumps({'type': 'error', 'content': f'Error: {str(e)}'})}\n\n"

    return StreamingResponse(
        generate_response(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream",
        },
    )


# Also add this debug endpoint right after the documents endpoint
@app.get("/debug-vector-store")
async def debug_vector_store(session: Session = Depends(get_session)):
    """Debug endpoint to check vector store contents"""
    try:
        # Check collections
        collections_query = "SELECT uuid, name FROM langchain_pg_collection"
        collections = session.exec(text(collections_query)).fetchall()

        # Check embeddings count
        if collections:
            collection_uuid = collections[0][0]  # Use first collection
            embeddings_query = "SELECT COUNT(*) FROM langchain_pg_embedding WHERE collection_id = :collection_uuid"
            embeddings_count = session.exec(
                text(embeddings_query), {"collection_uuid": collection_uuid}
            ).fetchone()[0]

            # Sample metadata
            sample_query = """
            SELECT cmetadata->>'source_file' as source_file,
                   cmetadata->>'class_level' as class_level,
                   cmetadata->>'board' as board,
                   cmetadata->>'medium' as medium,
                   cmetadata->>'subject' as subject,
                   cmetadata->>'chapter' as chapter,
                   cmetadata
            FROM langchain_pg_embedding 
            WHERE collection_id = :collection_uuid 
            LIMIT 5
            """
            sample_data = session.exec(
                text(sample_query), {"collection_uuid": collection_uuid}
            ).fetchall()

            return {
                "collections": [{"uuid": str(c[0]), "name": c[1]} for c in collections],
                "total_embeddings": embeddings_count,
                "sample_metadata": [
                    {
                        "source_file": row[0],
                        "class_level": row[1],
                        "board": row[2],
                        "medium": row[3],
                        "subject": row[4],
                        "chapter": row[5],
                        "full_metadata": row[6],
                    }
                    for row in sample_data
                ],
                "expected_collection_name": COLLECTION_NAME,
            }
        else:
            return {
                "collections": [],
                "message": "No collections found in vector store",
                "expected_collection_name": COLLECTION_NAME,
            }
    except Exception as e:
        return {"error": str(e), "expected_collection_name": COLLECTION_NAME}


@app.get("/documents", response_model=Dict[str, List[DocumentResponse]])
async def get_documents(
    class_level: Optional[str] = Query(None),
    board: Optional[str] = Query(None),
    medium: Optional[str] = Query(None),
    subject: Optional[str] = Query(None),
    chapter: Optional[str] = Query(None),
    session: Session = Depends(get_session),
):
    """Get documents based on hierarchical filters"""
    try:
        # Build WHERE clause
        where_conditions = ["1=1"]
        params = {"collection_name": COLLECTION_NAME}

        if class_level:
            where_conditions.append("cmetadata->>'class_level' = :class_level")
            params["class_level"] = class_level
        if board:
            where_conditions.append("cmetadata->>'board' = :board")
            params["board"] = board
        if medium:
            where_conditions.append("cmetadata->>'medium' = :medium")
            params["medium"] = medium
        if subject:
            where_conditions.append("cmetadata->>'subject' = :subject")
            params["subject"] = subject
        if chapter:
            where_conditions.append("cmetadata->>'chapter' = :chapter")
            params["chapter"] = chapter

        where_clause = " AND ".join(where_conditions)

        query_text = f"""
        SELECT DISTINCT cmetadata->>'source_file' as source_file,
               cmetadata->>'class_level' as class_level,
               cmetadata->>'board' as board,
               cmetadata->>'medium' as medium,
               cmetadata->>'subject' as subject,
               cmetadata->>'chapter' as chapter
        FROM langchain_pg_embedding 
        WHERE collection_id = (
            SELECT uuid FROM langchain_pg_collection 
            WHERE name = :collection_name
        ) AND {where_clause}
        ORDER BY class_level, board, medium, subject, chapter, source_file
        """

        result = session.execute(text(query_text), params)
        documents = [
            DocumentResponse(
                filename=row[0],
                class_level=row[1],
                board=row[2],
                medium=row[3],
                subject=row[4],
                chapter=row[5],
            )
            for row in result.fetchall()
        ]

        return {"data": documents}

    except Exception as e:
        logger.error(f"Error getting documents: {e}")
        return {"data": []}


@app.delete("/delete-document")
async def delete_document(
    delete_request: DeleteRequest, session: Session = Depends(get_session)
):
    """Delete document from vector store"""
    try:
        # Build WHERE clause for finding documents
        where_conditions = [
            "collection_id = (SELECT uuid FROM langchain_pg_collection WHERE name = :collection_name)"
        ]
        params = {"collection_name": COLLECTION_NAME}

        where_conditions.append("cmetadata->>'source_file' = :filename")
        params["filename"] = delete_request.filename

        if delete_request.class_level:
            where_conditions.append("cmetadata->>'class_level' = :class_level")
            params["class_level"] = delete_request.class_level
        if delete_request.board:
            where_conditions.append("cmetadata->>'board' = :board")
            params["board"] = delete_request.board
        if delete_request.medium:
            where_conditions.append("cmetadata->>'medium' = :medium")
            params["medium"] = delete_request.medium
        if delete_request.subject:
            where_conditions.append("cmetadata->>'subject' = :subject")
            params["subject"] = delete_request.subject
        if delete_request.chapter:
            where_conditions.append("cmetadata->>'chapter' = :chapter")
            params["chapter"] = delete_request.chapter

        where_clause = " AND ".join(where_conditions)

        # Delete from vector store
        delete_query = f"DELETE FROM langchain_pg_embedding WHERE {where_clause}"
        result = session.execute(text(delete_query), params)
        session.commit()

        if result.rowcount == 0:
            raise HTTPException(status_code=404, detail="Document not found")

        return {
            "message": f"Document '{delete_request.filename}' deleted successfully",
            "deleted_chunks": result.rowcount,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "database": "PostgreSQL with pgvector",
        "orm": "SQLModel + SQLAlchemy",
        "hierarchy_order": "Class → Board → Medium → Subject → Chapter",
        "features": [
            "hierarchical_classification",
            "filtered_retrieval",
            "streaming_chat",
            "crud_operations",
            "type_safety",
        ],
    }


@app.get("/")
def root():
    return {
        "message": "Hierarchical Education Platform API with SQLModel",
        "description": "Class → Board → Medium → Subject → Chapter based learning system",
        "hierarchy_order": "Class Level → Board → Medium → Subject → Chapter",
        "technologies": [
            "FastAPI",
            "SQLModel",
            "SQLAlchemy",
            "PostgreSQL + pgvector",
            "LangChain",
            "Google Gemini",
        ],
        "features": [
            "Type-safe database operations",
            "Hierarchical document classification",
            "Cascading dropdown filters",
            "Context-aware chat responses",
            "Complete CRUD operations for hierarchy",
            "Relationship management",
        ],
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
