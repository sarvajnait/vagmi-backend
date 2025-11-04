from typing import Optional
from sqlmodel import SQLModel

class DocumentUploadRequest(SQLModel):
    """Request schema for uploading educational documents with hierarchical context.
    
    Attributes:
        class_level: Name of the class level (e.g., "Class 10", "Class 12")
        board: Name of the educational board (e.g., "CBSE", "ICSE")
        medium: Medium of instruction (e.g., "English", "Hindi")
        subject: Name of the subject (e.g., "Mathematics", "Physics")
        chapter: Name of the specific chapter
    """
    class_level: str
    board: str
    medium: str
    subject: str
    chapter: str


class DocumentResponse(SQLModel):
    """Response schema for document operations with hierarchical information.
    
    Attributes:
        filename: Name of the document file
        class_level: Class level where the document is located
        board: Educational board context
        medium: Medium of instruction context
        subject: Subject context
        chapter: Specific chapter context
    """
    filename: str
    class_level: str
    board: str
    medium: str
    subject: str
    chapter: str


class DeleteRequest(SQLModel):
    """Request schema for deleting documents with optional hierarchical filters.
    
    Attributes:
        filename: Name of the file to be deleted
        class_level: Optional class level filter for bulk deletion
        board: Optional board filter for bulk deletion
        medium: Optional medium filter for bulk deletion
        subject: Optional subject filter for bulk deletion
        chapter: Optional chapter filter for bulk deletion
    """
    filename: str
    class_level: Optional[str] = None
    board: Optional[str] = None
    medium: Optional[str] = None
    subject: Optional[str] = None
    chapter: Optional[str] = None
