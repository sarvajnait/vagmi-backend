from typing import Optional
from sqlmodel import SQLModel

class DocumentUploadRequest(SQLModel):
    class_level: str
    board: str
    medium: str
    subject: str
    chapter: str


class DocumentResponse(SQLModel):
    filename: str
    class_level: str
    board: str
    medium: str
    subject: str
    chapter: str


class DeleteRequest(SQLModel):
    filename: str
    class_level: Optional[str] = None
    board: Optional[str] = None
    medium: Optional[str] = None
    subject: Optional[str] = None
    chapter: Optional[str] = None
