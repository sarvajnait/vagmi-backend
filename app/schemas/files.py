from pydantic import BaseModel


class PresignRequest(BaseModel):
    """Request schema for generating presigned URLs for file uploads.
    
    Attributes:
        fileName: The name of the file to be uploaded
        fileType: MIME type of the file (e.g., 'application/pdf', 'image/jpeg')
        path: The destination path where the file will be stored
    """
    fileName: str
    fileType: str
    path: str
