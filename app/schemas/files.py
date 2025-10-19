from pydantic import BaseModel


class PresignRequest(BaseModel):
    fileName: str
    fileType: str
    path: str
