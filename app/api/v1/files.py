from typing import List
from urllib.parse import unquote
from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, Query
from fastapi.security import HTTPBearer
from .session import get_session_by_id
from app.core.logging import logger
from app.models.session import Session
from app.services.database import DatabaseService
from app.core.config import settings
from app.utils.files import (
    get_or_create_session_dir,
    save_uploaded_files,
    get_session_files,
    get_valid_file,
    get_shapefile_components,
)
from fastapi.responses import FileResponse, StreamingResponse
import geopandas as gpd
import io
import shutil
import zipfile
from app.schemas.files import (
    ListSessionFilesResponse,
    FileInfo,
    UploadFilesResponse,
    DeleteFileResponse,
    FileRowsResponse,
)

router = APIRouter()
security = HTTPBearer()
db_service = DatabaseService()


@router.get("", response_model=ListSessionFilesResponse)
def list_session_files(session: Session = Depends(get_session_by_id)):
    """List files in a session"""
    try:
        session_path = get_or_create_session_dir(session.id)
        files_data = get_session_files(session_path, True)
        files = [FileInfo(**f) for f in files_data]
        return ListSessionFilesResponse(files=files)
    except Exception as e:
        logger.error(
            "list_session_files_failed",
            session_id=session.id,
            error=str(e),
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail=f"Error reading session files: {str(e)}")


@router.get("/download")
def download_file(
    filename: str = Query(...),
    session: Session = Depends(get_session_by_id),
):
    """Download a file or shapefile folder as zip."""
    try:
        filename = unquote(filename)
        file_path = get_valid_file(session.id, filename)

        # Handle shapefile folder download as zip
        if file_path.suffix.lower() == ".shp" and file_path.parent.is_dir():
            shapefile_folder = file_path.parent
            zip_buffer = io.BytesIO()

            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                session_path = shapefile_folder.parents[1]  # session root
                if shapefile_folder.resolve() == session_path.resolve():
                    # .shp at root of session folder
                    base_name = file_path.stem
                    for component_file in get_shapefile_components(shapefile_folder, base_name):
                        zip_file.write(component_file, arcname=component_file.name)
                else:
                    # .shp inside a subfolder; include all files in folder
                    for f in shapefile_folder.rglob("*"):
                        if f.is_file():
                            arcname = f.relative_to(shapefile_folder)
                            zip_file.write(f, arcname=arcname)

            zip_buffer.seek(0)
            headers = {"Content-Disposition": f'attachment; filename="{file_path.stem}.zip"'}
            return StreamingResponse(zip_buffer, media_type="application/zip", headers=headers)

        # Handle single file download
        if file_path.is_file():
            return FileResponse(file_path, filename=file_path.name)

        raise HTTPException(status_code=404, detail="File not found")

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found")
    except Exception as e:
        logger.error(
            "file_download_failed",
            session_id=session.id,
            filename=filename,
            error=str(e),
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")


@router.get("/rows", response_model=FileRowsResponse)
def get_file_rows(
    filename: str = Query(...),
    session: Session = Depends(get_session_by_id),
):
    """Get first 10 rows of a file"""
    try:
        file_path = get_valid_file(session.id, filename)
        gdf = gpd.read_file(file_path)
        preview = gdf.head(10).copy()
        if "geometry" in preview.columns:
            preview["geometry"] = preview["geometry"].apply(lambda g: g.wkt if g else None)

        return FileRowsResponse(columns=list(preview.columns), rows=preview.to_dict(orient="records"))
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found")
    except Exception as e:
        logger.error(
            "get_file_rows_failed",
            session_id=session.id,
            filename=filename,
            error=str(e),
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.post("", response_model=UploadFilesResponse)
def upload_files(
    files: List[UploadFile] = File(...),
    session: Session = Depends(get_session_by_id),
):
    """Upload files for a given session."""
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    try:
        session_path = get_or_create_session_dir(session.id)

        save_uploaded_files(files, session_path)

        logger.info("files_uploaded", session_id=session.id, file_count=len(files))
        return UploadFilesResponse(message="Files uploaded", session_id=session.id)

    except Exception as e:
        logger.error("file_upload_failed", session_id=session.id, error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="Internal error while uploading files")


@router.delete("", response_model=DeleteFileResponse)
def delete_file(filename: str = Query(...), session: Session = Depends(get_session_by_id)):
    """Delete a file or shapefile folder."""
    try:
        session_path = get_or_create_session_dir(session.id)
        file_path = get_valid_file(session.id, filename)

        # Delete shapefile folder
        if file_path.suffix.lower() == ".shp":
            folder_path = file_path.parent
            base_name = file_path.stem
            if folder_path.resolve() == session_path.resolve():
                deleted_files = [f.name for f in get_shapefile_components(folder_path, base_name)]
                for f in get_shapefile_components(folder_path, base_name):
                    f.unlink()
                if deleted_files:
                    return DeleteFileResponse(message=f"Deleted shapefile components: {', '.join(deleted_files)}")
                raise HTTPException(status_code=404, detail="No matching shapefile components found")
            if folder_path.exists() and folder_path.is_dir():
                shutil.rmtree(folder_path)
                return DeleteFileResponse(message=f"Shapefile folder '{folder_path.name}' deleted")

        # Delete single file
        if file_path.exists() and file_path.is_file():
            file_path.unlink()
            return DeleteFileResponse(message=f"File '{file_path.name}' deleted.")

        raise HTTPException(status_code=404, detail="File or folder not found")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found")
    except Exception as e:
        logger.error("delete_file_failed", session_id=session.id, filename=filename, error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}")
