import os
import re
import uuid
from fastapi import UploadFile, HTTPException
from starlette.datastructures import Headers
import boto3
from loguru import logger
from PIL import Image
from io import BytesIO

DO_REGION = "blr1"
DO_BUCKET = "vagmi"
DO_ENDPOINT = f"https://{DO_REGION}.digitaloceanspaces.com"
ACCESS_KEY = os.getenv("DO_SPACES_ACCESS_KEY")
SECRET_KEY = os.getenv("DO_SPACES_SECRET_KEY")


def _safe_filename(filename: str) -> str:
    if not filename:
        return "file"
    name = filename.strip().replace("\\", "_").replace("/", "_")
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name)
    return name or "file"


def upload_to_do(file: UploadFile, path: str) -> str:
    """Upload file directly to DigitalOcean Spaces and return the public URL."""
    if not ACCESS_KEY or not SECRET_KEY:
        raise HTTPException(status_code=500, detail="Missing DigitalOcean credentials")

    s3_client = boto3.client(
        "s3",
        region_name=DO_REGION,
        endpoint_url=DO_ENDPOINT,
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
    )

    safe_name = _safe_filename(file.filename)
    object_key = f"{path}/{uuid.uuid4().hex}_{safe_name}"

    try:
        s3_client.upload_fileobj(
            file.file,
            DO_BUCKET,
            object_key,
            ExtraArgs={"ACL": "public-read", "ContentType": file.content_type},
        )
    except Exception as e:
        logger.error(f"Error uploading file to DO: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    public_url = f"{DO_ENDPOINT}/{DO_BUCKET}/{object_key}"
    logger.info(f"File uploaded to DigitalOcean: {public_url}")
    return public_url


def delete_from_do(file_url: str):
    """Delete a file from DigitalOcean Spaces given its public URL."""
    if not ACCESS_KEY or not SECRET_KEY:
        raise HTTPException(status_code=500, detail="Missing DigitalOcean credentials")

    s3_client = boto3.client(
        "s3",
        region_name=DO_REGION,
        endpoint_url=DO_ENDPOINT,
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
    )

    try:
        object_key = file_url.split(f"{DO_BUCKET}/")[-1]
        s3_client.delete_object(Bucket=DO_BUCKET, Key=object_key)
        logger.info(f"Deleted file from DigitalOcean: {object_key}")
    except Exception as e:
        logger.error(f"Error deleting file from DigitalOcean: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting file: {e}")


def compress_image(
    file: UploadFile,
    max_width: int = 1600,
    max_height: int = 1600,
    quality: int = 80,
) -> UploadFile:
    """
    Compress an image UploadFile in-memory and return a new UploadFile.
    """

    image = Image.open(file.file)

    # Convert RGBA / P to RGB (JPEG safe)
    if image.mode in ("RGBA", "P"):
        image = image.convert("RGB")

    # Resize while keeping aspect ratio
    image.thumbnail((max_width, max_height))

    buffer = BytesIO()

    # Decide format
    format = image.format or "JPEG"
    if format.upper() not in ["JPEG", "PNG", "WEBP"]:
        format = "JPEG"

    save_kwargs = {}
    if format.upper() in ["JPEG", "WEBP"]:
        save_kwargs["quality"] = quality
        save_kwargs["optimize"] = True

    image.save(buffer, format=format, **save_kwargs)
    buffer.seek(0)

    # Determine content type based on format
    content_type_map = {
        "JPEG": "image/jpeg",
        "PNG": "image/png",
        "WEBP": "image/webp",
    }
    content_type = content_type_map.get(format.upper(), "image/jpeg")

    # Create headers with content type
    headers = Headers({"content-type": content_type})

    # Create a new UploadFile with headers
    compressed_file = UploadFile(
        file=buffer,
        filename=file.filename,
        headers=headers,
    )

    return compressed_file
