import os
from fastapi import UploadFile, HTTPException
import boto3
from loguru import logger
from PIL import Image
from io import BytesIO
from fastapi import UploadFile

DO_REGION = "blr1"
DO_BUCKET = "vagmi"
DO_ENDPOINT = f"https://{DO_REGION}.digitaloceanspaces.com"
ACCESS_KEY = os.getenv("DO_SPACES_ACCESS_KEY")
SECRET_KEY = os.getenv("DO_SPACES_SECRET_KEY")


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

    object_key = f"{path}/{file.filename}"

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

    # Create a new UploadFile-like object
    compressed_file = UploadFile(
        filename=file.filename,
        file=buffer,
        content_type=file.content_type,
    )

    return compressed_file
