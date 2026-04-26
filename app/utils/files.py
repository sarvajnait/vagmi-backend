import os
import random
import re
import time
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
DO_RETRY_MAX_ATTEMPTS = 3
DO_RETRY_BASE_DELAY_SEC = 1.0


def _safe_filename(filename: str) -> str:
    if not filename:
        return "file"
    name = filename.strip().replace("\\", "_").replace("/", "_")
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name)
    return name or "file"


def _retry_do_operation(operation_name: str, func):
    last_exc = None
    for attempt in range(1, DO_RETRY_MAX_ATTEMPTS + 1):
        try:
            return func()
        except Exception as exc:
            last_exc = exc
            if attempt >= DO_RETRY_MAX_ATTEMPTS:
                raise
            delay = DO_RETRY_BASE_DELAY_SEC * (2 ** (attempt - 1)) + random.uniform(0, 0.3)
            logger.warning(
                f"{operation_name} failed (attempt {attempt}/{DO_RETRY_MAX_ATTEMPTS}): {exc}. "
                f"Retrying in {delay:.1f}s"
            )
            time.sleep(delay)
    if last_exc:
        raise last_exc
    raise RuntimeError(f"{operation_name} failed without a captured exception")


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
        def _upload():
            file.file.seek(0)
            s3_client.upload_fileobj(
                file.file,
                DO_BUCKET,
                object_key,
                ExtraArgs={"ACL": "public-read", "ContentType": file.content_type},
            )

        _retry_do_operation(f"Upload to DO ({object_key})", _upload)
    except Exception as e:
        logger.error(f"Error uploading file to DO: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    public_url = f"{DO_ENDPOINT}/{DO_BUCKET}/{object_key}"
    logger.info(f"File uploaded to DigitalOcean: {public_url}")
    return public_url


def upload_bytes_to_do(data: bytes, filename: str, path: str, content_type: str = "application/octet-stream") -> str:
    """Upload raw bytes to DigitalOcean Spaces and return the public URL."""
    if not ACCESS_KEY or not SECRET_KEY:
        raise HTTPException(status_code=500, detail="Missing DigitalOcean credentials")

    s3_client = boto3.client(
        "s3",
        region_name=DO_REGION,
        endpoint_url=DO_ENDPOINT,
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
    )

    safe_name = _safe_filename(filename)
    object_key = f"{path}/{uuid.uuid4().hex}_{safe_name}"

    def _upload():
        s3_client.upload_fileobj(
            BytesIO(data),
            DO_BUCKET,
            object_key,
            ExtraArgs={"ACL": "public-read", "ContentType": content_type},
        )

    try:
        _retry_do_operation(f"Upload bytes to DO ({object_key})", _upload)
    except Exception as e:
        logger.error(f"Error uploading bytes to DO: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    public_url = f"{DO_ENDPOINT}/{DO_BUCKET}/{object_key}"
    logger.info(f"Bytes uploaded to DigitalOcean: {public_url}")
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
        _retry_do_operation(
            f"Delete from DO ({object_key})",
            lambda: s3_client.delete_object(Bucket=DO_BUCKET, Key=object_key),
        )
        logger.info(f"Deleted file from DigitalOcean: {object_key}")
    except Exception as e:
        logger.error(f"Error deleting file from DigitalOcean: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting file: {e}")


def delete_prefix_from_do(prefix: str):
    """Delete all objects under a DO Spaces prefix (folder). Silently skips if empty."""
    if not ACCESS_KEY or not SECRET_KEY:
        return

    s3_client = boto3.client(
        "s3",
        region_name=DO_REGION,
        endpoint_url=DO_ENDPOINT,
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
    )

    try:
        paginator = s3_client.get_paginator("list_objects_v2")
        keys = []
        for page in paginator.paginate(Bucket=DO_BUCKET, Prefix=prefix):
            for obj in page.get("Contents", []):
                keys.append({"Key": obj["Key"]})
        if not keys:
            return
        # S3 bulk delete allows up to 1000 keys per call
        for i in range(0, len(keys), 1000):
            s3_client.delete_objects(Bucket=DO_BUCKET, Delete={"Objects": keys[i:i + 1000]})
        logger.info(f"Deleted {len(keys)} objects under prefix: {prefix}")
    except Exception as e:
        logger.warning(f"Could not delete prefix {prefix} from DO: {e}")


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
