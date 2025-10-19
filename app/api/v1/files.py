from fastapi import APIRouter, HTTPException
from loguru import logger
import boto3
import os
from app.schemas import PresignRequest

router = APIRouter()


@router.post("/presigned-url")
async def generate_presigned_url(request: PresignRequest):
    """Generate a presigned upload URL for DigitalOcean Spaces"""
    try:
        access_key = os.getenv("DO_SPACES_ACCESS_KEY")
        secret_key = os.getenv("DO_SPACES_SECRET_KEY")

        if not access_key or not secret_key:
            raise HTTPException(
                status_code=500, detail="Missing DigitalOcean credentials"
            )

        s3_client = boto3.client(
            "s3",
            region_name="us-east-1",
            endpoint_url="https://blr1.digitaloceanspaces.com",
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        )

        bucket_name = "vagmi"
        object_key = f"{request.path}/{request.fileName}"

        presigned_url = s3_client.generate_presigned_url(
            "put_object",
            Params={
                "Bucket": bucket_name,
                "Key": object_key,
                "ContentType": request.fileType,
                "ACL": "public-read",
            },
            ExpiresIn=3600,  # 1 hour
        )

        logger.info(f"Generated presigned URL for {object_key}")
        return {"url": presigned_url}

    except Exception as e:
        logger.error(f"Error generating presigned URL: {e}")
        raise HTTPException(status_code=500, detail=str(e))
