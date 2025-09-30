import aiofiles
import base64
from fastapi import APIRouter, File, UploadFile
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from loguru import logger

router = APIRouter()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    verbose=False,
    temperature=0.5,
    streaming=True,
)


@router.post("/")
async def transcribe(audio: UploadFile = File(...)):
    tmp_path = f"/tmp/{audio.filename}"
    async with aiofiles.open(tmp_path, "wb") as f:
        await f.write(await audio.read())

    with open(tmp_path, "rb") as f:
        encoded_audio = base64.b64encode(f.read()).decode("utf-8")

    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": "Transcribe only the spoken words as plain text, no JSON, no timestamps.",
            },
            {"type": "media", "data": encoded_audio, "mime_type": audio.content_type},
        ]
    )

    try:
        response = llm.invoke([message])
        return {"text": response.content}
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        return {"error": str(e)}
