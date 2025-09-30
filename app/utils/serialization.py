from typing import Any

from pydantic import BaseModel


def custom_default(obj: Any) -> Any:
    """
    Custom serialization function for objects that may not be JSON serializable by default.
    This mimics the original default_serialization but is renamed and placed in a different location.
    """
    if isinstance(obj, BaseModel):
        return obj.model_dump()  # convert a Pydantic model into a plain dict
    return str(obj)  # fallback: just convert it to string
