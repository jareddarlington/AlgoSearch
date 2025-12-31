import os
from pathlib import Path
from typing import Any, Dict, Optional
from google import genai
from dotenv import load_dotenv

# Load .env from project root
load_dotenv(Path(__file__).parent.parent / ".env")


def create_model(
    model_name: str,
    temperature: float = 0.0,
    max_output_tokens: int = 8192,
    response_schema: Optional[Dict[str, Any]] = None,
):
    """Create a Gemini model with optional JSON schema for structured output."""
    config_kwargs = {
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
    }

    if response_schema:
        config_kwargs["response_mime_type"] = "application/json"
        config_kwargs["response_schema"] = response_schema

    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY", "").strip())
    config = genai.types.GenerateContentConfig(**config_kwargs)
    return client, config


def generate_content(client, model_name: str, config, prompt: str) -> str:
    """Generate content using the model."""
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=config,
    )
    return response.text
