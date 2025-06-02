from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
import httpx
import os
from typing import List, Dict, Any
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError

app = FastAPI()


class ChatRequest(BaseModel):
    messages: List[Dict[str, Any]]

class ChatResponse(BaseModel):
    response: str

# Get the URL from environment variable
LLAMA_SERVER_URL = os.getenv("LLAMA_SERVER_URL", "http://llama-server:8080") + "/v1/chat/completions"

# Retry configuration
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=4), reraise=True)
async def call_llama_server(payload: dict) -> dict:
    logger.debug(f"Attempting to call llama-server at {LLAMA_SERVER_URL}...")
    async with httpx.AsyncClient() as client:
        response = await client.post(LLAMA_SERVER_URL, json=payload, timeout=30.0)
        logger.debug(f"LLaMA server response code: {response.status_code}")
        if response.status_code != 200:
            logger.error(f"LLaMA server error response: {response.text}")
            if 500 <= response.status_code < 600:
                response.raise_for_status()
            else:
                raise HTTPException(status_code=response.status_code, detail=response.text)
        return response.json()

@app.post("/api/v1/chat")
async def chat(request: ChatRequest):
    logger.info(f"Received chat request with {len(request.messages)} messages")

    payload = {
        "model": "qwen-0.6B",
        "messages": request.messages,
        "stream": False,
        "temperature": 0.7,
        "max_tokens": 1000
    }

    try:
        llama_response = await call_llama_server(payload)
        response_text = llama_response.get("choices", [{}])[0].get("message", {}).get("content", "")
        return ChatResponse(response=response_text)

    except RetryError as e:
        logger.error(f"Failed after retries: {e}")
        return ChatResponse(response="Service temporarily unavailable")

    except Exception as e:
        logger.exception("Error processing request")
        raise HTTPException(status_code=500, detail=str(e))
@app.get("/chatbot")
async def redirect_to_chatbot():
    logger.info("Redirecting user to llama-server chatbot UI")
    return RedirectResponse(url="http://localhost:8080/")
@app.get("/health")
async def health_check():
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{os.getenv('LLAMA_SERVER_URL', 'http://llama-server:8080')}/health")
            return {
                "status": "healthy" if response.status_code == 200 else "unhealthy",
                "llama_server": "connected"
            }
    except Exception:
        return {"status": "unhealthy", "llama_server": "disconnected"}
