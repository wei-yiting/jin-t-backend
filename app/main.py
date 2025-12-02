import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import transcribe, api_key, livez
from redis.asyncio import Redis

redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
redis_client = Redis.from_url(redis_url, decode_responses=True)

app = FastAPI()

allowed_origins_str = os.getenv("ALLOWED_ORIGINS", "")
allowed_origins = [
    origin.strip() for origin in allowed_origins_str.split(",") if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "HEAD", "POST"],
    allow_headers=["*"],
)

app.include_router(transcribe.router)
app.include_router(api_key.router)
app.include_router(livez.router)
