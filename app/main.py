import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import transcribe, api_key, livez

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
