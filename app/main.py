import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from redis.asyncio import Redis

from app.routers.transcribe import router as transcribe_router
from app.routers.api_key import router as api_key_router
from app.routers.livez import router as livez_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    print(f"Connecting to Redis at {redis_url}")

    # Create redis instance and store it in the app state
    # Synchrounous operation, only setting up parameters, no connection is established yet
    # `decode_responses=True` => command inputs/outputs are `str` instead of `bytes`.
    app.state.redis = Redis.from_url(redis_url, decode_responses=True)

    # Test connection
    # This will trigger the actual TCP connection, if the URL is wrong, it will error here
    try:
        await app.state.redis.ping()
        print("Redis connection established successfully")
    except Exception as e:
        print(f"Error connecting to Redis: {e}")

    yield

    print("Closing Redis connection...")
    await app.state.redis.close()
    print("Redis connection closed successfully")


app = FastAPI(lifespan=lifespan)

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
    max_age=86400,
)

app.include_router(transcribe_router)
app.include_router(api_key_router)
app.include_router(livez_router)
