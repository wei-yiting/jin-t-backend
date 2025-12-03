from fastapi import APIRouter
from openai import AsyncOpenAI, AuthenticationError
from app.models import (
    CheckIsOpenaiApiKeyValidResponse,
    CheckIsOpenaiApiKeyValidRequest,
)

router = APIRouter()


@router.post("/validate-openai-api-key")
async def validate_openai_api_key(
    request: CheckIsOpenaiApiKeyValidRequest,
) -> CheckIsOpenaiApiKeyValidResponse:
    """Checks if an OpenAI API key is valid by attempting to list models."""
    client = AsyncOpenAI(api_key=request.openai_api_key)
    try:
        await client.models.list()
        return CheckIsOpenaiApiKeyValidResponse(
            is_api_key_valid=True,
            has_unexpectied_validation_error=False,
        )
    except AuthenticationError:
        # AuthenticationError is raised when the API key is invalid
        return CheckIsOpenaiApiKeyValidResponse(
            is_api_key_valid=False,
            has_unexpectied_validation_error=False,
        )
    except Exception as e:
        # Handle other potential errors, e.g., network issues
        print(f"An unexpected error occurred when checking OpenAI API key: {e}")
        return CheckIsOpenaiApiKeyValidResponse(
            is_api_key_valid=False,
            has_unexpectied_validation_error=True,
        )
