"""FastAPI dependencies for transcribe endpoint validation."""

import os
from typing import Annotated
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import padding
import base64

from fastapi import Header, HTTPException, Request, UploadFile, Depends, File
from redis.asyncio import Redis

from app.config import (
    FREE_TIER_SINGLE_AUDIO_MAX_DURATION_SECONDS,
    MAX_AUDIO_FILE_SIZE_MB,
)
from app.lib.retry_time_generator import get_formatted_retry_time_in_taipei_timezone
from app.lib.audio_tools import parse_audio_duration
from app.lib.rate_limit_rules import get_rate_limit_rules
from app.models import ValidatedAudioFile, UsageConfig, CheckIsOpenaiApiKeyValidRequest

async def validate_audio_file(
    audio_file: Annotated[UploadFile, File()],
) -> ValidatedAudioFile:
    """Validate audio file filename and minimum size (100 bytes)."""
    # Validate audio file
    if not audio_file.filename:
        raise HTTPException(status_code=400, detail="No audio file provided")

    # Check file size (must be > 100 bytes for valid audio)
    file_content = await audio_file.read()
    file_size_bytes = len(file_content)

    if file_size_bytes < 100:
        raise HTTPException(
            status_code=400, detail="Audio file is too small or corrupted"
        )

    file_size_mb = file_size_bytes / (1024 * 1024)

    return ValidatedAudioFile(
        file=audio_file, file_size_bytes=file_size_bytes, file_size_mb=file_size_mb
    )


async def validate_file_size(
    validated_audio: Annotated[ValidatedAudioFile, Depends(validate_audio_file)],
) -> ValidatedAudioFile:
    """Validate file size does not exceed 25MB limit."""
    if validated_audio.file_size_mb > MAX_AUDIO_FILE_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File size exceeds the maximum limit of {MAX_AUDIO_FILE_SIZE_MB}MB.",
        )

    # Reset file pointer for processing
    await validated_audio.file.seek(0)

    return validated_audio


def get_usage_config(
    encrypted_openai_api_key: Annotated[str, Header(alias="X-Custom-Openai-Api-Key")],
    consent_data_collection: Annotated[str, Header(alias="X-Consent-Data-Collection")],
) -> UsageConfig:
    """Determine API key configuration based on headers."""
    has_personal_openai_api_key = (
        encrypted_openai_api_key and encrypted_openai_api_key.strip() != ""
    )
    has_consent = bool(
        consent_data_collection and consent_data_collection.lower() == "true"
    )

    # Determine if using free tier
    if not has_personal_openai_api_key and not has_consent:
        # Scenario 1: No API key AND no consent
        raise HTTPException(
            status_code=403,
            detail="When not providing a custom API key, you must consent to data collection. If you do not consent to data collection, please provide your own OpenAI API key.",
        )

    if has_personal_openai_api_key:
        # Scenario 2: Custom API key provided
        using_free_tier = False
        api_key_to_use = core_decode_and_decrypt_openai_api_key(encrypted_openai_api_key)
    else:
        # Scenario 3: No API key AND consent given (free tier)
        using_free_tier = True
        api_key_to_use = os.getenv("FREE_TIER_OPENAI_API_KEY", "")
        if not api_key_to_use:
            raise HTTPException(
                status_code=500,
                detail="Free tier OpenAI API key is not configured on the server.",
            )

    return UsageConfig(
        api_key_to_use=api_key_to_use,
        using_free_tier=using_free_tier,
        consent_data_collection=has_consent,
    )


def validate_audio_duration(
    audio_duration: Annotated[str, Header(alias="X-Audio-Duration")],
    usage_config: Annotated[UsageConfig, Depends(get_usage_config)],
) -> str:
    """Validate free tier single audio duration does not exceed configured limit."""
    if usage_config.using_free_tier and audio_duration:
        try:
            duration_seconds = float(audio_duration)
            if duration_seconds > FREE_TIER_SINGLE_AUDIO_MAX_DURATION_SECONDS:
                raise HTTPException(
                    status_code=400,
                    detail=f"Audio duration exceeds the maximum limit of {FREE_TIER_SINGLE_AUDIO_MAX_DURATION_SECONDS // 60} minutes for free tier usage.",
                )
        except ValueError:
            # Invalid duration format, continue without validation
            pass

    return audio_duration


def get_redis_client(request: Request) -> Redis:
    return request.app.state.redis


def get_real_ip_address(request: Request) -> str:
    cf_connecting_ip = request.headers.get("CF-Connecting-IP")
    if cf_connecting_ip:
        return cf_connecting_ip

    x_forwarded_for = request.headers.get("X-Forwarded-For")
    if x_forwarded_for:
        return x_forwarded_for.split(",")[0].strip()

    return request.client.host if request.client else ""


async def check_and_update_rate_limit(
    device_id: Annotated[str, Header(alias="X-Device-Id")],
    usage_config: Annotated[UsageConfig, Depends(get_usage_config)],
    audio_duration: Annotated[str, Depends(validate_audio_duration)],
    redis: Annotated[Redis, Depends(get_redis_client)],
    real_ip: Annotated[str, Depends(get_real_ip_address)],
) -> str:
    """
    Check and update limit from Redis for free tier.
    Rate limit is per device ID and IP.
    If not using free tier, return device ID directly.
    If using free tier, check and update limit from Redis using device ID as key.
    """
    if not usage_config.using_free_tier:
        return device_id

    parsed_duration = parse_audio_duration(audio_duration)
    rules = get_rate_limit_rules(device_id, real_ip)

    # Create a dict to store the fetched data: { rule_key: (data_dict, ttl) }
    fetched_data = {}

    async with redis.pipeline() as pipe:
        for rule in rules:
            pipe.hgetall(rule.key)
            pipe.ttl(rule.key)

        results = await pipe.execute()

        # Parse results: results is [hgetall_1, ttl_1, hgetall_2, ttl_2, ...]
        for i, rule in enumerate(rules):
            data = results[i * 2]  # even index is data
            ttl = results[i * 2 + 1]  # odd index is TTL
            fetched_data[rule.key] = (data, ttl)

    # Unified validation logic (Validation Phase)
    for rule in rules:
        data, ttl = fetched_data[rule.key]
        current_count = int(data.get("transcribe_count", 0)) if data else 0
        current_duration = float(data.get("total_duration", 0.0)) if data else 0.0
        retry_time_str = get_formatted_retry_time_in_taipei_timezone(ttl)

        if rule.max_count is not None and current_count >= rule.max_count:
            count_limit_rule_str = rule.error_count_msg.format(limit=rule.max_count)
            raise HTTPException(
                status_code=429,
                detail=f"達到{count_limit_rule_str}轉錄的限制，請{retry_time_str}後再試一次",
            )

        # B. 檢查時長
        if rule.max_duration is not None and current_duration + parsed_duration > rule.max_duration:
            limit_min = rule.max_duration // 60
            duration_limit_rule_str = rule.error_duration_msg.format(limit=limit_min)
            raise HTTPException(
                status_code=429,
                detail=f"達到{duration_limit_rule_str}的限制，請{retry_time_str}後再試一次",
            )

    # Unified update logic (Update Phase)
    async with redis.pipeline() as pipe:
        for rule in rules:
            data, _ = fetched_data[rule.key]

            # 24h doesn't have count limit, so only update count for 1h
            if rule.max_count is not None:
                pipe.hincrby(rule.key, "transcribe_count", 1)

            pipe.hincrbyfloat(rule.key, "total_duration", parsed_duration)

            is_new_key = not data
            if is_new_key:
                pipe.expire(rule.key, rule.window_ttl)

        await pipe.execute()

    return device_id


_private_key_str = os.getenv("API_KEY_ENCRYPTION_PRIVATE_KEY", "")

def core_decode_and_decrypt_openai_api_key(encrypted_openai_api_key: str) -> str:
    try:
        # 1. Prepare the private key
        if not _private_key_str:
            raise HTTPException(
                status_code=500,
                detail="Private key is not configured on the server.",
            )
            
        formatted_key_str = _private_key_str.replace('\\n', '\n')
        private_key_bytes = formatted_key_str.encode('utf-8')
        private_key = serialization.load_pem_private_key(
            private_key_bytes,
            password=None,
            backend=default_backend()
        )
        
        if not isinstance(private_key, rsa.RSAPrivateKey):
            raise HTTPException(
                status_code=500,
                detail="Loaded private key is not RSA format, cannot decrypt.",
            )
        
        # 2. Base64 Decode
        encrypted_bytes = base64.b64decode(encrypted_openai_api_key)
        
        # 3. RSA Decrypt
        decrypted_bytes = private_key.decrypt(
            encrypted_bytes,
            padding.PKCS1v15()
        )
        
        decrypted_openai_api_key = decrypted_bytes.decode('utf-8')
        return decrypted_openai_api_key
    
    except HTTPException:
        raise

    except ValueError:
        raise HTTPException(
            status_code=400, 
            detail="Invalid encryption format (Base64 error)"
        )
    except Exception:
        # Decryption failed (Key is wrong) or other errors
        raise HTTPException(
            status_code=401, 
            detail="Decryption failed. Invalid security credentials."
        )


def get_openai_api_key_from_request_body(
    request_body: CheckIsOpenaiApiKeyValidRequest
) -> str:
    """Get the OpenAI API key from the request body."""
    return core_decode_and_decrypt_openai_api_key(request_body.encrypted_openai_api_key)