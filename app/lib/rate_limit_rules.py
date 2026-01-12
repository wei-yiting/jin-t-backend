from app.models import RateLimitRule
from app.config import (
    FREE_TIER_SINGLE_DEVICE_24H_MAX_TOTAL_AUDIO_DURATION_SECONDS,
    FREE_TIER_SINGLE_IP_24H_MAX_TOTAL_AUDIO_DURATION_SECONDS,
    FREE_TIER_SINGLE_DEVICE_1H_MAX_TRANSCRIBE_COUNT,
    FREE_TIER_SINGLE_IP_1H_MAX_TRANSCRIBE_COUNT,
)


def get_rate_limit_rules(device_id: str, real_ip: str) -> list[RateLimitRule]:
    return [
        # Device 1H
        RateLimitRule(
            key=f"usage:device:1h:{device_id}",
            window_ttl=60 * 60,
            max_count=FREE_TIER_SINGLE_DEVICE_1H_MAX_TRANSCRIBE_COUNT,
            error_count_msg="每小時最多 {limit} 次",
        ),
        # Device 24H
        RateLimitRule(
            key=f"usage:device:24h:{device_id}",
            window_ttl=24 * 60 * 60,
            max_duration=FREE_TIER_SINGLE_DEVICE_24H_MAX_TOTAL_AUDIO_DURATION_SECONDS,
            error_duration_msg="每 24 小時最多 {limit} 分鐘",
        ),
        # IP 1H
        RateLimitRule(
            key=f"usage:ip:1h:{real_ip}",
            window_ttl=60 * 60,
            max_count=FREE_TIER_SINGLE_IP_1H_MAX_TRANSCRIBE_COUNT,
            error_count_msg="每小時最多 {limit} 次",
        ),
        # IP 24H
        RateLimitRule(
            key=f"usage:ip:24h:{real_ip}",
            window_ttl=24 * 60 * 60,
            max_duration=FREE_TIER_SINGLE_IP_24H_MAX_TOTAL_AUDIO_DURATION_SECONDS,
            error_duration_msg="每 24 小時最多 {limit} 分鐘",
        ),
    ]
