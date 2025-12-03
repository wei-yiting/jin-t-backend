from datetime import datetime, timedelta
from zoneinfo import ZoneInfo


def get_formatted_retry_time_in_taipei_timezone(ttl_seconds: int) -> str:
    """
    Convert TTL seconds to readable retry time point in "HH:MM" or "Tomorrow HH:MM" format.
    """
    if ttl_seconds is None or ttl_seconds < 0:
        return "稍後"  # Error prevention

    # 1. Set timezone (Otherwise, Render will show UTC time)
    taipei_tz = ZoneInfo("Asia/Taipei")

    # 2. Get current time in Taipei timezone
    now = datetime.now(taipei_tz)

    # 3. Calculate future time point
    reset_time = now + timedelta(seconds=ttl_seconds)

    # 4. Check if it's across the day (if reset time's date != today's date)
    if reset_time.date() > now.date():
        # Format: Tomorrow 10:05
        return f"明天 {reset_time.strftime('%H:%M')}"
    else:
        # Format: 16:30 (Today)
        return reset_time.strftime("%H:%M")
