import os

# Model names
TRANSCRIBE_MODEL_NAME = "gpt-4o-mini-transcribe"
PUNC_FIX_MODEL_NAME = "gpt-4.1-nano"
REFINE_MODEL_NAME = "gpt-4.1-nano"
CHUNKS_CONSOLIDATION_MODEL_NAME = "gpt-4.1-nano"

# Model temperatures
PUNC_FIX_TEMPERATURE = 0.0
REFINE_TEMPERATURE = 0.0
CHUNKS_CONSOLIDATION_TEMPERATURE = 0.0

# Threshold
SHORT_TEXT_EXEMPTION_THRESHOLD = 10
HEALTHY_SEMANTIC_UNITS_PER_PUNC_THRESHOLD = 20

# Prompt file paths
TRANSCRIBE_PROMPT_FILE_PATH = "prompts/transcribe_prompt.txt"
PUNC_FIX_PROMPT_FILE_PATH = "prompts/punc_fix_instruction.txt"
TRANSCRIPT_REFINE_PROMPT_FILE_PATH = "prompts/transcript_refine_instruction.txt"
CHUNKS_CONSOLIDATION_PROMPT_FILE_PATH = "prompts/chunks_consolidation_instruction.txt"

# Punctuation symbols
FULL_WIDTH_PUNCS = {"。", "，", "？", "！", "、", "；", "："}
HALF_WIDTH_PUNCS = {".", ",", "?", "!", ";", ":"}
FINAL_PUNCS = {"。", "？", "！", ".", "?", "!", "..."}

# Validation limits
MAX_AUDIO_FILE_SIZE_MB = 25

# Free tier limits
FREE_TIER_SINGLE_AUDIO_MAX_DURATION_SECONDS = 30 * 60  # 30 minutes
FREE_TIER_SINGLE_DEVICE_24H_MAX_TOTAL_AUDIO_DURATION_SECONDS = 30 * 60  # 30 minutes
FREE_TIER_SINGLE_IP_24H_MAX_TOTAL_AUDIO_DURATION_SECONDS = 60 * 60  # 60 minutes
FREE_TIER_SINGLE_DEVICE_1H_MAX_TRANSCRIBE_COUNT = 5
FREE_TIER_SINGLE_IP_1H_MAX_TRANSCRIBE_COUNT = (
    20  # higher than device limit in case multiple users use the same IP
)

# Transcribe task
TRANSCRIBE_STREAM_EVENT_TTL = 30 * 60  # 30 minutes
TEMP_AUDIO_FILES_DIR = "/tmp/audio_processing"
MAX_CONCURRENT_TRANSCRIBE_WORKERS = 3
INITIAL_CONCURRENT_CHUNK_DURATIONS_SECONDS = [120, 180, 240]
SUBSEQUENT_CHUNK_DURATION_SECONDS = 180
AUDIO_CHUNK_OVERLAP_MS = 5000
# Audio that fits inside the first chunk is transcribed in a single call:
# slicing it would re-encode the whole file to produce one chunk, and
# consolidation would run with no boundaries to stitch.
SHORT_AUDIO_MAX_DURATION_MS = INITIAL_CONCURRENT_CHUNK_DURATIONS_SECONDS[0] * 1000

# The transcription API infers the container format from the filename it is
# given, so a stored upload must keep the extension it arrived with. Anything
# outside this set is rejected rather than guessed at.
# https://platform.openai.com/docs/guides/speech-to-text
SUPPORTED_AUDIO_EXTENSIONS = frozenset(
    {".flac", ".m4a", ".mp3", ".mp4", ".mpeg", ".mpga", ".oga", ".ogg", ".wav", ".webm"}
)

os.makedirs(TEMP_AUDIO_FILES_DIR, exist_ok=True)
