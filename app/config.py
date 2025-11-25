# Model names
TRANSCRIBE_MODEL_NAME = "gpt-4o-mini-transcribe"
PUNC_FIX_MODEL_NAME = "gpt-4.1-nano"
REFINE_MODEL_NAME = "gpt-4.1-nano"

# Model temperatures
PUNC_FIX_TEMPERATURE = 0.0
REFINE_TEMPERATURE = 0.0

# Threshold
SHORT_TEXT_EXEMPTION_THRESHOLD = 10
HEALTHY_SEMANTIC_UNITS_PER_PUNC_THRESHOLD = 20

# Prompt file paths
TRANSCRIBE_PROMPT_FILE_PATH = "prompts/transcribe_prompt.txt"
PUNC_FIX_PROMPT_FILE_PATH = "prompts/punc_fix_instruction.txt"
TRANSCRIPT_REFINE_PROMPT_FILE_PATH = "prompts/transcript_refine_instruction.txt"

# Punctuation symbols
FULL_WIDTH_PUNCS = {"。", "，", "？", "！", "、", "；", "："}
HALF_WIDTH_PUNCS = {".", ",", "?", "!", ";", ":"}
FINAL_PUNCS = {"。", "？", "！", ".", "?", "!", "..."}
