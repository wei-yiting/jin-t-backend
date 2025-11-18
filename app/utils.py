import re

from app.config import (
    FINAL_PUNCS,
    FULL_WIDTH_PUNCS,
    HALF_WIDTH_PUNCS,
    HEALTHY_SEMANTIC_UNITS_PER_PUNC_THRESHOLD,
    SHORT_TEXT_EXEMPTION_THRESHOLD,
)


def add_spacing_between_chinese_english(text: str) -> str:
    """
    Add spacing between Chinese and English/numbers/English punctuation marks.
    """
    # Pattern 1: Chinese followed by English/numbers
    text = re.sub(r"([\u4e00-\u9fa5])([a-zA-Z0-9])", r"\1 \2", text)

    # Pattern 2: English/numbers followed by Chinese
    text = re.sub(r"([a-zA-Z0-9])([\u4e00-\u9fa5])", r"\1 \2", text)

    # Pattern 3: English punctuation marks followed by Chinese
    text = re.sub(r"([.,!?;:)\]}'\"])([\u4e00-\u9fa5])", r"\1 \2", text)

    return text


def check_transcript_punctuation_health(text: str) -> bool:
    """
    Use heuristic rules to check if the punctuation is "healthy".

    - "Healthy" (True): Punctuation is normal.
    - "Unhealthy" (False): Likely to miss punctuation, needs to be remedied.
    """

    clean_text = text.strip()
    if not clean_text:
        return True

    # Calculate the number of Chinese characters and English words
    chinese_char_count = len(re.findall(r"[\u4e00-\u9fa5]", clean_text))
    english_word_count = len(re.findall(r"[a-zA-Z0-9]+", clean_text))
    semantic_units_count = chinese_char_count + english_word_count

    # Rule 1: Short text exemption
    if semantic_units_count < SHORT_TEXT_EXEMPTION_THRESHOLD:
        return True

    # Rule 2: Ending check
    if clean_text[-1] not in FINAL_PUNCS:
        return False

    # Rule 3: Punctuation ratio check
    all_puncs = FULL_WIDTH_PUNCS.union(HALF_WIDTH_PUNCS)
    punctuation_count = sum(1 for char in clean_text if char in all_puncs)

    if punctuation_count == 0:
        return False
    semantic_units_per_punc = semantic_units_count / punctuation_count

    if semantic_units_per_punc > HEALTHY_SEMANTIC_UNITS_PER_PUNC_THRESHOLD:
        return False

    return True


def read_prompt(prompt_path: str) -> str:
    try:
        with open(prompt_path, "r") as file:
            return file.read()
    except FileNotFoundError:
        print(f"Error: Prompt file not found: {prompt_path}")
    except Exception as e:
        print(f"Error: Unknown error when reading Prompt file: {e}")
    return ""
