import json
import os
import re


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


# Punctuation symbols
FULL_WIDTH_PUNCS = {"。", "，", "？", "！", "、", "；", "："}
HALF_WIDTH_PUNCS = {".", ",", "?", "!", ";", ":"}
FINAL_PUNCS = {"。", "？", "！", ".", "?", "!", "..."}
ALL_PUNCS = FULL_WIDTH_PUNCS.union(HALF_WIDTH_PUNCS)
HEALTHY_SEMANTIC_UNITS_PER_PUNC_THRESHOLD = 20


def check_transcript_punctuation_health(text: str) -> bool:
    """
    Use heuristic rules to check if the punctuation is "healthy".

    - "Healthy" (True): Punctuation is normal.
    - "Unhealthy" (False): Likely to miss punctuation, needs to be remedied.
    """

    clean_text = text.strip()
    if not clean_text:
        return True

    full_width_puncs = set(json.loads(os.environ.get("FULL_WIDTH_PUNCS", "[]")))
    half_width_puncs = set(json.loads(os.environ.get("HALF_WIDTH_PUNCS", "[]")))
    final_punc = set(json.loads(os.environ.get("FINAL_PUNCS", "[]")))
    all_puncs = full_width_puncs.union(half_width_puncs)
    healthy_semantic_units_per_punc_threshold = int(
        os.environ.get("HEALTHY_SEMANTIC_UNITS_PER_PUNC_THRESHOLD", "50")
    )
    short_text_exemption_threshold = int(
        os.environ.get("SHORT_TEXT_EXEMPTION_THRESHOLD", "5")
    )

    # Calculate the number of Chinese characters and English words
    chinese_char_count = len(re.findall(r"[\u4e00-\u9fa5]", clean_text))
    english_word_count = len(re.findall(r"[a-zA-Z0-9]+", clean_text))
    semantic_units_count = chinese_char_count + english_word_count

    # Rule 1: Short text exemption
    if semantic_units_count < short_text_exemption_threshold:
        return True

    # Rule 2: Ending check
    if clean_text[-1] not in final_punc:
        return False

    # Rule 3: Punctuation ratio check
    punctuation_count = sum(1 for char in clean_text if char in all_puncs)

    if punctuation_count == 0:
        return False
    semantic_units_per_punc = semantic_units_count / punctuation_count

    if semantic_units_per_punc > healthy_semantic_units_per_punc_threshold:
        return False

    return True
