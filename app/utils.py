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
