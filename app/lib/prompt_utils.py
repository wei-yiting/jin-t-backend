import os


def read_prompt(prompt_path: str) -> str:
    full_prompt_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), prompt_path
    )

    try:
        with open(full_prompt_path, "r") as file:
            return file.read()
    except FileNotFoundError:
        print(f"Error: Prompt file not found: {prompt_path}")
    except Exception as e:
        print(f"Error: Unknown error when reading Prompt file: {e}")
    return ""
