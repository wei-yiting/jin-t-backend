def read_prompt(prompt_path: str) -> str:
    try:
        with open(prompt_path, "r") as file:
            return file.read()
    except FileNotFoundError:
        print(f"Error: Prompt file not found: {prompt_path}")
    except Exception as e:
        print(f"Error: Unknown error when reading Prompt file: {e}")
    return ""
