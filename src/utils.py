

def clean_whisper_output(text: str) -> str:
    """
    Cleans the raw output from whisper.cpp by removing timestamps, metadata,
    and extra punctuation.
    """
    if not text:
        return ""

    # Remove common whisper.cpp artifacts
    text = text.replace("[BLANK_AUDIO]", "").replace("[SOUND]", "")

    cleaned_lines = []
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue

        # Remove timestamp lines like [00:00:00.000 --> 00:00:01.000]
        if line.startswith('[') and '-->' in line and ']' in line:
            parts = line.split(']', 1)
            line = parts[1].strip() if len(parts) > 1 else ""

        # Clean up trailing punctuation that can interfere with sentence structure
        if line.endswith('.'):
            line = line[:-1]
        if line.endswith('...'):
            line = line[:-3]

        if line:
            cleaned_lines.append(line)

    cleaned_text = " ".join(cleaned_lines).strip()
    return cleaned_text