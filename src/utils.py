

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

def create_system_prompt(tool_list_json: str) -> str:
    """Creates the detailed system prompt for the LLM, including the tool list."""
    return f"""
You are a tool-calling assistant. Your only job is to convert a user's request into a JSON array of tool calls based on the tools provided.

**RULES:**
1.  **JSON ONLY:** Your entire response must be a valid JSON array. Do not add any other text, explanations, or greetings.
2.  **CORRECT FORMAT:** The array must contain one or more objects. Each object must have a "name" key (the tool name as a string) and an "arguments" key (an object with the parameters).
3.  **USE PROVIDED TOOLS:** Only use the exact tool names available in the `<tools>` list. Do not make up tools.
4.  **BE CONCISE:** If the user request is simple, only use the necessary tools.

<tools>
{tool_list_json}
</tools>

**EXAMPLE:**
User request: open the youtube APP.
Your response:
[
  {{"name": "mobile_use_default_device", "arguments": {{}}}},
  {{"name": "mobile_launch_app", "arguments": {{"packageName": "com.google.android.youtube"}}}}
]
"""