import json
from fastmcp import Client
from fastmcp.client.transports import SSETransport
from config import MCP_SERVER_URL

async def get_mcp_tools_as_json() -> str | None:
    """
    Fetches the tool list from the MCP server and formats it as a JSON string.
    """
    print(f"Attempting to connect to MCP server at {MCP_SERVER_URL} to get tools...")
    try:
        async with Client(transport=SSETransport(MCP_SERVER_URL)) as client:
            tool_list = await client.list_tools()
            print("Successfully fetched tool list from MCP server.")
            
            # Convert list of Tool objects to a list of dictionaries for JSON serialization
            tools_as_dicts = [
                {
                    "name": getattr(tool, 'name', 'N/A'),
                    "description": getattr(tool, 'description', ''),
                    "parameters": getattr(tool, 'input_schema', {})
                }
                for tool in tool_list
            ]
            return json.dumps(tools_as_dicts, indent=2)
            
    except Exception as e:
        print(f"FATAL ERROR: Could not connect to MCP server or fetch tools: {e}")
        print("Please ensure the MCP server is running and accessible.")
        return None

# def create_system_prompt(tool_list_json: str) -> str:
#     """Creates the detailed system prompt for the LLM, including the tool list."""
#     return f"""
# You are a tool-calling assistant. Your only job is to convert a user's request into a JSON array of tool calls based on the tools provided.

# **RULES:**
# 1.  **JSON ONLY:** Your entire response must be a valid JSON array. Do not add any other text, explanations, or greetings.
# 2.  **CORRECT FORMAT:** The array must contain one or more objects. Each object must have a "name" key (the tool name as a string) and an "arguments" key (an object with the parameters).
# 3.  **USE PROVIDED TOOLS:** Only use the exact tool names available in the `<tools>` list. Do not make up tools.
# 4.  **BE CONCISE:** If the user request is simple, only use the necessary tools.

# <tools>
# {tool_list_json}
# </tools>

# **EXAMPLE:**
# User request: open the youtube APP.
# Your response:
# [
#   {{"name": "mobile_use_default_device", "arguments": {{}}}},
#   {{"name": "mobile_launch_app", "arguments": {{"packageName": "com.google.android.youtube"}}}}
# ]
# """