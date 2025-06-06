from typing import Optional

import PIL
import pixeltable as pxt


@pxt.udf
def create_content(
    memory_context: list[dict],
    current_message: str,
) -> str:
    
    # Build the conversation context as a text string without system prompt
    context = ""
    
    # Add memory context
    for msg in memory_context:
        context += f"{msg['role'].title()}: {msg['content']}\n"
    
    # Add current message
    context += f"User: {current_message}\n"
    context += "Assistant: "
    
    return context
