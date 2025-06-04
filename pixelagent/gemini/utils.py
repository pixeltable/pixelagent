import base64
import io
from typing import Optional

import PIL
import pixeltable as pxt


@pxt.udf
def create_content(
    system_prompt: str,
    memory_context: list[dict],
    current_message: str,
) -> str:
    
    # Build the conversation context as a text string
    context = f"System: {system_prompt}\n\n"
    
    # Add memory context
    for msg in memory_context:
        context += f"{msg['role'].title()}: {msg['content']}\n"
    
    # Add current message
    context += f"User: {current_message}\n"
    context += "Assistant: "
    
    return context
