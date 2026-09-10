from typing import Optional

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


@pxt.udf
def merge_system_instruction(
    system_prompt: str, model_kwargs: Optional[dict] = None
) -> dict:
    """Gemini takes the system prompt as `system_instruction` inside `config`."""
    return {"system_instruction": system_prompt, **(model_kwargs or {})}
