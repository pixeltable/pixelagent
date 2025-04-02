import pixeltable as pxt


@pxt.udf
def create_messages(
    system_prompt: str, memory_context: list[dict], current_message: str
) -> list[dict]:
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(memory_context.copy())
    messages.append({"role": "user", "content": current_message})
    return messages
