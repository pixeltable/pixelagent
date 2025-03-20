import pixeltable as pxt


@pxt.udf
def create_messages(memory_context: list[dict], current_message: str) -> list[dict]:
    messages = memory_context.copy()
    messages.append({"role": "user", "content": current_message})
    return messages
