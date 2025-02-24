import pixeltable as pxt

def setup_pixeltable(name: str, reset: bool = False):
    tables = [i for i in pxt.list_tables() if i.startswith(name)]
    if reset or len(tables) == 0:
        pxt.drop_dir(name, force=True)
        pxt.create_dir(name)

        messages = pxt.create_table(
            f"{name}.messages",
            {
                "message_id": pxt.IntType(),
                "role": pxt.StringType(),
                "content": pxt.StringType(),
                "attachments": pxt.StringType(nullable=True),
                "timestamp": pxt.TimestampType(),
            },
            primary_key="message_id",
        )

        chat = pxt.create_table(
            f"{name}.chat",
            {
                "system_prompt": pxt.StringType(),
                "user_input": pxt.StringType(),
                "response": pxt.StringType(nullable=True),
            },
        )

        tool_calls = pxt.create_table(
            f"{name}.tool_calls",
            {
                "tool_call_id": pxt.StringType(),
                "message_id": pxt.IntType(),
                "tool_name": pxt.StringType(),
                "arguments": pxt.JsonType(),
                "result": pxt.StringType(nullable=True),
                "timestamp": pxt.TimestampType(),
            },
        )
    else:
        messages = pxt.get_table(f"{name}.messages")
        chat = pxt.get_table(f"{name}.chat")
        tool_calls = pxt.get_table(f"{name}.tool_calls")

    return messages, chat, tool_calls