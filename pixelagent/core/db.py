import pixeltable as pxt

def setup_pixeltable(name: str, tool_calls_table: bool = False, reset: bool = False):
    tables = [i for i in pxt.list_tables() if i.startswith(name)]
    if reset or len(tables) == 0:
        pxt.drop_dir(name, force=True)
        pxt.create_dir(name)

        messages = pxt.create_table(
            f"{name}.messages",
            {
                "message_id": pxt.Int,
                "system_prompt": pxt.String,
                "user_input": pxt.String,
                "response": pxt.String,                
                "role": pxt.String,
                "content": pxt.String,
                "attachments": pxt.String,
                "timestamp": pxt.Timestamp,
            },
        )

        if tool_calls_table:
            tool_calls = pxt.create_table(
                f"{name}.tool_calls",
                {
                    "tool_call_id": pxt.String,
                    "message_id": pxt.Int,
                    "tool_name": pxt.String,
                    "arguments": pxt.Json,
                    "result": pxt.String,
                    "timestamp": pxt.Timestamp,
                },
            )
    else:
        messages = pxt.get_table(f"{name}.messages")
        if tool_calls_table:
            tool_calls = pxt.get_table(f"{name}.tool_calls")

    if tool_calls_table:
        return messages, tool_calls
    else:
        return messages, None