from datetime import datetime
import logging
from typing import List, Dict, Optional
import pixeltable as pxt


# Logging setup for any agent
def setup_logger(agent_name: str, verbose: bool = False) -> logging.Logger:
    """Set up logging for an agent."""
    logger = logging.getLogger(f"Agent.{agent_name}")
    logger.setLevel(logging.DEBUG if verbose else logging.WARNING)
    if not logger.handlers:
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    return logger


# Create memory table for an agent
def create_message_table(agent_name: str) -> pxt.Table:
    """Create a new message table for an agent.
    Args:
        agent_name: Name of the agent
    Returns:
        The created messages table
    """
    messages_table_name = f"{agent_name}_messages"
    return pxt.create_table(
        path_str=messages_table_name,
        schema_or_df={
            "role": pxt.String,
            "content": pxt.String,
            "timestamp": pxt.Timestamp,
        },
        if_exists="ignore",
    )


# Get messages for an agent
def get_messages(agent_name: str) -> List[Dict]:
    """Get the messages for an agent.

    Args:
        agent_name: The name of the agent
    """
    df = pxt.get_table(f"{agent_name}_messages").collect()
    msgs = [{"role": row["role"], "content": row["content"]} for row in df]
    return msgs


# Inject messages into the conversation history
def batch_inject_messages(messages_table, messages_to_inject: List[Dict]):
    """Inject multiple messages into the conversation history at once.

    Args:
        messages_to_inject: List of message dictionaries with 'role' and 'content' keys
    """
    # Validate message format
    for msg in messages_to_inject:
        if not all(key in msg for key in ["role", "content"]):
            raise ValueError("Each message must have 'role' and 'content' keys")

    # Insert all messages with current timestamp
    current_time = datetime.now()
    messages_table.insert(
        [
            {"role": msg["role"], "content": msg["content"], "timestamp": current_time}
            for msg in messages_to_inject
        ]
    )


# Search tool on a Pixeltable index
def create_search_tool(index: Optional[pxt.Table] = None):
    """Create a search tool that can be used by any agent.

    Args:
        index: Optional pixeltable index containing the data. If None, no search tool will be created.

    Returns:
        A pixeltable tool for searching data, or None if no index provided
    """
    if index is None:
        return None

    @pxt.query
    def search(query_text: str) -> str:
        """Search tool to find relevant passages."""
        similarity = index.text.similarity(query_text)
        return (
            index.order_by(similarity, asc=False)
            .select(index.text, sim=similarity)
            .limit(10)
        )

    return pxt.tools(search)


# Create messages with history and optional injected messages
@pxt.udf
def create_messages_with_history(
    past_context: List[Dict],
    system_prompt: str,
    current_prompt: str,
    injected_messages: List[Dict] = None,
) -> List[Dict]:
    """Create messages with history and optional injected messages.

    Args:
        past_context: Previous conversation history
        system_prompt: The system prompt
        current_prompt: Current user prompt
        injected_messages: Optional list of messages to inject before the current prompt

    Returns:
        List of messages including system prompt, history, injected messages, and current prompt
    """
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(
        [{"role": msg["role"], "content": msg["content"]} for msg in past_context]
    )

    # Insert injected messages if provided
    if injected_messages:
        messages.extend(injected_messages)

    messages.append({"role": "user", "content": current_prompt})
    return messages


# Create a prompt combining the question and search results
@pxt.udf
def tool_result_prompt(question: str, tool_result: list[dict]) -> str:
    """Create a prompt combining the question and search results.


    Args:
        question: The user's question
        tool_result: Results from the search tool

    Returns:
        A formatted prompt string
    """
    return f"""
    QUESTION:
    {question}
    
    TOOL RESULT:
    {tool_result}
    """
