import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import pixeltable as pxt
from pixeltable.functions import openai

logger = logging.getLogger("OpenAIAgent")
logger.setLevel(logging.DEBUG)


def setup_logger(agent_name: str, verbose: bool = False) -> logging.Logger:
    """Set up logging for an agent."""
    logger = logging.getLogger(f"OpenAIAgent.{agent_name}")
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


def init(
    agent_name: str,
    system_prompt: str,
    model_name: str = "gpt-4o-mini",
    agent_tools: Any = None,
    index: Optional[pxt.Table] = None,
    reset_memory: bool = False,
    verbose: bool = False,
) -> None:
    """Initialize or get existing agent tables."""

    logger = setup_logger(agent_name, verbose)

    logger.debug(f"Initializing tables for agent: {agent_name}")

    # Delete agent tables if requested
    if reset_memory:
        logger.info(f"Purging existing tables for agent: {agent_name}")
        pxt.drop_table(f"{agent_name}_messages", if_not_exists="ignore", force=True)
        pxt.drop_table(agent_name, if_not_exists="ignore", force=True)

    # Create or get messages table
    messages_table_name = f"{agent_name}_messages"
    if messages_table_name not in pxt.list_tables():
        logger.debug(f"Creating new messages table: {messages_table_name}")
        messages_table = pxt.create_table(
            path_str=messages_table_name,
            schema_or_df={
                "role": pxt.String,
                "content": pxt.String,
                "timestamp": pxt.Timestamp,
            },
            if_exists="ignore",
        )
    else:
        messages_table = pxt.get_table(messages_table_name)

    # Get existing or create new agent table
    if agent_name not in pxt.list_tables():
        agent_table = pxt.create_table(
            path_str=agent_name,
            schema_or_df={
                "prompt": pxt.String,
            },
            if_exists="ignore",
        )

        # Use provided tools or create search tool if index exists
        tools = agent_tools
        if tools is None and index is not None:
            tools = create_search_tool(index)

        setup_agent_table(agent_table, tools, system_prompt, messages_table, model_name)


def run(
    agent_name: str, message: str, additional_context: Optional[List[Dict]] = None
) -> str:
    """Run the agent with optional message injection."""

    logger.info(
        f"Processing message: '{message[:50]}{'...' if len(message) > 50 else ''}'"
    )

    # Get the latest tables
    agent_table = pxt.get_table(agent_name)
    messages_table = pxt.get_table(f"{agent_name}_messages")

    # Inject additional messages if provided
    if additional_context:
        logger.debug(f"Injecting {len(additional_context)} context messages")
        batch_inject_messages(messages_table, additional_context)

    # Store user message in memory
    logger.debug("→ Storing user message")
    messages_table.insert(
        [{"role": "user", "content": message, "timestamp": datetime.now()}]
    )

    # Process through chat session
    logger.debug("→ Processing through agent")
    agent_table.insert([{"prompt": message}])

    # Get response
    result = agent_table.select(agent_table.answer).tail(1)
    response = result["answer"][0]

    # Store assistant response in memory
    logger.debug("← Storing assistant response")
    messages_table.insert(
        [{"role": "assistant", "content": response, "timestamp": datetime.now()}]
    )

    logger.info("✓ Processing complete")
    return response


def create_search_tool(index=None):
    """Create a search tool that can be used by the agent.

    Args:
        index: Optional pixeltable index containing the data. If None, no search tool will be created.

    Returns:
        A pixeltable tool for searching data, or None if no index provided
    """
    if index is None:
        return None

    @pxt.query
    def search(query_text: str) -> str:
        """Search tool to find relevant passages.

        Args:
            query_text: The search query
        Returns:
            Top 10 most relevant passages
        """
        similarity = index.text.similarity(query_text)
        return (
            index.order_by(similarity, asc=False)
            .select(index.text, sim=similarity)
            .limit(10)
        )

    return pxt.tools(search)


# Create messages with history
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


def setup_agent_table(
    agent_table: pxt.Table,
    agent_tools: pxt.tools,
    system_prompt: str,
    messages_table: pxt.Table,
    model_name: str,
):
    """Set up the agent table for tool use with message history.

    Args:
        agent_table: The pixeltable table to set up
        agent_tools: The tools available to the agent, or None if no tools
        system_prompt: The system prompt for the agent
        messages_table: Table storing conversation history
        model_name: Name of the model to use
    """

    # Query to get recent message history
    @pxt.query
    def fetch_messages():
        return (
            messages_table.order_by(messages_table.timestamp, asc=False).select(
                role=messages_table.role, content=messages_table.content
            )
            # .limit(10)
        )

    # Add message history column
    agent_table.add_computed_column(messages_context=fetch_messages())

    # Add computed columns in sequence
    agent_table.add_computed_column(
        messages=create_messages_with_history(
            agent_table.messages_context, system_prompt, agent_table.prompt
        )
    )

    if agent_tools is not None:
        # Add tool-based response generation if tools are provided
        agent_table.add_computed_column(
            tool_response=openai.chat_completions(
                model=model_name,
                messages=agent_table.messages,
                tools=agent_tools,
            )
        )

        agent_table.add_computed_column(
            tool_result=openai.invoke_tools(agent_tools, agent_table.tool_response)
        )

        agent_table.add_computed_column(
            interpret_tool_result=create_prompt(
                agent_table.prompt, agent_table.tool_result
            )
        )

        # Set up final response with tool results
        result_messages = [
            {
                "role": "system",
                "content": "Answer the user's question from the tool result.",
            },
            {"role": "user", "content": agent_table.interpret_tool_result},
        ]
    else:
        # Direct response generation without tools
        result_messages = agent_table.messages

    agent_table.add_computed_column(
        final_response=openai.chat_completions(
            model=model_name, messages=result_messages
        )
    )

    agent_table.add_computed_column(
        answer=agent_table.final_response.choices[0].message.content
    )


@pxt.udf
def create_prompt(question: str, tool_result: list[dict]) -> str:
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


def get_messages(agent_name: str) -> List[Dict]:
    """Get the messages for an agent.

    Args:
        agent_name: The name of the agent
    """
    df = pxt.get_table(f"{agent_name}_messages").collect()
    msgs = [{"role": row["role"], "content": row["content"]} for row in df]
    return msgs
