from datetime import datetime
import pixeltable as pxt
from pixeltable.functions import openai
from typing import List, Dict, Optional, Any
import logging

from .utils import (
    setup_logger,
    batch_inject_messages,
    create_messages_with_history,
    create_search_tool,
    tool_result_prompt,
)

logger = logging.getLogger("OpenAIAgent")
logger.setLevel(logging.DEBUG)

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

    # Query to get message history
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
            interpret_tool_result=tool_result_prompt(
                agent_table.prompt, agent_table.tool_result
            )
        )

        # Set up final response with tool results
        result_messages = [
            {
                "role": "system",
                "content": system_prompt,
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
