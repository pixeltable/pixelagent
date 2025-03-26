# ============================================================================
# Pixelagent Basics Tutorial
# ============================================================================
# This tutorial demonstrates the fundamental capabilities of the Pixelagent
# framework, including agent creation, memory persistence, and tool integration.

import pixeltable as pxt
from pixelagent.openai import Agent

# ============================================================================
# SECTION 1: BASIC AGENT CREATION
# ============================================================================
# Create a simple conversational agent with a system prompt

# Initialize the agent with a name and personality
agent = Agent(
    agent_name="openai_agent",  # Unique identifier for this agent
    system_prompt="You're my assistant.",  # Defines agent personality and capabilities
    reset=True  # Start with fresh conversation history
)

# ============================================================================
# SECTION 2: CONVERSATIONAL MEMORY
# ============================================================================
# Demonstrate how agents maintain conversation context and history

# Basic conversation with persistent memory
print("Initial greeting:")
print(agent.chat("Hi, how are you?"))

# Agent remembers previous messages in the conversation
print("\nMemory test:")
print(agent.chat("What was my last question?"))

# ============================================================================
# SECTION 3: ACCESSING AGENT MEMORY
# ============================================================================
# How to retrieve and analyze the agent's conversation history

# Access the agent's memory table using the agent_name
memory = pxt.get_table("openai_agent.memory")

# Retrieve all conversation history
print("\nFull conversation memory:")
print(memory.collect())

# ============================================================================
# SECTION 4: AGENT WITH TOOLS
# ============================================================================
# Extend agent capabilities with custom Python functions as tools

# Define a simple weather tool as a user-defined function (UDF)
@pxt.udf
def weather(city: str) -> str:
    """
    Get the current weather for a specified city.
    
    Args:
        city (str): The name of the city to check weather for
        
    Returns:
        str: Weather description for the requested city
    """
    return f"The weather in {city} is sunny."

# Add tool to our exisitng agent with custom system prompt instructions
agent = Agent(
    agent_name="openai_agent", 
    system_prompt="Use your tools to answer the users questions.", 
    tools=pxt.tools(weather)
)

# ============================================================================
# SECTION 6: TOOL CALLING
# ============================================================================
# Demonstrate how agents can use tools to access external functionality

# Execute a tool call with a specific query
print("\nTool call test:")
print(agent.tool_call("Get weather in San Francisco"))

# ============================================================================
# SECTION 7: TOOL CALL OBSERVABILITY
# ============================================================================
# How to monitor and analyze the agent's tool usage

# Access the agent's tool call history
tools = pxt.get_table("openai_agent.tools")

# Retrieve all tool usage records
print("\nTool call history:")
print(tools.collect())