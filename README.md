# Pixelagent: An Agent Engineering Blueprint 

Pixelagent is a data-first agent framework for building AI agents, powered by Pixeltable's AI data infrastructure. It handles data orchestration, persistence, and multimodal support, letting you focus on agent logic.

## Key Features 

- **Automated Data Orchestration**: Built on Pixeltable's infrastructure for seamless data management
- **Native Multimodal**: Built-in support for text, images, and beyond
- **Declarative Model**: Define tables and columns; Pixeltable handles the rest
- **LLM Protocol Support**: Handles OpenAI and Anthropic message protocols
- **Tool Integration**: Built-in tool-call handshake system

## Quick Start 

```python
import pixeltable as pxt
from pixelagent.anthropic import Agent
import yfinance as yf

# ============================================================================
# SECTION 1: DEFINE A TOOL
# ============================================================================

@pxt.udf
def stock_price(ticker: str) -> dict:
    """
    Retrieve the current stock price for a given ticker symbol.
    
    Args:
        ticker (str): Stock ticker symbol (e.g., 'AAPL' for Apple)
        
    Returns:
        dict: Dictionary containing stock information and metrics
    """
    stock = yf.Ticker(ticker)
    return stock.info

# Register the tool with Pixeltable
tools = pxt.tools(stock_price)

# ============================================================================
# SECTION 2: CREATE AN AGENT
# ============================================================================

agent = Agent(
    agent_name="financial_analyst",  # Unique identifier for this agent
    system_prompt="You are a CFA working at a top-tier investment bank.",  # Agent personality
    tools=tools,  # Register tools with the agent
    reset=True,  # Start with a fresh conversation history
)

# ============================================================================
# SECTION 3: INTERACT WITH THE AGENT
# ============================================================================

# Basic conversation
print("--------------")
res = agent.chat("Hi, how are you?")
print(res)

# Tool calling test
print("--------------")
tool_result = agent.tool_call("Get NVIDIA and Apple stock price")
print(tool_result)

# Memory persistence test
print("--------------")
print(agent.chat("What was my last question?"))

# ============================================================================
# SECTION 4: ACCESS AGENT DATA
# ============================================================================

# Access the agent's conversation history
memory = pxt.get_table("financial_analyst.memory")
print("--------------")
print("Conversation Memory:")
print(memory.collect())

# Access the agent's tool call history
tools_log = pxt.get_table("financial_analyst.tools")
print("--------------")
print("Tool Call History:")
print(tools_log.collect())
```

## How It's Built

Want to see how Pixelagent's `Agent` class comes together? We've broken it down into simple, step-by-step blueprints for both Anthropic and OpenAI. These guides show you how to build an agent with just chat and tool-calling, leveraging Pixeltable's magic:

- **[Build with Anthropic](examples/build-your-own-agent/anthropic/README.md)**: Learn how we craft an agent using Claude, with cost-saving tricks like skipping chat history in tool calls.
- **[Build with OpenAI](examples/build-your-own-agent/openai/README.md)**: See how we use GPT models to create a lean, powerful agent with the same Pixeltable-driven efficiency.

Each guide starts with a minimal core and shows how Pixeltable handles persistence, orchestration, and updates—giving you a foundation to customize and extend.

## Tutorials and Examples

- **Basics**: Check out [Getting Started](examples/getting-started/pixelagent_basics_tutorial.py)for a step-by-step introduction to core concepts
- **Advanced Patterns**: Explore [Reflection](examples/reflection/anthropic/reflection.py) and [Planning](examples/planning/anthropic/react.py) for more complex agent architectures
- **Specialized Directories**: Browse our example directories for deeper implementations of specific techniques

## Common Extensions 

- **[Memory](examples/memory)**: Implement long-term memory systems with semantic search capabilities
- **[Knowledge](examples/knowledge)**: Build RAG systems with multimodal support
- **[Teams](examples/teams)**: Create multi-agent collaborative setups
- **[Reflection](examples/reflection)**: Add self-improvement loops
- **[Planning](examples/planning)**: Add planning loops
- **[Multimodal](examples/multimodal)**: Support images, videos, and other media types

## Why Choose Pixelagent? 

- **Data-First**: Focus on robust data management and persistence
- **Engineering Freedom**: Build exactly what you need without framework constraints
- **Simplified Workflow**: Automated handling of:
  - Data persistence and retrieval
  - LLM protocols
  - Tool integrations
  - State management

Ready to start building? Dive into the blueprints, tweak them to your needs, and let Pixelagent handle the infrastructure while you focus on innovation!