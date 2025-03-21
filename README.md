
---

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

# Define a simple tool with a description
@pxt.udf
def stock_price(ticker: str) -> dict:
    """Retrieve the current stock price for a given ticker symbol."""
    stock = yf.Ticker(ticker)
    return stock.info


# Create the tools object with the UDF
tools = pxt.tools(stock_price)

# Create an agent with tools
agent = Agent(
    agent_name="financial_analyst",
    system_prompt="You are a CFA working at a top-tier investment banks.",
    tools=tools,
    reset=True,
)

# Test chat and tool_call functionality
print("--------------")
print(agent.chat("Hi, how are you?"))
print("--------------")
print(agent.tool_call("Get NVIDIA and Apple stock price"))
print("--------------")
print(agent.chat("What was my last question?"))

```

## How It’s Built

Want to see how Pixelagent’s `Agent` class comes together? We’ve broken it down into simple, step-by-step blueprints for both Anthropic and OpenAI. These guides show you how to build an agent with just chat and tool-calling, leveraging Pixeltable’s magic:

- **[Build with Anthropic](examples/build-your-own-agent/anthropic/README.md)**: Learn how we craft an agent using Claude, with cost-saving tricks like skipping chat history in tool calls.
- **[Build with OpenAI](examples/build-your-own-agent/openai/README.md)**: See how we use GPT models to create a lean, powerful agent with the same Pixeltable-driven efficiency.

Each guide starts with a minimal core and shows how Pixeltable handles persistence, orchestration, and updates—giving you a foundation to customize and extend.

## Common Extensions 

- **Memory**: Implement long-term memory systems
- **Knowledge**: Build RAG systems with multimodal support
- **Teams**: Create multi-agent collaborative setups
- **Reflection**: Add self-improvement loops
- **Planning**: Add planning loops
- **Multimodal**: Support images, videos, and other media types

## Why Choose Pixelagent? 

- **Data-First**: Focus on robust data management and persistence
- **Engineering Freedom**: Build exactly what you need without framework constraints
- **Simplified Workflow**: Automated handling of:
  - Data persistence and retrieval
  - LLM protocols
  - Tool integrations
  - State management

Ready to start building? Dive into the blueprints, tweak them to your needs, and let Pixelagent handle the infrastructure while you focus on innovation!

---