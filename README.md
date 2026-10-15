<div align="center">
  <img src="https://github.com/user-attachments/assets/640a77dc-8e34-41f1-a698-a24c8c6aa021" alt="Pixelagent" width="800"/>

  <!-- Badges - First Row -->
  <p>
    <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-0530AD.svg" alt="License"></a>
    <img src="https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue?logo=python&logoColor=white" alt="Python Versions">
    <img src="https://img.shields.io/badge/platform-Linux%20%7C%20macOS%20%7C%20Windows-E5DDD4" alt="Platform Support">
  </p>
  
  <!-- Badges - Second Row -->
  <p>
    <a href="https://pypi.org/project/pixelagent/"><img src="https://img.shields.io/pypi/v/pixelagent?color=4D148C" alt="PyPI Package"></a>
    <a href="https://discord.gg/QPyqFYx2UN"><img src="https://img.shields.io/badge/💬-Discord-%235865F2.svg" alt="Discord"></a>
  </p>
  
  <!-- Documentation Links -->
  <p>
    <b><a href="https://docs.pixeltable.com/">Documentation</a></b> |
    <b><a href="https://pixeltable.github.io/pixeltable/">API Reference</a></b> |
    <b><a href="https://docs.pixeltable.com/docs/examples/use-cases">Examples</a></b> |
  </p>
  
  <!-- Demo Video -->
  <a href="https://www.youtube.com/watch?v=BS6PRsnxkBA">
    <img src="https://img.youtube.com/vi/BS6PRsnxkBA/maxresdefault.jpg" alt="Demo Video" width="800"/>
    <p>👆 Click to watch the demo video</p>
  </a>
</div>

---
# Pixelagent: An Agent Engineering Blueprint 

We see agents as the intersection of an LLM, storage, and orchestration. [Pixeltable](https://github.com/pixeltable/pixeltable) unifies this interface into a single declarative framework, making it the de-facto choice for engineers to build custom agentic applications with build-your-own functionality for memory, tool-calling, and more.


## Build your own agent framework: 

- **Data Orchestration and Storage**: Built on Pixeltable's data infrastructure
- **Native Multimodal**: `chat()` takes text and an image
- **Declarative Model**: A type-safe python framework
- **Model agnostic**: Extensible to multiple providers 
- **Observability**: Messages and tool calls are stored in tables
- **Agentic Extensions**: Examples for reasoning, reflection, and memory

## Use with Cursor, Windsurf, Cline

Install the package and point your editor at it:

```bash
pip install pixelagent
```

All four providers ship in the package:

```python
from pixelagent.openai import Agent   # or .anthropic, .bedrock, .gemini

agent = Agent(name="my_agent", system_prompt="You are a helpful assistant.")
print(agent.chat("Hello"))
```

## Plug-and-Play Extensions 

- **[Tools](examples/tool-calling)**: Add custom python functions as tools
- **[Memory](examples/memory)**: Implement long-term memory systems with semantic search capabilities
- **[Reflection](examples/reflection)**: Add self-improvement loops
- **[Reasoning](examples/planning)**: Add planning loops
- **[Multimodal Agentic Rag](examples/agentic-rag)**: Multimodal agentic retrieval

## Usage

Transform your agent blueprint into a distributable package on PyPI, extending the build-your-own philosophy to deployment and sharing.

### Installation

```bash
pip install "pixeltable>=0.7.15,<0.8" "pixelagent[openai]"
# or pixelagent[anthropic], pixelagent[gemini], pixelagent[bedrock]
```

### Quick Start

```python
from pixelagent.anthropic import Agent  # Or from pixelagent.openai import Agent

# Create a simple agent
agent = Agent(
    name="my_assistant",
    system_prompt="You are a helpful assistant."
)

# Chat with your agent
response = agent.chat("Hello, who are you?")
print(response)
```

### Adding Tools

```python
# tools.py — Pixeltable stores a UDF by its import path, so it lives in a module.
import pixeltable as pxt
import yfinance as yf


@pxt.udf
def stock_price(ticker: str) -> dict:
    """Get stock information for a ticker symbol"""
    stock = yf.Ticker(ticker)
    return stock.info
```

```python
import pixeltable as pxt
from pixelagent.anthropic import Agent

from tools import stock_price

agent = Agent(
    name="financial_assistant",
    system_prompt="You are a financial analyst assistant.",
    tools=pxt.tools(stock_price),
)

# Use tool calling
result = agent.tool_call("What's the current price of NVDA?")
print(result)
```

### State management

```python
import pixeltable as pxt

# Agent memory is automatically persisted in tables
memory = pxt.get_table("my_assistant.memory")
conversations = memory.collect()

# Access tool call history
tools_log = pxt.get_table("financial_assistant.tools")
tool_history = tools_log.collect()

# cusomatizable memory database
conversational_agent = Agent(
    name="conversation_agent",
    system_prompt="Focus on remebering the conversation",
    n_latest_messages=14
)
```

### Custom Agentic Strategies
```python

# ReAct pattern for step-by-step reasoning and planning
import re
from datetime import datetime

import pixeltable as pxt

# Tools live in a module, not beside the agent: pixeltable needs a UDF to be
# importable by name so a stored computed column can find it again.
from react_tools import stock_info

from pixelagent.openai import Agent

REACT_PROMPT = """
Today is {date}

IMPORTANT: You have {max_steps} maximum steps. You are on step {step}.

1. THOUGHT: Think about what information you need to answer the question.
2. ACTION: Either use a tool OR write "FINAL" if you are ready to answer.

Available tools:
{tools}

THOUGHT: [your reasoning]
ACTION: [tool_name] OR simply write "FINAL"
"""


def extract_section(text, section_name):
    pattern = rf'{section_name}:?\s*(.*?)(?=\n\s*(?:THOUGHT|ACTION):|$)'
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""


# One agent, built once. The system prompt is per-turn data, so a step that
# needs a different prompt passes it to chat().
agent = Agent(
    name="financial_planner",
    system_prompt="You are a financial analyst.",
    tools=pxt.tools(stock_info),
    reset=True,
)


def run_react_loop(question, max_steps=5):
    for step in range(1, max_steps + 1):
        response = agent.chat(
            question,
            system_prompt=REACT_PROMPT.format(
                date=datetime.now().strftime("%Y-%m-%d"),
                tools=["stock_info"],
                step=step,
                max_steps=max_steps,
            ),
        )
        action = extract_section(response, "ACTION")
        if "FINAL" in action.upper():
            break
        if "stock_info" in action.lower():
            agent.tool_call(question)

    return agent.chat(question)


recommendation = run_react_loop("Create an investment recommendation for AAPL")
```

Check out our [tutorials](examples/) for more examples including reflection loops, planning patterns, and multi-provider implementations.

## Migrating from 0.1.x

0.2.0 requires Pixeltable >= 0.7.15, < 0.8 and Python 3.11+. Upgrade with
`pip install -U "pixeltable>=0.7.15,<0.8" pixelagent`.

What changed for callers:

- **A different `model` updates the chat pipeline.** Existing rows keep their
  previous responses. `reset=True` drops the directory and rebuilds it.
  `SchemaConflict` is raised when Pixeltable refuses the edit, for example a
  toolset or column type it cannot reconcile.
- **`chat()` takes `system_prompt`, `model_kwargs`, `max_tokens`, and
  `conversation_id`.** `n_latest_messages` is a constructor argument. The
  history window is fixed when the agent is built.
- **Memory rows carry `conversation_id`** (default `"default"`), so one agent
  can hold several threads. The agent table also stores `system_prompt`,
  `model_kwargs`, `max_tokens`, `n_latest`, and `image`.

A failed turn leaves memory empty. `chat()` reads the reply from the insert.


## Tutorials and Examples

- **Basics**: Check out [Getting Started](examples/getting-started/pixelagent_basics_tutorial.py) for a step-by-step introduction to core concepts
- **Advanced Patterns**: Explore [Reflection](examples/reflection/anthropic/reflection.py) and [Planning](examples/planning/anthropic/react.py) for more complex agent architectures
- **Specialized Directories**: Browse our example directories for deeper implementations of specific techniques


Ready to start building? `pip install pixelagent`, work through the examples, and let Pixeltable handle the AI data infrastructure while you focus on your agent.
