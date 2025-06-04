<div align="center">
  <img src="https://github.com/user-attachments/assets/640a77dc-8e34-41f1-a698-a24c8c6aa021" alt="Pixelagent" width="800"/>

  <!-- Badges - First Row -->
  <p>
    <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-0530AD.svg" alt="License"></a>
    <img src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue?logo=python&logoColor=white" alt="Python Versions">
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
- **Native Multimodal**: Built-in support for text, images, audio and video
- **Declarative Model**: A type-safe python framework
- **Model agnostic**: Extensible to multiple providers 
- **Observability**: Complete traceability with automatic logging of messages, tool calls, and performance metrics
- **Agentic Extensions**: Add reasoning, reflection, memory, knowledge, and team workflows.

## Connect blueprints to Cursor, Windsurf, Cline:

- **[Anthropic](https://github.com/pixeltable/pixelagent/blob/main/blueprints/single_provider/anthropic/README.md)**
- **[OpenAI](https://github.com/pixeltable/pixelagent/blob/main/blueprints/single_provider/openai/README.md)**
- **[AWS Bedrock](https://github.com/pixeltable/pixelagent/blob/main/blueprints/single_provider/bedrock/README.md)** 
- **[Multiprovider](https://github.com/pixeltable/pixelagent/tree/main/blueprints/multi_provider/README.md)**

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
pip install pixelagent
# Install provider-specific dependencies
pip install anthropic  # For Claude models
pip install openai     # For GPT models
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
import pixeltable as pxt
from pixelagent.anthropic import Agent
import yfinance as yf

# Define a tool as a UDF
@pxt.udf
def stock_price(ticker: str) -> dict:
    """Get stock information for a ticker symbol"""
    stock = yf.Ticker(ticker)
    return stock.info

# Create agent with tool
agent = Agent(
    name="financial_assistant",
    system_prompt="You are a financial analyst assistant.",
    tools=pxt.tools(stock_price)
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
from pixelagent.openai import Agent
import pixeltable as pxt

# Define a tool
@pxt.udf
def stock_info(ticker: str) -> dict:
    """Get stock information for analysis"""
    import yfinance as yf
    stock = yf.Ticker(ticker)
    return stock.info

# ReAct system prompt with structured reasoning pattern
REACT_PROMPT = """
Today is {date}

IMPORTANT: You have {max_steps} maximum steps. You are on step {step}.

Follow this EXACT step-by-step reasoning and action pattern:

1. THOUGHT: Think about what information you need to answer the question.
2. ACTION: Either use a tool OR write "FINAL" if you're ready to give your final answer.

Available tools:
{tools}

Always structure your response with these exact headings:

THOUGHT: [your reasoning]
ACTION: [tool_name] OR simply write "FINAL"
"""

# Helper function to extract sections from responses
def extract_section(text, section_name):
    pattern = rf'{section_name}:?\s*(.*?)(?=\n\s*(?:THOUGHT|ACTION):|$)'
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""

# Execute ReAct planning loop
def run_react_loop(question, max_steps=5):
    step = 1
    while step <= max_steps:
        # Dynamic system prompt with current step
        react_system_prompt = REACT_PROMPT.format(
            date=datetime.now().strftime("%Y-%m-%d"),
            tools=["stock_info"],
            step=step,
            max_steps=max_steps,
        )
        
        # Agent with updated system prompt
        agent = Agent(
            name="financial_planner",
            system_prompt=react_system_prompt,
            reset=False,  # Maintain memory between steps
        )
        
        # Get agent's response for current step
        response = agent.chat(question)
        
        # Extract action to determine next step
        action = extract_section(response, "ACTION")
        
        # Check if agent is ready for final answer
        if "FINAL" in action.upper():
            break
            
        # Call tool if needed
        if "stock_info" in action.lower():
            tool_agent = Agent(
                name="financial_planner",
                tools=pxt.tools(stock_info)
            )
            tool_agent.tool_call(question)
            
        step += 1
    
    # Generate final recommendation
    return Agent(name="financial_planner").chat(question)

# Run the planning loop
recommendation = run_react_loop("Create an investment recommendation for AAPL")
```

Check out our [tutorials](examples/) for more examples including reflection loops, planning patterns, and multi-provider implementations.

## Tutorials and Examples

- **Basics**: Check out [Getting Started](examples/getting-started/pixelagent_basics_tutorial.py) for a step-by-step introduction to core concepts
- **Advanced Patterns**: Explore [Reflection](examples/reflection/anthropic/reflection.py) and [Planning](examples/planning/anthropic/react.py) for more complex agent architectures
- **Specialized Directories**: Browse our example directories for deeper implementations of specific techniques


Ready to start building? Dive into the blueprints, tweak them to your needs, and let Pixeltable handle the AI data infrastructure while you focus on innovation!
