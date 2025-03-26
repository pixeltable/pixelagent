# Self-Reflection Pattern Tutorial

This tutorial shows you how to use PixelAgent's self-reflection pattern to create agents that can critique and improve their own responses. We'll build a financial analysis agent that gets better through self-reflection.

## What is Self-Reflection?

Self-reflection is a pattern where one agent critiques another agent's output to improve it. The pattern uses:
- A main agent that generates responses
- A reflection agent that critiques and suggests improvements
- Automatic tool handling and memory management

## Basic Example

Here's a simple example of the reflection pattern:

```python
from pixelagent.patterns import reflection_loop
from pixelagent.openai import Agent
import pixeltable as pxt
import yfinance as yf

# First, define a financial analysis tool
@pxt.udf
def stock_info(ticker: str) -> dict:
    """Get stock information from Yahoo Finance"""
    stock = yf.Ticker(ticker)
    return stock.info

# Create the main agent with the tool
tools = pxt.tools(stock_info)
main_agent = Agent(
    agent_name="finance_bot",
    system_prompt="You're a financial analyst. Provide clear insights about stocks.",
    tools=tools,  # Tools are automatically handled - no manual tool calling needed
    reset=True
)

# Create the reflection agent
reflection_agent = Agent(
    agent_name="critic",
    system_prompt="""
    Review financial analyses for:
    1. Data accuracy and completeness
    2. Clear explanations of metrics
    3. Risk assessment
    If improvements needed, list recommendations.
    If analysis is good, output: <OK>
    """,
    reset=True
)

# Run the reflection loop
analysis = reflection_loop(
    main_agent,
    reflection_agent,
    "Analyze NVDA stock performance",
    max_iterations=2,
    is_tool_call=True  # Automatically handles tool calling
)
```

## How the Pattern Works

The reflection loop follows these steps:

1. **Initial Analysis**: The main agent uses `stock_info` to get data and generate analysis
2. **Review**: The reflection agent checks the analysis quality
3. **Refinement**: If needed, the main agent improves based on feedback
4. **Repeat**: Until quality meets standards or max iterations reached

Key features that make this work:
- Tool calls are handled automatically - no need to manage request/response
- Conversation history is saved automatically
- Original data context is preserved during refinement

## Step-by-Step Breakdown

Let's look at what happens in each iteration:

```python
# 1. Initial tool call and analysis
response = main_agent.tool_call("Analyze NVDA stock")
# Gets stock data and generates initial analysis

# 2. Reflection agent review
critique = reflection_agent.chat(f"Review this analysis:\n{response}")
# Might suggest: "Add P/E ratio context, discuss industry trends"

# 3. Main agent refinement
improved = main_agent.chat(
    f"Improve analysis based on: {critique}"
)
# Enhances analysis with suggested improvements
```

## Common Use Cases

The reflection pattern works well for:
- Stock analysis reports
- Financial recommendations
- Market trend analysis
- Risk assessments

## Tips for Better Results

1. **Clear Criteria**: Give your reflection agent specific review points
2. **Preserve Context**: The pattern automatically keeps original data available
3. **Iteration Balance**: 2-3 iterations usually give best results
4. **Tool Integration**: Use PixelAgent's built-in tool handling

## Example: Enhanced Financial Analysis

Here's a more detailed example showing how to build a comprehensive stock analyzer:

```python
# Define additional financial tools
@pxt.udf
def market_news(ticker: str) -> list:
    """Get recent news about the stock"""
    stock = yf.Ticker(ticker)
    return stock.news

tools = pxt.tools([stock_info, market_news])

main_agent = Agent(
    agent_name="finance_expert",
    system_prompt="""
    You are a thorough financial analyst.
    - Use stock_info for fundamental data
    - Use market_news for recent developments
    - Combine quantitative and qualitative analysis
    """,
    tools=tools,
    reset=True
)

reflection_agent = Agent(
    agent_name="finance_critic",
    system_prompt="""
    Evaluate financial analysis for:
    1. Data accuracy and completeness
    2. Market context and trends
    3. Risk assessment
    4. Clear actionable insights
    Output <OK> if analysis meets all criteria.
    """,
    reset=True
)

# Run analysis with automatic tool handling
analysis = reflection_loop(
    main_agent,
    reflection_agent,
    "Should investors consider NVDA stock now?",
    max_iterations=2,
    is_tool_call=True
)
```

## Next Steps

1. Try modifying the example code with different tools
2. Experiment with different reflection criteria
3. Adjust max_iterations to find the right balance

For more examples, check out other patterns in `pixelagent.patterns`.