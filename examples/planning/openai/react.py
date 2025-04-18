# ============================================================================
# Stock Advisor Agent Tutorial
# ============================================================================
# This tutorial demonstrates how to build a financial analyst agent that uses
# ReAct (Reasoning + Action) methodology to provide investment recommendations
# The agent follows a structured thought process and can access external data

import re
from datetime import datetime

# Import necessary libraries
import pixeltable as pxt  # Database for AI agent memory
import yfinance as yf  # Financial data API

from pixelagent.openai import Agent  # Agent framework

# ============================================================================
# SECTION 1: DEFINE TOOLS
# ============================================================================
# The agent needs access to external tools to gather financial information
# Here we create a UDF (User-Defined Function) that fetches stock information


@pxt.udf
def stock_info(ticker: str) -> dict:
    """
    Retrieve comprehensive stock information for a given ticker symbol.

    Args:
        ticker (str): Stock ticker symbol (e.g., 'AAPL' for Apple)

    Returns:
        dict: Dictionary containing stock information and metrics
    """
    stock = yf.Ticker(ticker)
    return stock.info


# List of tools available to the agent
react_tools = ["stock_info"]

# ============================================================================
# SECTION 2: DEFINE THE REACT SYSTEM PROMPT
# ============================================================================
# The ReAct methodology requires a specific prompt structure to guide the agent
# through alternating reasoning and action steps

REACT_SYSTEM_PROMPT = """
Today is {date}

IMPORTANT: You have {max_steps} maximum steps. You are on step {step}.

Follow this EXACT step-by-step reasoning and action pattern:

1. THOUGHT: Think about what information you need to answer the user's question.
2. ACTION: Either use a tool OR write "FINAL" if you're ready to give your final answer.

Available tools:
{tools}


Always structure your response with these exact headings:

THOUGHT: [your reasoning]
ACTION: [tool_name] OR simply write "FINAL"

Your memory will automatically update with the tool calling results. Use those results to inform your next action.
"""

# ============================================================================
# SECTION 3: HELPER FUNCTIONS
# ============================================================================
# Utility function to parse agent responses and extract specific sections


def extract_section(text, section_name):
    """
    Extract a specific section from the agent's response text.

    Args:
        text (str): The full text response from the agent
        section_name (str): The section to extract (e.g., 'THOUGHT', 'ACTION')

    Returns:
        str: The extracted section content or empty string if not found
    """
    pattern = rf"{section_name}:?\s*(.*?)(?=\n\s*(?:THOUGHT|ACTION):|$)"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""


# ============================================================================
# SECTION 4: MAIN EXECUTION LOOP
# ============================================================================
# Initialize variables for the ReAct loop
step = 1
max_steps = 10  # Maximum steps before forced termination
question = "Create a detailed investment recommendation for Apple Inc. (AAPL) based on current market conditions."

# Clear previous agent history before starting
pxt.drop_dir("financial_analyst_react", force=True)

# Start the ReAct loop - the agent will alternate between thinking and acting
print("Starting Financial Analyst Agent with ReAct methodology...")
while step <= max_steps:

    print(f"Step {step}:\n")

    # Generate the dynamic system prompt with current step information
    react_system_prompt = REACT_SYSTEM_PROMPT.format(
        date=datetime.now().strftime("%Y-%m-%d"),
        tools="\n".join(react_tools),
        step=step,
        max_steps=max_steps,
    )

    print("React System Prompt:\n")
    print(react_system_prompt)

    # Initialize the ReAct agent with persistent memory (reset=False)
    agent = Agent(
        name="financial_analyst_react",
        system_prompt=react_system_prompt,  # Dynamic React System Prompt
        reset=False,  # Maintains persistent memory across steps
        # n_latest_messages=10,  # Optional: Define N rolling memory window
    )

    # Get the agent's response for the current step
    response = agent.chat(question)
    print("Response:")
    print(response)

    # Extract the ACTION section to determine next steps
    action = extract_section(response, "ACTION")

    # Check if the agent is ready to finalize its answer
    if "FINAL" in action.upper():
        print("Agent has decided to finalize answer.")
        break

    # Determine which tool to call based on the agent's action
    call_tool = [tool for tool in react_tools if tool.lower() in action.lower()]

    if call_tool:
        print(f"Agent has decided to use tool: {call_tool[0]}")

        # Create a tool-specific agent to handle the tool call
        tool_agent = Agent(
            name="financial_analyst_react",
            system_prompt="Use your tools to answer the user's question.",
            tools=pxt.tools(stock_info),  # Register the stock_info tool
        )

        # Execute the tool call and get results
        tool_call_result = tool_agent.tool_call(question)
        print("Tool Call Result:\n")
        print(tool_call_result)

    # Increment the step counter
    step += 1

    # Safety check to prevent infinite loops
    if step > max_steps:
        print("Reached maximum steps. Forcing termination.")
        break

# ============================================================================
# SECTION 5: GENERATE FINAL ANSWER
# ============================================================================
# Create a final summary agent that accesses the conversation history
print("Generating final investment recommendation...")

summary_agent = Agent(
    name="financial_analyst_react",  # Same agent name to access history
    system_prompt="Answer the user's question. Use your previous chat history to answer the question.",
)

# Generate the final comprehensive answer
final_answer = summary_agent.chat(question)
print("Final Investment Recommendation:")
print(final_answer)
