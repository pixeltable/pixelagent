import re
from datetime import datetime

import pixeltable as pxt
from pixelagent.openai import Agent

import yfinance as yf

# Define tools
@pxt.udf
def stock_info(ticker: str) -> dict:
    """Retrieve the current stock price for a given ticker symbol."""
    stock = yf.Ticker(ticker)
    return stock.info

react_tools = ["stock_info"]

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

def extract_section(text, section_name):
    """Extract a section from the text using regex to handle variations in format"""
    pattern = rf'{section_name}:?\s*(.*?)(?=\n\s*(?:THOUGHT|ACTION):|$)'
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""

step = 1
max_steps = 10
question = "Create a detailed investment recommendation for Apple Inc. (AAPL) based on current market conditions."

# clear Agent history before Loop
pxt.drop_dir("financial_analyst_react", force=True)

# Start Agentic Loop
while step <= max_steps:

    print(f"Step {step}:\n")
    
    react_system_prompt = REACT_SYSTEM_PROMPT.format(
        date=datetime.now().strftime("%Y-%m-%d"),
        tools="\n".join(react_tools),
        step=step,
        max_steps=max_steps,
    )
    
    print("React System Prompt:\n")
    print(react_system_prompt)
    
    agent = Agent(
        agent_name="financial_analyst_react",
        system_prompt=react_system_prompt, # Dynamic React System Prompt
        reset=False, # maintains persistent memory
        # n_latest_messages=10, # define N rolling memory or it defaults to infinite memory
    )

    # Get Response
    response = agent.chat(question)
    print("Response:")
    print(response)

    # Extract Action
    action = extract_section(response, "ACTION")
    if "FINAL" in action.upper():
        break
    
    # Determine Tool Call
    call_tool = [tool for tool in react_tools if tool.lower() in action.lower()]
    if call_tool:
        # Agent has access to both tools and can call them in parallel if need be.
        tool_agent = Agent(
            agent_name="financial_analyst_react",
            system_prompt="Use your tools to answer the user's question.",
            tools=pxt.tools(stock_info),
        )        
        tool_call_result = tool_agent.tool_call(question)
        print("Tool Call Result:\n")
        print(tool_call_result)
    
    # Update step in max steps
    step += 1
    if step > max_steps:
        break

# Create final answer
summary_agent = Agent(
    agent_name="financial_analyst_react",
    system_prompt="Answer the user's question. Use your previous chat history to answer the question.",
)

final_answer = summary_agent.chat(question)
print("Final Answer:")
print(final_answer)
