# ============================================================================
# Agent Reflection Loop Tutorial
# ============================================================================
# This tutorial demonstrates how to implement a self-improving agent using
# a reflection loop pattern. The agent generates responses, reflects on them,
# and iteratively improves based on its own critique.

import pixeltable as pxt
from pixelagent.anthropic import Agent
import yfinance as yf

# ============================================================================
# SECTION 1: REFLECTION LOOP CORE FUNCTION
# ============================================================================
# This function implements the full reflection cycle for agent improvement

def reflection_loop(
    main_agent: Agent,
    reflection_agent: Agent,
    user_msg: str,
    max_iterations: int = 3,
    verbose: int = 0,
    is_tool_call: bool = False
) -> str:
    """
    Run a complete reflection loop for a user message or tool call.
    
    The reflection loop follows these steps:
    1. Generate initial response
    2. Critique the response
    3. Improve based on critique
    4. Repeat until satisfied or max iterations reached
    
    Args:
        main_agent (Agent): The main PixelAgent instance that generates responses
        reflection_agent (Agent): The reflection PixelAgent instance that critiques
        user_msg (str): The user message or query to process
        max_iterations (int): Maximum number of reflection-improvement cycles
        verbose (int): Verbosity level (0=quiet, 1=show all steps)
        is_tool_call (bool): Whether to use tool_call instead of chat
    
    Returns:
        str: The final refined response after reflection
    """
    # Step 1: Initial response generation
    if is_tool_call:
        response = main_agent.tool_call(user_msg)
        original_result = response  # Save original tool call result for context
    else:
        response = main_agent.chat(user_msg)
    
    # Print initial response if verbose mode is enabled
    if verbose > 0:
        print(f"\n\nINITIAL {'TOOL CALL' if is_tool_call else 'GENERATION'}\n\n", response)
    
    # Step 2-4: Reflection and improvement iterations
    for i in range(max_iterations):
        if verbose > 0:
            print(f"\n\nITERATION {i+1}/{max_iterations}\n\n")
        
        # Generate critique of the current response
        critique = reflection_agent.chat(f"Please critique the following response:\n\n{response}")
        
        if verbose > 0:
            print("\n\nREFLECTION\n\n", critique)
        
        # Check if the reflection agent is satisfied with the response
        if "<OK>" in critique:
            if verbose > 0:
                print(f"\n\nResponse is satisfactory. Stopping reflection loop.\n\n")
            break
        
        # Refine the response based on the critique
        if is_tool_call:
            # For tool calls, include the original result for context
            prompt = f"The following was the result of a tool call: {original_result}\n\n" \
                     f"Please improve this result based on this critique:\n\n{critique}"
        else:
            # For regular chat, just ask for improvements based on critique
            prompt = f"Please improve your previous response based on this critique:\n\n{critique}"
        
        # Generate improved response
        response = main_agent.chat(prompt)
        
        if verbose > 0:
            print(f"\n\nREFINED {'TOOL CALL RESULT' if is_tool_call else 'RESPONSE'}\n\n", response)
    
    # Return the final refined response
    return response

# ============================================================================
# SECTION 2: TOOL DEFINITION
# ============================================================================
# Define a financial data tool that our agent can use

@pxt.udf
def stock_info(ticker: str) -> dict:
    """
    Retrieve stock information for a given ticker symbol.
    
    Args:
        ticker (str): Stock ticker symbol (e.g., 'AAPL' for Apple)
        
    Returns:
        dict: Dictionary containing stock information and metrics
    """
    stock = yf.Ticker(ticker)
    return stock.info

# ============================================================================
# SECTION 3: AGENT SETUP
# ============================================================================
# Configure the main agent and reflection agent with appropriate prompts

# Register the stock_info tool
tools = pxt.tools(stock_info)

# Main agent that generates responses and uses tools
main_agent = Agent(
    agent_name="main_bot",
    system_prompt="You're a helpful financial assistant.",
    tools=tools,
    reset=True  # Start with a fresh conversation history
)

# Reflection agent that critiques responses
reflection_agent = Agent(
    agent_name="reflection_bot",
    system_prompt="""
    You are tasked with generating critique and recommendations for responses.
    If the response has something wrong or something to be improved, output a list of recommendations
    and critiques. If the response is good and there's nothing to change, output this: <OK>
    """,
    reset=True  # Start with a fresh conversation history
)

# ============================================================================
# SECTION 4: EXAMPLE USAGE
# ============================================================================
# Demonstrate how to use the reflection loop with a financial query

if __name__ == "__main__":
    # Example query that uses the stock_info tool
    tool_query = "Get NVIDIA and Apple stock price and explain what the results mean"
    
    # Run the reflection loop
    refined_tool_result = reflection_loop(
        main_agent=main_agent, 
        reflection_agent=reflection_agent, 
        user_msg=tool_query, 
        verbose=1,  # Show all steps
        is_tool_call=True  # Use tool_call instead of chat
    )
    
    # Output the final result
    print("\nFINAL TOOL RESULT:", refined_tool_result)