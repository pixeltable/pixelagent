import pixeltable as pxt
from pixelagent.openai import Agent
import yfinance as yf

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
    
    Args:
        main_agent (Agent): The main PixelAgent instance
        reflection_agent (Agent): The reflection PixelAgent instance
        user_msg (str): The user message to process
        max_iterations (int): Maximum number of reflection iterations
        verbose (int): Verbosity level
        is_tool_call (bool): Whether to use tool_call instead of chat
    
    Returns:
        str: The final refined response
    """
    # Initial generation
    if is_tool_call:
        response = main_agent.tool_call(user_msg)
        original_result = response  # Save for refinement context
    else:
        response = main_agent.chat(user_msg)
    
    if verbose > 0:
        print(f"\n\nINITIAL {'TOOL CALL' if is_tool_call else 'GENERATION'}\n\n", response)
    
    # Reflection iterations
    for i in range(max_iterations):
        if verbose > 0:
            print(f"\n\nITERATION {i+1}/{max_iterations}\n\n")
        
        # Generate critique
        critique = reflection_agent.chat(f"Please critique the following response:\n\n{response}")
        
        if verbose > 0:
            print("\n\nREFLECTION\n\n", critique)
        
        # Check if satisfied
        if "<OK>" in critique:
            if verbose > 0:
                print(f"\n\nResponse is satisfactory. Stopping reflection loop.\n\n")
            break
        
        # Refine based on context
        if is_tool_call:
            prompt = f"The following was the result of a tool call: {original_result}\n\n" \
                     f"Please improve this result based on this critique:\n\n{critique}"
        else:
            prompt = f"Please improve your previous response based on this critique:\n\n{critique}"
        
        response = main_agent.chat(prompt)
        
        if verbose > 0:
            print(f"\n\nREFINED {'TOOL CALL RESULT' if is_tool_call else 'RESPONSE'}\n\n", response)
    
    return response

# Define a tool for our example
@pxt.udf
def stock_info(ticker: str) -> dict:
    stock = yf.Ticker(ticker)
    return stock.info

# Set up agents and tools
tools = pxt.tools(stock_info)

main_agent = Agent(
    agent_name="main_bot",
    system_prompt="You're a helpful financial assistant.",
    tools=tools,
    reset=True
)

reflection_agent = Agent(
    agent_name="reflection_bot",
    system_prompt="""
    You are tasked with generating critique and recommendations for responses.
    If the response has something wrong or something to be improved, output a list of recommendations
    and critiques. If the response is good and there's nothing to change, output this: <OK>
    """,
    reset=True
)

# Example usage
if __name__ == "__main__":
    tool_query = "Get NVIDIA and Apple stock price and explain what the results mean"
    refined_tool_result = reflection_loop(
        main_agent, reflection_agent, tool_query, 
        verbose=1, is_tool_call=True
    )
    print("\nFINAL TOOL RESULT:", refined_tool_result)