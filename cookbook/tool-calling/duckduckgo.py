# pip install pixeltable openai duckduckgo-search

import pixeltable as pxt
from pixelagent.openai import Agent
from duckduckgo_search import DDGS
import time  # Add time module import

@pxt.udf
def search_the_web(keywords: str, max_results: int = 20) -> str:
    """Search the web using DuckDuckGo and return results."""
    try:
        with DDGS() as ddgs:
            results = ddgs.news(
                keywords=keywords,
                region="wt-wt",
                safesearch="off",
                timelimit="m",
                max_results=max_results,
            )
            formatted_results = []
            for i, r in enumerate(results, 1):
                formatted_results.append(
                    f"{i}. Title: {r['title']}\n"
                    f"   Source: {r['source']}\n"
                    f"   Published: {r['date']}\n"
                    f"   Snippet: {r['body']}\n"
                )
            return "\n".join(formatted_results)
    except Exception as e:
        return f"Search failed: {str(e)}"

# Start timing
start_time = time.time()

# Create tools collection
ddg_tools = pxt.tools(search_the_web)

# Create agent with DuckDuckGo tools and updated system prompt
agent = Agent(
    name="web_agent",
    model="gpt-4o-mini",
    system_prompt="you are a helpful assistant that can search the web for information",
    tools=ddg_tools,
)

# Example usage
question = "whats the latest news on the humane ai pin? Who won the superbowl?"
res = agent.run(question)

print(res)

# Calculate and print total execution time
end_time = time.time()
execution_time = end_time - start_time
print(f"\nTotal execution time: {execution_time:.2f} seconds")

# Uncomment to save chat history
pxt.get_table("web_agent.chat").collect().to_pandas().to_csv("web_agent_chat.csv")
pxt.get_table("web_agent.messages").collect().to_pandas().to_csv("web_agent_messages.csv")
