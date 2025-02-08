# pip install pixeltable openai duckduckgo-search
import pixeltable as pxt
from duckduckgo_search import DDGS

from pxl.providers import openai_agent


@pxt.udf
def search_news(keywords: str, max_results: int = 20) -> str:
    """Search news using DuckDuckGo and return results."""
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


# Initialize the web research agent
openai_agent.init(
    agent_name="Web_Research_Agent",
    system_prompt="You are a web research agent, who can access web search data. Help the user with their research.",
    model_name="gpt-4o-mini",
    verbose=True,
    agent_tools=pxt.tools(search_news),
    reset_memory=False,  # set to true to delete the agent and start fresh
)

# Run the agent
response = openai_agent.run(
    agent_name="Web_Research_Agent", message="Who is playing in the superbowl?"
)
print(response)
