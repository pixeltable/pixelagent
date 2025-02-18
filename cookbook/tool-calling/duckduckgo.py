# pip install pixeltable openai duckduckgo-search

import pixeltable as pxt
from pixelagent.openai import Agent

from duckduckgo_search import DDGS

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

# Create tools collection
ddg_tools = pxt.tools(search_news)

# Create agent with DuckDuckGo tools
agent = Agent(
    name="web_research_agent",
    system_prompt="You are a web research agent, who can access web search data. Help the user with their research.",
    tools=ddg_tools
)

# Example research
query = "Who is playing in the Super Bowl?"

response = agent.run(query)
print("\nQuery:")
print(query)
print("\nAnalysis:")
print(response)
