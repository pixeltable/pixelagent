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
    model="gpt-4o-mini",
    system_prompt="speak in a heavy southern accent"
)

res = agent.run("write a technical paper (less than 100 words) about the history of the internet")
print(res)
