from duckduckgo_search import DDGS

from pixelagent.anthropic import AgentX, power

@power
def search_the_web(keywords: str, max_results: int) -> str:
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


agent = AgentX(
    name="web_agent",
    model="claude-3-5-sonnet-20241022",
    system_prompt="you are a helpful assistant that can search the web for information",
    powers=[search_the_web],
)

res = agent.execute("whats the latest news in Denver? Who won the superbowl?")
print(res)
