from duckduckgo_search import DDGS

from pixelagent.openai import Agent, tool


@tool
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

agent = Agent(
    name="web_agent",
    model="gpt-4o-mini",
    system_prompt="you are a helpful assistant that can search the web for information",
    tools=[search_the_web],
    reset=True
)

question = "whats the latest news in Denver? Who won the superbowl?"
res = agent.run(question)
print(res)