from pixelagent.openai import Agent
import pixeltable as pxt

from tools import search_channel_messages

agent = Agent(
    name = "slackbot",
    model = "o3-2025-04-16",
    system_prompt = """
You are a helpful assistant that can answer questions and provide information about our slack channel
""",
    tools = pxt.tools(search_channel_messages),
    reset=True
)

print(agent.tool_call("summarize topics in the general channel about pixeltable"))