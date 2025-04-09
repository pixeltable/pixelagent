from pixelagent.openai import Agent

vision_agent = Agent(
    agent_name="openai_vision_agent",
    system_prompt="Answer questions about the image",
    model="gpt-4o-mini",
    reset=True,
    chat_kwargs={"temperature": 0.0},
)

url = "https://raw.githubusercontent.com/pixeltable/pixeltable/main/docs/resources/port-townsend-map.jpeg"
print(vision_agent.chat("What is in the image?", image=url))
