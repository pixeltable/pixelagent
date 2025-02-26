from pixelagent.openai import AgentX

url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
agent = AgentX(
    name="image_analyzer",
    system_prompt="You are an image analysis expert.",
    model="gpt-4o-mini",
    reset=True,
)
response = agent.execute("Analyze the image", attachments=url)
print(response)
