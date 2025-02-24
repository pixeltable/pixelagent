from pixelagent.openai import Agent

url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
agent = Agent(
    name="image_analyzer",
    system_prompt="You are an image analysis expert.",
    model="gpt-4o-mini",
    reset=True,
)
response = agent.run("Analyze the image", attachments=url)
print(response)
