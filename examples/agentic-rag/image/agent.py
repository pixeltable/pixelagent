import pixeltable as pxt
from pixelagent.openai import Agent

# Connect to your table
img_t = pxt.get_table("image_search.images")

# Sample image URLs
IMAGE_URL = (
    "https://raw.github.com/pixeltable/pixeltable/release/docs/resources/images/"
)

image_urls = [
    IMAGE_URL + doc for doc in [
        "000000000030.jpg",
        "000000000034.jpg",
        "000000000042.jpg",
    ]
]

# Add images to the database
img_t.insert({"image": url} for url in image_urls)

@pxt.query
def find_images(query: str):
    sim = img_t.image_description.similarity(query)
    return (
        img_t.order_by(sim, asc=False)
        .select(img_t.image_description)
        .limit(5)
    )

tools = pxt.tools(find_images)
agent = Agent(
    agent_name = "image_search.agent", 
    system_prompt = "Use your tool to search the image index.", 
    tools = tools,
    reset=True
)

print(agent.tool_call("Describe the image that contains flowers"))