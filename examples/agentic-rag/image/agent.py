import pixeltable as pxt

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

# Search images
@pxt.query
def find_images(query: str, top_k: int):
    sim = img_t.image_description.similarity(query)
    return (
        img_t.order_by(sim, asc=False)
        .select(
            img_t.image,
            img_t.image_description,
            similarity=sim
        )
        .limit(top_k)
    )

agent = Agent(
    name = "image_search.agent", 
    system_prompt = "Use your tool to search the image database.", 
    tools = pxt.tools(find_images)
)

agent.tool_call("Provide the URL/Filepaths for images containing blue flowers")