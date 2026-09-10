import pixeltable as pxt
from pixeltable.functions.huggingface import sentence_transformer
from pixeltable.functions.openai import chat_completions

# Initialize app structure
pxt.drop_dir("image_search", force=True)
pxt.create_dir("image_search")

# Create images table
img_t = pxt.create_table(
  "image_search.images", 
  {"image": pxt.Image}
)

# Add OpenAI Vision analysis
img_t.add_computed_column(
image_description=chat_completions(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image in detail."},
                {"type": "image_url", "image_url": {"url": img_t.image}},
            ],
        }],
    ).choices[0].message.content
)

# Configure embedding model
embed_model = sentence_transformer.using(
  model_id="intfloat/e5-large-v2"
)

# Add search capability
img_t.add_embedding_index(
  column="image_description", 
  string_embed=embed_model
)
