import pixeltable as pxt
from pixeltable.functions.openai import vision
from pixeltable.functions.huggingface import sentence_transformer

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
  image_description=vision(
      prompt="Describe the image. Be specific on the colors you see.",
      image=img_t.image,
      model="gpt-4o-mini",
  )
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
