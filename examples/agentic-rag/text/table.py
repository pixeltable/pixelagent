import pixeltable as pxt
from pixeltable.functions.huggingface import sentence_transformer

# Initialize app structure
pxt.drop_dir("text_search", force=True)
pxt.create_dir("text_search")

# Create texts table
t = pxt.create_table(
  "text_search.text_table", 
  {"text": pxt.String}
)

# Configure embedding model
embed_model = sentence_transformer.using(
  model_id="intfloat/e5-large-v2"
)

# Add search capability
t.add_embedding_index(
  column="text", 
  string_embed=embed_model
)
