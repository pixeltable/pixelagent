import pixeltable as pxt
from pixeltable.functions.openai import embeddings

# Initialize app structure
pxt.drop_dir("customer_feedback", force=True)
pxt.create_dir("customer_feedback")

# Create reviews table
t = pxt.create_table("customer_feedback.reviews", {"review": pxt.String})

# Add search capability
t.add_embedding_index(column="review", embedding=embeddings.using(model = "text-embedding-3-small"))
