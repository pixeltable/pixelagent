import pixeltable as pxt
from pixelagent.openai import Agent

# Connect to your table
feedback_t = pxt.get_table("customer_feedback.reviews")

# Sample customer feedback texts
feedback_texts = [
    "This app is incredibly user-friendly and fast!",
    "I encountered a bug when saving my profile.",
    "The new update improved my productivity significantly.",
]

# Add feedback texts to the database
feedback_t.insert({"review": text} for text in feedback_texts)

@pxt.query
def find_feedback(query: str) -> dict:
    sim = feedback_t.review.similarity(query)
    return (
        feedback_t.order_by(sim, asc=False)
        .select(feedback_t.review)
        .limit(5)
    )

tools = pxt.tools(find_feedback)
agent = Agent(
    agent_name = "feedback_agent", 
    system_prompt = "Use your tool to search the customer feedback reviews.", 
    tools = tools,
    reset=True
)

print(agent.tool_call("Find reviews about app performance"))