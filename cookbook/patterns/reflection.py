from pixelagent.openai import Agent

from flow import SelfReflection
# Initialize the helpful assistant agent.
helpful_assistant = Agent(
    name="helpful_assistant",
    system_prompt="you are a helpful assistant",
    model="gpt-4o-mini",
    reset=True  # ensure a fresh conversation state
)

# Initialize the critic agent.
# Its system prompt instructs it to act as a meticulous critic.
critic_agent = Agent(
    name="helpful_assistant_critic",
    system_prompt=(
        "You are a meticulous critic. Evaluate the following response from a helpful assistant. "
        "Return a JSON object with keys 'approved' (a boolean) and 'feedback' (a string). "
        "If the response is satisfactory, set 'approved' to true."
    ),
    model="gpt-4o-mini",
    reset=True
)

# Have a conversation with the helpful assistant.
# The conversation history persists between runs.
helpful_assistant.run("hi my name is john")
helpful_assistant.run("i went to mit")
initial_response = helpful_assistant.run("what is my name")

print("Initial Response from helpful_assistant:")
print(initial_response)

# Now apply self-reflection to refine the response.
reflection_workflow = SelfReflection(max_iterations=3)
final_response = reflection_workflow.run(agent=helpful_assistant, initial_response=initial_response)

print("\nFinal Response after Self-Reflection:")
print(final_response)