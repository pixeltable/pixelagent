from pxl.providers import openai_agent

# Initialize the dog trainer agent
openai_agent.init(
    agent_name="Dog_Trainer",
    system_prompt="You specialize in training dogs",
    model_name="gpt-4o-mini",
    reset_memory=False,
)

# Run the agent
result = openai_agent.run(
    agent_name="Dog_Trainer",
    message="in 5 words tell me how to train my dog to sit"
)


print(result)
