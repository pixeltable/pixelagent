from pxl.agent import initialize_agent, run_agent

# Initialize the dog trainer agent
initialize_agent(
    agent_name="Dog_Trainer",
    system_prompt="You specialize in training dogs",
    model_name="gpt-4o-mini",
    reset_memory=False,
)

# Run the agent
result = run_agent("Dog_Trainer", "in 5 words tell me how to train my dog to sit")

print(result)
