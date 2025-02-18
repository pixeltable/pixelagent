from pixelagent.openai import Agent
from pixelagent.openai.workflows import Planning

# Create an agent with a general problem-solving system prompt
agent = Agent(
    name="problem_solver",
    system_prompt="""You are a helpful AI assistant that breaks down problems 
    and solves them systematically. You provide clear, detailed explanations.""",
    model="gpt-4o-mini"
)

# Create a planning workflow
planning_workflow = Planning(max_iterations=3)

# Example query that benefits from planning
complex_query = """
I need to organize a small team offsite event. The team has 8 people, 
and we need to plan activities, food, and logistics for a full day. 
Our budget is $1000. What's the best way to plan this?
"""

# Run the planning workflow
result = planning_workflow.run(agent.name, complex_query)
print("Final Response:", result)
