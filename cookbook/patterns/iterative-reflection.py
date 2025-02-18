from pixelagent.openai import Agent, IterativeReflection

# Create specialized writing agents
writer = Agent(
    name="technical_writer",
    system_prompt = """You are an expert technical writer. Write clear, accurate, and engaging explanations 
    of complex topics. Focus on making the content accessible while maintaining technical accuracy."""
)

editor = Agent(
    name="editor",
    system_prompt = """
    You are a professional editor. Review text for:
    1. Technical accuracy
    2. Clarity and readability
    3. Engagement and flow
    4. Grammar and style
    
    If the text meets all criteria, respond with <OK>.
    Otherwise, provide specific suggestions for improvement."""
)

# Create the reflection workflow
reflection = IterativeReflection(
    primary_agent=writer,
    critic_agent=editor,
    max_iterations=3
)

# Example topics to explain
topics = [
    "Explain how blockchain technology works to a non-technical audience",
    "Describe the process of photosynthesis in an engaging way",
    "Explain the concept of quantum entanglement to high school students"
]

# Process each topic
for topic in topics:
    print(f"\n{'='*80}\nTopic: {topic}\n{'='*80}")
    
    final_text = reflection.run(topic)
    
    print("\nFinal Text:")
    print(final_text)
    
    # Print the revision history
    print("\nRevision History:")
    history = reflection.state_table.select().collect()
    for record in history:
        print(f"\nIteration {record['iteration']}")
        print(f"Agent: {record['agent']}")
        print(f"Input: {record['input'][:100]}...")
        print(f"Output: {record['output'][:100]}...")
        print(f"Phase: {record['metadata']}")