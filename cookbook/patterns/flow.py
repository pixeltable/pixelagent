import json

import pixeltable as pxt
from pixelagent.openai import Agent

def run(name: str, instructions: str, content: str) -> str:
    # Log the user message
    message_table = pxt.get_table(f'{name}.messages')
    message_table.insert([{'role': 'user', 'content': content}])

    # Invoke the LLM
    chat = pxt.get_table(f'{name}.chat')
    chat.insert([{'system_prompt': instructions}])

    # Log the agent response
    response = chat.select(chat.response).tail(1)['response'][0]
    message_table.insert([{'role': 'assistant', 'content': response}])
    
    return response


class Workflow:
    """A composable workflow that can be applied to any agent."""
    def __init__(self, name: str, stopping_criteria: str = None, max_iterations: int = 10):
        self.name = name
        self.stopping_criteria = stopping_criteria
        self.max_iterations = max_iterations
        
    def create_tables(self, agent_name: str):
        """Create tables needed for this workflow."""
        workflow_name = f"{agent_name}_{self.name}"
        pxt.drop_dir(workflow_name, force=True)
        pxt.create_dir(workflow_name)
        
        # Table to store workflow state
        return pxt.create_table(
            path_str=f"{workflow_name}.state",
            schema_or_df={
                'iteration': pxt.Int,
                'input': pxt.String,
                'critic_output': pxt.String,
                'status': pxt.String
            }
        )

class SelfReflection(Workflow):
    """
    A workflow that makes an agent reflect on its response using a critic.
    The critic is expected to return a JSON object with:
       - 'approved': a boolean indicating if the response is satisfactory,
       - 'feedback': a string with improvement suggestions.
    """
    def __init__(self, critic_prompt: str = None, **kwargs):
        super().__init__(name="reflection", **kwargs)
        self.critic_prompt = critic_prompt or (
            "Evaluate the response below. "
            "Return a JSON object with keys:\n"
            "  'approved': a boolean indicating if the response is satisfactory,\n"
            "  'feedback': a string explaining what to improve if not approved.\n"
            "If the response is satisfactory, set 'approved' to true."
        )
    
    def run(self, agent: Agent, initial_response: str) -> str:
        """
        Run the self-reflection workflow.
        
        Args:
            agent: An instance of the Agent class.
            initial_response: The initial response to be critiqued.
        
        Returns:
            The final, refined response.
        """
        state_table = self.create_tables(agent.name)
        iteration = 0
        current_response = initial_response

        while iteration < self.max_iterations:
            # Invoke the critic agent to evaluate the current response.
            # The critic agent's name is assumed to be f"{agent.name}_critic"
            critic_output = run(
                name=f"{agent.name}_critic",
                instructions=self.critic_prompt,
                content=current_response
            )
            
            # Try to parse the critic's output as JSON.
            try:
                critic_data = json.loads(critic_output)
            except Exception as e:
                critic_data = {"approved": False, "feedback": f"Critic output not in JSON format: {str(e)}"}
            
            # Log the iteration state.
            state_table.insert([{
                'iteration': iteration,
                'input': current_response,
                'critic_output': critic_output,
                'status': 'approved' if critic_data.get("approved") else 'needs_improvement'
            }])
            
            if critic_data.get("approved"):
                break

            # Request improvement based on critic feedback.
            improvement_instructions = (
                f"Improve your previous response based on the following feedback: {critic_data.get('feedback')}"
            )
            current_response = run(
                name=agent.name,
                instructions=improvement_instructions,
                content=current_response
            )
            iteration += 1
            
        return current_response