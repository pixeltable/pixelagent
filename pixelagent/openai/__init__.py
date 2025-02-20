from typing import Dict, List, Optional
from datetime import datetime
import pixeltable as pxt
from pixeltable.functions import openai

@pxt.udf
def _create_messages(past_context: List[Dict], system_prompt: str, tool_output: Optional[str] = None) -> List[Dict]:
    """Create messages list with system prompt, memory context and new message"""
    messages = [{'role': 'system', 'content': system_prompt}]
    messages.extend([{'role': msg['role'], 'content': msg['content']} for msg in past_context])
    if tool_output:
        messages.append({'role': 'user', 'content': str(tool_output)})
    return messages
    
def create_chat(name: str, openai_model: str, tools: Optional[pxt.tools] = None):
    """Create a chat table with optional tool support.
    
    Args:
        name: Name of the chat
        openai_model: OpenAI model to use
        tools: Optional tools to make available to the agent
    """
    # Create directory for chat
    pxt.drop_dir(name, force=True)
    pxt.create_dir(name)

    # Table to store conversation history between AI and User.
    message_table = pxt.create_table(
        path_str=f'{name}.messages',
        schema_or_df={'role': pxt.String, 'content': pxt.String}
    )

    # Chat interaction table
    chat = pxt.create_table(path_str=f'{name}.chat', schema_or_df={'system_prompt': pxt.String})

    @pxt.query
    def _get_messages():
        return message_table.select(role=message_table.role, content=message_table.content)

    # Response sequence
    chat.add_computed_column(get_messages=_get_messages())
    chat.add_computed_column(
        messages=_create_messages(chat.get_messages, chat.system_prompt)
    )
    
    if tools:
        # Add tool-enabled response sequence
        chat.add_computed_column(
            initial_tool_response=openai.chat_completions(
                messages=chat.messages,
                model=openai_model,
                tools=tools,
            )
        )
        
        chat.add_computed_column(
            invoke_tools=openai.invoke_tools(tools, chat.initial_tool_response)
        )
        
        chat.add_computed_column(
            tool_messages=_create_messages(chat.get_messages, chat.system_prompt, chat.invoke_tools)
        )
    
        
        chat.add_computed_column(
            tool_response=openai.chat_completions(model=openai_model, messages=chat.tool_messages)
        )
        
        chat.add_computed_column(response=chat.tool_response.choices[0].message.content)
    else:
        # Standard response sequence without tools
        chat.add_computed_column(
            invoke_llm=openai.chat_completions(messages=chat.messages, model=openai_model)
        )
        chat.add_computed_column(response=chat.invoke_llm.choices[0].message.content)

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

class Agent:
    """Base agent class that can be composed into workflows"""
    def __init__(self, name: str, system_prompt: str, model: str = "gpt-4o-mini", tools: Optional[pxt.tools] = None):
        self.name = name
        self.system_prompt = system_prompt
        self.model = model
        self.tools = tools
        create_chat(name, model, tools)
        
    def run(self, content: str) -> str:
        return run(self.name, self.system_prompt, content)