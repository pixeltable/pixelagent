from typing import List, Dict, Any, Callable, Optional
from pydantic import  Field
from pixeltable.functions import openai
from .base import LLMBaseModel

class OpenAIModel(LLMBaseModel):
    """Model class for handling OpenAI chat completions with Pixeltable."""
    
    model_name: str = Field(default="gpt-4")
    temperature: float = Field(default=0.7)
    max_tokens: Optional[int] = Field(default=None)
    top_p: float = Field(default=1.0)
    presence_penalty: float = Field(default=0.0)
    frequency_penalty: float = Field(default=0.0)
    
    def format_messages(self, prompt_col: Any, system_prompt_col: Any) -> List[Dict[str, str]]:
        """Format messages for OpenAI chat completion API using column expressions."""
        # Create a list containing both messages
        messages = [
            {"role": "developer", "content": system_prompt_col},
            {"role": "user", "content": prompt_col}
        ]
        return messages
    
    def get_pixeltable_function(self) -> Callable:
        """Return the OpenAI chat completions function from Pixeltable."""
        return openai.chat_completions
    
    def get_completion_kwargs(self) -> Dict[str, Any]:
        """Return kwargs for the OpenAI completion function."""
        kwargs = {
            "model": self.model_name,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
        }
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        return kwargs