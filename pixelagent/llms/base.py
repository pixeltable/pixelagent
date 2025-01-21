from typing import List, Dict, Any, Callable, Optional
from pydantic import BaseModel as PydanticBaseModel, Field

class LLMBaseModel(PydanticBaseModel):
    """Base model class for handling different LLM providers."""
    
    temperature: float = Field(default=0.7)
    max_tokens: Optional[int] = Field(default=None)
    top_p: float = Field(default=1.0)
    presence_penalty: float = Field(default=0.0)
    frequency_penalty: float = Field(default=0.0)
    
    def format_messages(self, prompt_col: Any, system_prompt_col: Any) -> List[Dict[str, str]]:
        """Format messages for LLM API using column expressions."""
        raise NotImplementedError("Subclasses must implement format_messages")
    
    def get_pixeltable_function(self) -> Callable:
        """Return the LLM completions function from Pixeltable."""
        raise NotImplementedError("Subclasses must implement get_pixeltable_function")
    
    def get_completion_kwargs(self) -> Dict[str, Any]:
        """Return kwargs for the LLM completion function."""
        raise NotImplementedError("Subclasses must implement get_completion_kwargs")
