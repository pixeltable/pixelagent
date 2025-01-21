from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
import pixeltable as pxt
from datetime import datetime
from ..llms import LLMBaseModel

class ChatResponse(BaseModel):
    user_prompt: str
    system_prompt: str
    answer: str
    timestamp: datetime

class Agent(BaseModel):
    dir_name: str = Field(default="chatbot")
    model: LLMBaseModel
    table: Optional[pxt.Table] = None
    system_prompt: Optional[str] = None
    
    model_config = dict(arbitrary_types_allowed=True)
    
    def __init__(self, **data):
        super().__init__(**data)
        self.table = self._initialize_table()
        
    def _initialize_table(self) -> pxt.Table:
        pxt.create_dir(self.dir_name, if_exists="ignore")
        
        # Create base table with both prompt and system_prompt columns
        table = pxt.create_table(
            path_str=f"{self.dir_name}.conversation",
            schema_or_df={
                "user_prompt": pxt.String,
                "system_prompt": pxt.String,
                "timestamp": pxt.Timestamp,
            },
            if_exists="ignore",
        )
       
        # Add computed columns using model-specific formatting
        table.add_computed_column(
            messages=self.model.format_messages(table.user_prompt, table.system_prompt),
            if_exists="ignore"
        )
        
        # Get completion function and kwargs from model
        completion_fn = self.model.get_pixeltable_function()
        completion_kwargs = self.model.get_completion_kwargs()
        
        table.add_computed_column(
            response=completion_fn(
                messages=table.messages,
                **completion_kwargs
            ),
            if_exists="ignore"
        )
        
        # Extract answer consistently
        table.add_computed_column(
            answer=table.response.choices[0].message.content,
            if_exists="ignore"
        )
        
        return table
    
    def run(self, prompt: str) -> ChatResponse:
        """Send a query and get response"""
        self.table.insert([{
            "user_prompt": prompt,
            "system_prompt": self.system_prompt or "",  # Use empty string if no system prompt
            "timestamp": datetime.now()
        }])
        result = self.table.order_by(self.table.timestamp, asc=False).select(
            self.table.user_prompt,
            self.table.system_prompt,
            self.table.answer,
            self.table.timestamp
        ).limit(1).collect()
        
        # Convert the first row of the result to a dictionary and create ChatResponse
        return ChatResponse(
            user_prompt=result[0, "user_prompt"],
            system_prompt=result[0, "system_prompt"],
            answer=result[0, "answer"],
            timestamp=result[0, "timestamp"]
        )