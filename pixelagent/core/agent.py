from typing import Optional, Dict, Any, List, Callable
from pydantic import BaseModel, Field
import pixeltable as pxt
from datetime import datetime
from ..llms import LLMBaseModel

class ChatResponse(BaseModel):
    user_prompt: str
    system_prompt: str
    answer: str
    timestamp: datetime
    tool_outputs: Optional[Dict] = None

class Agent(BaseModel):
    dir_name: str = Field(default="chatbot")
    model: LLMBaseModel
    table: Optional[pxt.Table] = None
    system_prompt: Optional[str] = None
    tools: Optional[List[Callable]] = None
    
    model_config = dict(arbitrary_types_allowed=True)
    
    def __init__(self, **data):
        super().__init__(**data)
        self.table = self._initialize_table()
        
    def _initialize_table(self) -> pxt.Table:
        pxt.create_dir(self.dir_name, if_exists="ignore")
        
        # Create base table
        table = pxt.create_table(
            path_str=f"{self.dir_name}.conversation",
            schema_or_df={
                "user_prompt": pxt.String,
                "system_prompt": pxt.String,
                "timestamp": pxt.Timestamp,
            },
            if_exists="ignore",
        )
        
        # Convert tools to pixeltable tools
        tools = pxt.tools(*self.tools) if self.tools else None
        
        # Add computed columns using model-specific formatting
        table.add_computed_column(
            messages=self.model.format_messages(table.user_prompt, table.system_prompt),
            if_exists="ignore"
        )
        
        # Get completion function and kwargs from model
        completion_fn = self.model.get_pixeltable_function()
        completion_kwargs = self.model.get_completion_kwargs()
        if tools:
            completion_kwargs['tools'] = tools
        
        # Initial completion with tools
        table.add_computed_column(
            initial_response=completion_fn(
                messages=table.messages,
                **completion_kwargs
            ),
            if_exists="ignore"
        )
        
        # Add tool execution if tools are provided
        if tools:
            table.add_computed_column(
                tool_outputs=pxt.functions.openai.invoke_tools(
                    tools, table.initial_response
                ),
                if_exists="ignore"
            )
            
            # Format prompt with tool outputs
            table.add_computed_column(
                tool_response_prompt=self._create_tool_prompt(
                    table.user_prompt, table.tool_outputs
                ),
                if_exists="ignore"
            )
            
            # Final completion with tool outputs
            table.add_computed_column(
                final_response=completion_fn(
                    messages=[
                        {
                            "role": "system",
                            "content": "Answer the user's question based on the provided tool outputs.",
                        },
                        {"role": "user", "content": table.tool_response_prompt},
                    ],
                    **completion_kwargs
                ),
                if_exists="ignore"
            )
            
            # Extract final answer
            table.add_computed_column(
                answer=table.final_response.choices[0].message.content,
                if_exists="ignore"
            )
        else:
            # If no tools, use original answer extraction
            table.add_computed_column(
                answer=table.initial_response.choices[0].message.content,
                if_exists="ignore"
            )
            
        return table
    
    @staticmethod
    @pxt.udf
    def _create_tool_prompt(question: str, tool_outputs: List[Dict]) -> str:
        return f"""
        QUESTION:
        {question}
        
        TOOL OUTPUTS:
        {tool_outputs}
        """
    
    def run(self, prompt: str) -> ChatResponse:
        """Send a query and get response"""
        self.table.insert([{
            "user_prompt": prompt,
            "system_prompt": self.system_prompt or "",
            "timestamp": datetime.now()
        }])
        
        result = self.table.order_by(self.table.timestamp, asc=False).select(
            self.table.user_prompt,
            self.table.system_prompt,
            self.table.answer,
            self.table.timestamp,
            self.table.tool_outputs if self.tools else None
        ).limit(1).collect()
        
        tool_outputs = result[0, "tool_outputs"]
        if tool_outputs and isinstance(tool_outputs, list):
            tool_outputs = tool_outputs[0]  # Extract the dictionary from the list
        
        return ChatResponse(
            user_prompt=result[0, "user_prompt"],
            system_prompt=result[0, "system_prompt"],
            answer=result[0, "answer"],
            timestamp=result[0, "timestamp"],
            tool_outputs=tool_outputs
        )