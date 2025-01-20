import pixeltable as pxt
from pixeltable.functions import openai
from pydantic import BaseModel, Field
from typing import Optional

class Agent(BaseModel):
    dir_name: str = Field(default="chatbot")
    model: str = Field(default="gpt-4o-mini")
    table: Optional[pxt.Table] = None
    
    model_config = dict(arbitrary_types_allowed=True)  # Allow pixeltable.Table type
    
    def model_post_init(self, _) -> None:
        """Initialize table after Pydantic model initialization"""
        self.table = self._initialize_table()
    
    def _initialize_table(self) -> pxt.Table:
        """Initialize Pixeltable with proper schema and computed columns."""
        # Initialize Pixeltable
        pxt.drop_dir(self.dir_name, force=True)
        pxt.create_dir(self.dir_name)
        
        # Create table
        table = pxt.create_table(
            path_str=f"{self.dir_name}.conversations",
            schema_or_df={"prompt": pxt.String},
            if_exists="ignore",
        )
        
        # Add computed columns
        table.add_computed_column(
            messages=[{"role": "user", "content": table.prompt}]
        )
        
        table.add_computed_column(
            response=openai.chat_completions(
                messages=table.messages,
                model=self.model,
            )
        )
        
        table.add_computed_column(
            answer=table.response.choices[0].message.content
        )
        
        return table
    
    def run(self, prompt: str) -> str:
        """Send a query and get response"""
        self.table.insert([{"prompt": prompt}])
        result = self.table.select(
            self.table.prompt, 
            self.table.answer
        ).collect()
        return result[-1]['answer']