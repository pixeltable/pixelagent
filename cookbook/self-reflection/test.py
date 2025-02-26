import inspect
import json
from datetime import datetime
from functools import wraps
from typing import Callable, Dict, List, Optional, Type, Union, get_type_hints, Any
import time
import re

from openai import OpenAI
from pydantic import BaseModel
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich import print as rprint
from duckduckgo_search import DDGS

from pixelagent.utils import setup_pixeltable
from pixelagent.openai import Agent, tool

# Define Pydantic models for structured output
class ReflectionState(BaseModel):
    current_iteration: int
    max_iterations: int
    question: str
    current_answer: str
    reflections: List[str] = []
    needs_improvement: bool = True
    improvement_reason: Optional[str] = None
    search_queries: List[str] = []
    final_answer: Optional[str] = None
    confidence_score: float = 0.0
    improvement_threshold: float = 0.2  # If confidence improves by less than this, we stop

class ImprovementData(BaseModel):
    needs_improvement: bool = False
    reason: Optional[str] = None
    search_queries: List[str] = []
    confidence_score: float = 0.0

@tool
def search_the_web(keywords: str, max_results: int) -> str:
    """Search the web using DuckDuckGo and return results."""
    try:
        with DDGS() as ddgs:
            results = ddgs.news(
                keywords=keywords,
                region="wt-wt",
                safesearch="off",
                timelimit="m",
                max_results=max_results,
            )
            formatted_results = []
            for i, r in enumerate(results, 1):
                formatted_results.append(
                    f"{i}. Title: {r['title']}\n"
                    f"   Source: {r['source']}\n"
                    f"   Published: {r['date']}\n"
                    f"   Snippet: {r['body']}\n"
                )
            return "\n".join(formatted_results)
    except Exception as e:
        return f"Search failed: {str(e)}"

class SelfReflectingAgent:
    def __init__(
        self,
        name: str,
        base_model: str = "gpt-4o-mini",
        reflection_model: Optional[str] = None,
        max_iterations: int = 5,
        min_confidence_threshold: float = 0.7,  # Stop if we reach this confidence
        improvement_threshold: float = 0.2,     # Stop if improvement is less than this
        tools: List[Callable] = None,
        verbose: bool = True,
    ):
        self.name = name
        self.base_model = base_model
        self.reflection_model = reflection_model or base_model
        self.max_iterations = max_iterations
        self.min_confidence_threshold = min_confidence_threshold
        self.improvement_threshold = improvement_threshold
        self.tools = tools if tools else []
        self.verbose = verbose
        
        # Create base agent
        self.base_agent = Agent(
            name=f"{name}_base",
            model=base_model,
            system_prompt="""You are a helpful assistant that can search the web for information.
Your goal is to provide accurate, well-reasoned answers to user questions.
If you are unsure about something, you should search for more information.
It's OK to acknowledge the limitations of your knowledge cutoff date for very recent events.
When you don't know something for sure, clearly state your uncertainty and provide your best estimate based on available information.""",
            tools=self.tools,
            reset=True,
            enable_observability=verbose,
        )
        
        # Create reflection agent
        self.reflection_agent = Agent(
            name=f"{name}_reflection",
            model=self.reflection_model,
            system_prompt="""You are a critical evaluator of an AI assistant's responses.
Your job is to carefully analyze an answer for:
1. Factual accuracy - Are there any incorrect statements or unverified claims?
2. Completeness - Has the assistant fully answered all parts of the question?
3. Relevance - Is the answer focused on what was asked?
4. Clarity - Is the answer clear and well-organized?

For each issue you find, clearly explain what the problem is and suggest an improvement.

IMPORTANT: For factual accuracy issues related to recent events, acknowledge the LLM knowledge cutoff limitation.
If the answer already acknowledges uncertainty about recent events, consider this appropriate.
Provide a confidence score from 0.0 to 1.0 that reflects the overall quality of the answer.

Your feedback should be balanced - focus on major issues rather than minor imperfections.""",
            reset=True,
            enable_observability=verbose,
        )
        
        # Create structured improvement agent with Pydantic model
        self.improvement_agent = Agent(
            name=f"{name}_improvement_extractor",
            model=self.base_model,
            system_prompt="""You analyze reflections and extract actionable improvements in a structured format.
Your output MUST be valid JSON that matches the expected schema exactly.
Only include the JSON output with no explanations, markdown formatting, or other text.""",
            reset=True,
            enable_observability=False,
        )
        
        # Create console for display
        self.console = Console() if verbose else None

    def _display_reflection_state(self, state: ReflectionState):
        """Display the current reflection state."""
        if not self.verbose or not self.console:
            return
        
        table = Table(title=f"Reflection State (Iteration {state.current_iteration}/{state.max_iterations})")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Question", state.question)
        table.add_row("Current Answer", state.current_answer[:200] + "..." if len(state.current_answer) > 200 else state.current_answer)
        table.add_row("Needs Improvement", str(state.needs_improvement))
        table.add_row("Confidence Score", f"{state.confidence_score:.2f}")
        if state.improvement_reason:
            table.add_row("Improvement Reason", state.improvement_reason)
        if state.search_queries:
            table.add_row("Search Queries", "\n".join(state.search_queries[-3:]) + f" (total: {len(state.search_queries)})")
        if state.final_answer:
            table.add_row("Final Answer", state.final_answer[:200] + "..." if len(state.final_answer) > 200 else state.final_answer)
        
        self.console.print(table)
        
        if state.reflections:
            self.console.print(Panel(
                state.reflections[-1], 
                title=f"Latest Reflection (Iteration {state.current_iteration})", 
                border_style="yellow"
            ))
            
    def _extract_json_from_text(self, text):
        """
        Extract JSON from text that might contain additional content or formatting.
        """
        # Try to find JSON content between ```json and ``` markers first
        json_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
        json_match = re.search(json_pattern, text)
        
        if json_match:
            json_str = json_match.group(1)
        else:
            # Look for content that appears to be JSON (starts with { and ends with })
            json_pattern = r"\s*(\{[\s\S]*\})\s*"
            json_match = re.search(json_pattern, text)
            if json_match:
                json_str = json_match.group(1)
            else:
                # If we can't find JSON, return the original text
                json_str = text
                
        # Clean the string by removing any non-JSON content
        json_str = json_str.strip()
        
        return json_str

    def _extract_improvement_data(self, reflection, current_confidence):
        """
        Extract improvement data with robust error handling.
        """
        # Default values if extraction fails
        default_data = ImprovementData(
            needs_improvement=True if current_confidence < self.min_confidence_threshold else False,
            reason="Continuing improvement process",
            confidence_score=current_confidence
        )
        
        # Prepare a clear, structured prompt
        improvement_prompt = f"""Based on this reflection:
{reflection}

Extract the following information:
1. Whether the current answer needs improvement (true/false)
2. If improvement is needed, a brief reason why
3. If improvement is needed, suggest up to 3 search queries that would help
4. A confidence score from 0.0 to 1.0 (currently estimated at {current_confidence:.2f})

YOUR RESPONSE MUST BE VALID JSON IN THIS EXACT FORMAT:
```json
{{
  "needs_improvement": true,
  "reason": "brief reason text",
  "search_queries": ["query 1", "query 2", "query 3"],
  "confidence_score": 0.75
}}
```

ONLY output the JSON with no additional text, explanations, or formatting."""

        try:
            # Get raw response
            improvement_response = self.improvement_agent.run(improvement_prompt)
            
            # Extract JSON from the response
            json_str = self._extract_json_from_text(improvement_response)
            
            # Try to parse the JSON
            improvement_data = json.loads(json_str)
            
            # Validate required fields
            if "needs_improvement" not in improvement_data:
                improvement_data["needs_improvement"] = default_data.needs_improvement
            
            if "confidence_score" not in improvement_data:
                improvement_data["confidence_score"] = current_confidence
            
            # Return a proper object
            return ImprovementData(
                needs_improvement=improvement_data.get("needs_improvement", default_data.needs_improvement),
                reason=improvement_data.get("reason", default_data.reason),
                search_queries=improvement_data.get("search_queries", []),
                confidence_score=improvement_data.get("confidence_score", current_confidence)
            )
            
        except Exception as e:
            if self.verbose:
                self.console.print(f"[bold red]Error extracting improvement data: {str(e)}[/bold red]")
            return default_data

    def _extract_confidence_score(self, reflection, previous_confidence):
        """
        Extract confidence score from reflection text with robust error handling.
        """
        try:
            # Try explicit pattern matching first
            confidence_pattern = r"confidence(?:\s+)?score(?:\s+)?(?::|\s+is|\s+of)?(?:\s+)?(\d+(?:\.\d+)?)"
            confidence_match = re.search(confidence_pattern, reflection.lower())
            
            if confidence_match:
                try:
                    score = float(confidence_match.group(1))
                    # Validate the score is in valid range
                    if 0 <= score <= 1:
                        return score
                    elif 1 < score <= 10:  # Handle cases where score is on 0-10 scale
                        return score / 10
                except:
                    pass
            
            # If pattern matching fails, try a dedicated extraction
            estimate_prompt = f"""Based on this reflection, extract only the confidence score from 0.0 to 1.0:
{reflection}

Respond with ONLY a number between 0.0 and 1.0."""
            
            confidence_response = self.reflection_agent.run(estimate_prompt)
            # Try to clean and parse the response
            confidence_response = confidence_response.strip()
            # Extract just the first number that appears
            number_match = re.search(r"(\d+(?:\.\d+)?)", confidence_response)
            if number_match:
                score = float(number_match.group(1))
                # Validate the score is in valid range
                if 0 <= score <= 1:
                    return score
                elif 1 < score <= 10:  # Handle cases where score is on 0-10 scale
                    return score / 10
            
            # If all else fails, increment previous confidence slightly
            return min(previous_confidence + 0.1, 1.0)
            
        except Exception as e:
            if self.verbose:
                self.console.print(f"[bold yellow]Error extracting confidence score: {str(e)}[/bold yellow]")
            return min(previous_confidence + 0.1, 1.0)

    def run(self, question: str) -> str:
        """Run the self-reflecting agent with the given question."""
        start_time = time.time()
        
        if self.verbose and self.console:
            self.console.print(Panel(f"Starting self-reflection for question: {question}", 
                                    title="Self-Reflecting Agent", 
                                    border_style="bold magenta"))
        
        # Initialize reflection state
        state = ReflectionState(
            current_iteration=1,
            max_iterations=self.max_iterations,
            question=question,
            current_answer="",
            reflections=[],
            needs_improvement=True,
            improvement_threshold=self.improvement_threshold,
            confidence_score=0.0
        )
        
        # First pass - get initial answer
        initial_answer = self.base_agent.run(question)
        state.current_answer = initial_answer
        
        # Display initial state
        self._display_reflection_state(state)
        
        # Track previous confidence score
        previous_confidence = 0.0
        repeated_search_count = {}  # Track repeated search queries
        
        # Reflection loop
        while state.needs_improvement and state.current_iteration < state.max_iterations:
            # Get reflection
            reflection_prompt = f"""Question: {state.question}

Current Answer:
{state.current_answer}

Analyze this answer critically. Is it accurate, complete, relevant, and clear?
If there are issues, explain each one and suggest specific improvements or search queries.
Remember that for very recent events (after Oct 2024), the LLM has a knowledge cutoff limitation.
If the answer already acknowledges uncertainty about recent events, this is appropriate.

IMPORTANTLY: Provide a confidence score from 0.0 to 1.0 that reflects the overall quality of the answer.
Format your confidence score as "Confidence Score: X.X" at the end of your reflection."""

            reflection = self.reflection_agent.run(reflection_prompt)
            state.reflections.append(reflection)
            
            # Extract confidence score
            current_confidence = self._extract_confidence_score(reflection, previous_confidence)
            state.confidence_score = current_confidence
            
            # Check if we've reached confidence threshold
            if state.confidence_score >= self.min_confidence_threshold:
                state.needs_improvement = False
                state.improvement_reason = f"Reached sufficient confidence ({state.confidence_score:.2f})"
                state.final_answer = state.current_answer
                break
                
            # Check if improvement is significant
            confidence_improvement = state.confidence_score - previous_confidence
            if state.current_iteration > 1 and confidence_improvement < state.improvement_threshold:
                state.needs_improvement = False
                state.improvement_reason = f"Minimal improvement in confidence ({confidence_improvement:.2f})"
                state.final_answer = state.current_answer
                break
                
            previous_confidence = state.confidence_score
            
            # Extract improvement data
            improvement_data = self._extract_improvement_data(reflection, current_confidence)
            
            # Update state from improvement data
            state.needs_improvement = improvement_data.needs_improvement
            state.improvement_reason = improvement_data.reason
            
            # Process search queries with deduplication
            for query in improvement_data.search_queries:
                if query in repeated_search_count:
                    repeated_search_count[query] += 1
                    # Skip if we've tried this query too many times
                    if repeated_search_count[query] > 2:
                        continue
                else:
                    repeated_search_count[query] = 1
                    state.search_queries.append(query)
            
            # If improvement needed, run another iteration
            if state.needs_improvement:
                # Prepare improved prompt with reflections and search results
                improved_prompt = f"""Original Question: {state.question}

Your current answer:
{state.current_answer}

Reflection on your current answer:
{reflection}

Improvement needed: {state.improvement_reason}

"""
                
                # Run searches if queries were provided and not running too long
                elapsed_time = time.time() - start_time
                if state.search_queries and elapsed_time < 60:  # Add a time limit
                    # Only use the most recent queries
                    recent_queries = state.search_queries[-3:]
                    for query in recent_queries:
                        search_results = search_the_web(keywords=query, max_results=3)
                        improved_prompt += f"\nSearch results for '{query}':\n{search_results}\n"
                
                improved_prompt += f"""
Based on the reflection and additional information, please provide an improved answer to the original question.

IMPORTANT GUIDELINES:
1. If there are factual uncertainties about recent events (after Oct 2024), acknowledge your knowledge limitations
2. It's better to be honest about uncertainty than to make definitive claims you're not sure about
3. Focus on addressing the specific issues mentioned in the reflection
4. Maintain clarity and conciseness in your response"""
                
                # Get improved answer
                improved_answer = self.base_agent.run(improved_prompt)
                state.current_answer = improved_answer
            else:
                # No improvement needed, set final answer
                state.final_answer = state.current_answer
            
            # Increment iteration counter
            state.current_iteration += 1
            
            # Display updated state
            self._display_reflection_state(state)
            
            # Time-based exit condition to prevent very long runs
            if time.time() - start_time > 120:  # 2 minute maximum
                if self.verbose:
                    self.console.print("[bold yellow]Exiting due to time limit[/bold yellow]")
                break
        
        # Set final answer if we've reached max iterations
        if state.current_iteration >= state.max_iterations and not state.final_answer:
            final_prompt = f"""Original Question: {state.question}

Your current answer:
{state.current_answer}

You've gone through several iterations of reflection. Please provide your final, best answer based on all the information and reflections so far.

IMPORTANT: Acknowledge any remaining uncertainties, especially about recent events beyond your knowledge cutoff. It's better to be honest about limitations than to make definitive claims you're not sure about."""

            state.final_answer = self.base_agent.run(final_prompt)
            state.current_iteration = state.max_iterations
            self._display_reflection_state(state)
        
        if self.verbose and self.console:
            elapsed_time = time.time() - start_time
            self.console.print(Panel(
                state.final_answer or state.current_answer,
                title=f"Final Answer (after {state.current_iteration} iterations, {elapsed_time:.1f}s)",
                border_style="bold green"
            ))
        
        return state.final_answer or state.current_answer

# Example usage
if __name__ == "__main__":
    # Create the self-reflecting agent
    reflecting_agent = SelfReflectingAgent(
        name="web_reflection_agent",
        base_model="gpt-4o-mini",
        max_iterations=5,
        min_confidence_threshold=0.8,  # Stop if we reach this confidence
        improvement_threshold=0.1,     # Stop if improvement is less than this
        tools=[search_the_web],
        verbose=True,
    )
    
    # Ask a question
    question = "What's the latest news in Denver? Who won the Super Bowl?"
    answer = reflecting_agent.run(question)
    print("\n\nFinal answer:")
    print(answer)