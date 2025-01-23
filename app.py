from typing import List, Dict, Callable
from dataclasses import dataclass
from anthropic import Anthropic
import os
import dotenv

dotenv.load_dotenv()

@dataclass
class Tool:
    name: str
    description: str
    func: Callable

class Agent:
    def __init__(self, tools: List[Tool]):

        self.tools = tools
        self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        
    def get_tool_descriptions(self) -> str:
        return "\n".join([
            f"- {tool.name}: {tool.description}"
            for tool in self.tools
        ])
    
    def run(self, query: str) -> str:
        result = ""
        steps = []
        
        while True:
            prompt = f"""You are an agent helping answer a user query.
            Available tools: {self.get_tool_descriptions()}
            
            Original query: {query}
            Current results: {result}
            Steps taken: {', '.join(steps) if steps else 'None'}
            
            If you have enough information to answer the query, respond with: DONE: [final answer]
            Otherwise, respond with the name of the next tool to use (just the tool name)."""

            response = self.client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=100,
                temperature=0,
                messages=[{"role": "user", "content": prompt}]
            )
            
            answer = response.content[0].text.strip()
            
            if answer.startswith("DONE:"):
                return answer[5:].strip()  # Return everything after "DONE:"
            
            # Find and execute the matching tool
            tool_name = answer.lower()
            for tool in self.tools:
                if tool.name.lower() == tool_name:
                    tool_result = tool.func(query)
                    steps.append(f"Used {tool.name}")
                    result += f"\n{tool_result}"
                    break

# Simulate tools (not actually implemented)
def check_calendar(task: str) -> str:
    return "Calendar shows: Meeting at 2pm"

def search_email(task: str) -> str:
    return "Found email from Bob about project deadline"

# Create tools list
tools = [
    Tool("calendar", "Checks calendar events", check_calendar),
    Tool("email", "Searches emails", search_email),
]

def main():
    agent = Agent(tools)
    
    while True:
        query = input("What would you like to know? (or 'quit' to exit): ")
        if query.lower() == 'quit':
            break
            
        result = agent.run(query)
        print(result)

if __name__ == "__main__":
    main()
