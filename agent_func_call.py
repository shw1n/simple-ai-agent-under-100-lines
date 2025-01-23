from typing import List, Dict, Callable
from dataclasses import dataclass
from anthropic import Anthropic, HUMAN_PROMPT
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
        
    def get_tool_schemas(self) -> List[Dict]:
        calendar_schema = {
            "name": "calendar",
            "description": "Checks calendar events for a person",
            "input_schema": {
                "type": "object",
                "properties": {
                    "person_name": {
                        "type": "string",
                        "description": "The name of the person to look up"
                    }
                },
                "required": ["person_name"]
            }
        }
        
        email_schema = {
            "name": "email",
            "description": "Searches emails for a person",
            "input_schema": {
                "type": "object",
                "properties": {
                    "person_name": {
                        "type": "string",
                        "description": "The name of the person to look up"
                    }
                },
                "required": ["person_name"]
            }
        }
        
        return [calendar_schema, email_schema]
    
    def run(self, query: str) -> str:
        result = ""
        steps = []
        messages = [{
            "role": "user", 
            "content": f"Help answer this query: {query}\nRespond with DONE: [answer] if no tools needed, we're looping, or no tools left to call for new info. Be thorough and provide all answers you possibly can if there are multiple tools."
        }]
        
        while True:
            print("looping")
            response = self.client.messages.create(
                model="claude-3-5-haiku-20241022",
                max_tokens=100,
                temperature=0,
                messages=messages,
                tools=self.get_tool_schemas()
            )

            #print(response)
            message = response.content[0].text
            
            if message.startswith("DONE:"):
                return message[5:].strip()
            
            # Check for tool use blocks in the content
            for block in response.content:
                if block.type == 'tool_use':
                    tool_name = block.name.lower()
                    person_name = block.input.get("person_name")
                    # Skip if no person_name provided
                    if not person_name:
                        continue
                    for tool in self.tools:
                        if tool.name.lower() == tool_name:
                            tool_result = tool.func(person_name)
                            steps.append(f"Used {tool.name} for {person_name}")
                            result += f"\n{tool_result}"
                            
                            # Add the tool result to messages
                            messages.append({
                                "role": "assistant",
                                "content": f"I used {tool.name} for {person_name} and got this result: {tool_result}"
                            })

                            messages.append({
                                "role": "user",
                                "content": f"Continue or respond with DONE: [answer] if we have enough information."
                            })

# Simulate tools with name-based responses
def check_calendar(name: str) -> str:
    calendar_data = {
        "alice": "Meeting with clients at 2pm",
        "bob": "Team standup at 10am",
        "charlie": "Lunch meeting at 12pm",
        "default": "No meetings found"
    }
    return f"Calendar for {name}: {calendar_data.get(name.lower(), calendar_data['default'])}"

def search_email(name: str) -> str:
    email_data = {
        "alice": "Latest email about Q4 planning",
        "bob": "Project status update",
        "charlie": "Vacation request pending",
        "default": "No recent emails"
    }
    return f"Emails from {name}: {email_data.get(name.lower(), email_data['default'])}"

# Create tools list
tools = [
    Tool("calendar", "Checks calendar events for a person", check_calendar),
    Tool("email", "Searches emails for a person", search_email),
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
