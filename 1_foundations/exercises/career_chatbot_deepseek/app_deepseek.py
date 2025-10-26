"""Career Conversation Agent - DeepSeek Version
A professional AI assistant that represents you on your website.
"""

from dotenv import load_dotenv
from openai import OpenAI
from textwrap import dedent
import json
import os
import requests
from pypdf import PdfReader
import gradio as gr


class CareerAgent:
    """AI agent for career conversation with tool calling support."""

    def __init__(self, name: str = "Kin Lu"):
        """Initialize the career agent."""
        load_dotenv(override=True)
        
        self.name = name
        self.client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com/v1"
        )

        # Load profile data
        self.linkedin = self._load_linkedin()
        self.summary = self._load_summary()
        self.system_prompt = self._build_system_prompt()

        # Pushover config
        self.pushover = {
            "user": os.getenv("PUSHOVER_USER"),
            "token": os.getenv("PUSHOVER_TOKEN"),
            "url": "https://api.pushover.net/1/messages.json"
        }
    
    def _load_linkedin(self) -> str:
        """Load LinkedIn profile from PDF."""
        try:
            reader = PdfReader("assets/linkedin.pdf")
            return "\n".join(p.extract_text() for p in reader.pages if p.extract_text())
        except Exception as e:
            print(f"Error loading LinkedIn: {e}")
            return "LinkedIn profile not available."
    
    def _load_summary(self) -> str:
        """Load personal summary."""
        try:
            with open("assets/summary.txt", "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            print(f"Error loading summary: {e}")
            return "Summary not available."
    
    def _build_system_prompt(self) -> str:
        """Build the system prompt for the AI agent."""
        return dedent(f"""
            You are acting as {self.name}. You are answering questions on {self.name}'s 
            website, particularly questions related to {self.name}'s career, background, 
            skills and experience.

            Your responsibility is to represent {self.name} for interactions on the website 
            as faithfully as possible.

            Be professional and engaging, as if talking to a potential client or future 
            employer.

            If you don't know the answer to any question, use your record_unknown_question 
            tool to record it for future improvement.
            
            If the user wants to connect, ask for their email and use your 
            record_user_details tool.

            ## Summary:
            {self.summary}

            ## LinkedIn Profile:
            {self.linkedin}

            With this context, please chat with the user, always staying in character as 
            {self.name}.
        """).strip()
    
    def _send_notification(self, message: str) -> None:
        """Send push notification."""
        if self.pushover["user"] and self.pushover["token"]:
            try:
                requests.post(self.pushover["url"], data={
                    "user": self.pushover["user"],
                    "token": self.pushover["token"],
                    "message": message
                })
            except Exception as e:
                print(f"Notification error: {e}")
    
    def record_user_details(self, email: str, name: str = "Not provided",
                           notes: str = "Not provided") -> dict:
        """Tool: Record user contact details."""
        self._send_notification(f"New contact: {name} ({email})\nNotes: {notes}")
        return {"status": "recorded"}
    
    def record_unknown_question(self, question: str) -> dict:
        """Tool: Record unknown question."""
        self._send_notification(f"Unknown question: {question}")
        return {"status": "recorded"}
    
    def _get_tools(self) -> list:
        """Get tool definitions."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "record_user_details",
                    "description": "Record user contact information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "email": {"type": "string", "description": "Email address"},
                            "name": {"type": "string", "description": "User's name"},
                            "notes": {"type": "string", "description": "Additional context"}
                        },
                        "required": ["email"],
                        "additionalProperties": False
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "record_unknown_question",
                    "description": "Record unanswered questions",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "question": {"type": "string", "description": "The question"}
                        },
                        "required": ["question"],
                        "additionalProperties": False
                    }
                }
            }
        ]
    
    def _execute_tool_calls(self, tool_calls) -> list:
        """Execute tool calls."""
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            
            # Call the method
            method = getattr(self, tool_name, None)
            result = method(**arguments) if method else {"error": "Unknown tool"}
            
            results.append({
                "role": "tool",
                "content": json.dumps(result),
                "tool_call_id": tool_call.id
            })
        return results
    
    def chat(self, message: str, history: list) -> str:
        """Handle chat conversation."""
        messages = [
            {"role": "system", "content": self.system_prompt},
            *history,
            {"role": "user", "content": message}
        ]
        
        while True:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                tools=self._get_tools()
            )
            
            finish_reason = response.choices[0].finish_reason
            assistant_message = response.choices[0].message
            
            if finish_reason == "tool_calls":
                messages.append(assistant_message)
                messages.extend(self._execute_tool_calls(assistant_message.tool_calls))
            else:
                return assistant_message.content


if __name__ == "__main__":
    agent = CareerAgent(name="Kin Lu")
    gr.ChatInterface(agent.chat, type="messages").launch()