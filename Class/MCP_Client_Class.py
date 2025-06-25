
import sys
from typing import Optional
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()  # load environment variables from .env

class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.llm = OpenAI()

    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server
        
        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")
            
        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )
        
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        
        await self.session.initialize()
        
        # List available tools
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])

        #List available resources
        response = await self.session.list_resources()
        resources = response.resources
        print("\nAvailable resources:", [resource.name for resource in resources])

    async def process_query(self, messages: str) -> str:
        """Process a query using Claude and available tools"""

        response = await self.session.list_tools()
        available_tools = [{
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.inputSchema
            }} for tool in response.tools]
        
        while True:
            response = self.llm.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                tools=available_tools,
                tool_choice="auto"
            )
            
            msg = response.choices[0].message
            
            
            if msg.tool_calls:
                tool_call = msg.tool_calls[0]
                func_name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)
                
                result = (await self.session.call_tool(func_name, args)).content[0].text
                
                messages.append({"role": "assistant", "tool_calls": [tool_call]})
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })
                continue
            else:
                # No more tool calls → model has responded
                messages.append(msg)
                break
        return messages[len(messages)-1].content

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")
        messages = []
        while True:
            try:
                query = input("\nQuery: ").strip()
                
                if query.lower() == 'quit':
                    break
                    
                messages.append(
                    {
                        "role": "user",
                        "content": query
                    }
                )

                response = await self.process_query(messages)
                print("\n" + response)
                    
            except Exception as e:
                print(f"\nError: {str(e)}")
    
    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()

async def main(path = sys.argv):
    if path == sys.argv:
        if len(path) < 2:
            print("Usage: python client.py <path_to_server_script>")
            sys.exit(1)
        
    client = MCPClient()
    try:
        await client.connect_to_server(path)
        await client.chat_loop()
    finally:
        await client.cleanup()
