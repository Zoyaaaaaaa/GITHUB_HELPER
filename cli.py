# from __future__ import annotations
# from dotenv import load_dotenv
# from typing import List
# import asyncio
# import logfire
# import httpx
# import os

# from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, UserPromptPart
# from github_agent import github_agent, GitHubDeps

# # Load environment variables
# load_dotenv()

# # Configure logfire to suppress warnings
# logfire.configure(send_to_logfire='never')

# class CLI:
#     def __init__(self):
#         self.messages: List[ModelMessage] = []
#         self.deps = GitHubDeps(
#             client=httpx.AsyncClient(),
#             github_token=os.getenv('GITHUB_TOKEN'),
#         )

#     async def chat(self):
#         print("GitHub Agent CLI (type 'quit' to exit)")
#         print("Enter your message:")
        
#         try:
#             while True:
#                 user_input = input("> ").strip()
#                 if user_input.lower() == 'quit':
#                     break

#                 # Run the agent with streaming
#                 result = await github_agent.run(
#                     user_input,
#                     deps=self.deps,
#                     message_history=self.messages
#                 )

#                 # Store the user message
#                 self.messages.append(
#                     ModelRequest(parts=[UserPromptPart(content=user_input)])
#                 )

#                 # Store itermediatry messages like tool calls and responses
#                 filtered_messages = [msg for msg in result.new_messages() 
#                                 if not (hasattr(msg, 'parts') and 
#                                         any(part.part_kind == 'user-prompt' or part.part_kind == 'text' for part in msg.parts))]
#                 self.messages.extend(filtered_messages)

#                 # Optional if you want to print out tool calls and responses
#                 # print(filtered_messages + "\n\n")

#                 print(result.data)

#                 # Add the final response from the agent
#                 self.messages.append(
#                     ModelResponse(parts=[TextPart(content=result.data)])
#                 )
#         finally:
#             await self.deps.client.aclose()

# async def main():
#     cli = CLI()
#     await cli.chat()

# if __name__ == "__main__":
#     asyncio.run(main())

from __future__ import annotations
from dotenv import load_dotenv
from typing import List
import asyncio
import logfire
import httpx
import os
import re

from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.groq import GroqModel
from dataclasses import dataclass

# Load environment variables
load_dotenv()

# Configure logfire to suppress warnings
logfire.configure(send_to_logfire='never')

# Initialize the Groq model with the API key
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("API key for Groq is missing!")

groq_model = GroqModel('llama-3.3-70b-versatile', api_key=groq_api_key)
github_agent = Agent(groq_model)

# GitHub Dependencies class
@dataclass
class GitHubDeps:
    client: httpx.AsyncClient
    github_token: str | None = None

# System prompt for GitHub-related queries
system_prompt = """
You are a coding expert with access to GitHub to help the user manage their repository and get information from it.

Your job is to assist the user in understanding, navigating, and managing a GitHub repository. You should only answer questions related to the repository unless otherwise directed.

You can answer questions on:
1. General information about the repository (description, language, stars, etc.).
2. The purpose or objective of the repository (What is the repo about?).
3. Detailed repository structure (files and directories).
4. Content of specific files within the repository.
5. Contributor details and how to contribute to the repository.
6. Issues and Pull Requests (open, closed, or merged).
7. License information.
8. Repository activity and history (e.g., commits, updates).
"""

# Initialize GitHub agent
github_agent = Agent(
    groq_model,
    system_prompt=system_prompt,
    deps_type=GitHubDeps,
    retries=2
)

# Define GitHub agent tools

@github_agent.tool
async def get_repo_info(ctx: RunContext[GitHubDeps], github_url: str) -> str:
    """Get repository information using GitHub API."""
    match = re.search(r'github\.com[:/]([^/]+)/([^/]+?)(?:\.git)?$', github_url)
    if not match:
        return "Invalid GitHub URL format."
    
    owner, repo = match.groups()
    headers = {'Authorization': f'token {ctx.deps.github_token}'} if ctx.deps.github_token else {}
    
    response = await ctx.deps.client.get(
        f'https://api.github.com/repos/{owner}/{repo}',
        headers=headers
    )
    
    if response.status_code != 200:
        return f"Failed to get repository info: {response.text}"
    
    data = response.json()
    size_mb = data['size'] / 1024  # Size in MB
    
    # Additional details
    owner_name = data['owner']['login']
    forks_count = data['forks_count']
    open_issues_count = data['open_issues_count']
    pull_requests_url = f"https://github.com/{owner}/{repo}/pulls"
    default_branch = data['default_branch']
    visibility = data['visibility']
    topics = ", ".join(data.get('topics', [])) if 'topics' in data else "No topics"
    
    # Safely get the license information
    license_info = data.get('license', None)
    if license_info:
        license_info = license_info.get('name', 'No license available')
    else:
        license_info = 'No license available'

    return (
        f"Repository: {data['full_name']}\n"
        f"Owner: {owner_name}\n"
        f"Description: {data['description']}\n"
        f"Size: {size_mb:.1f}MB\n"
        f"Stars: {data['stargazers_count']}\n"
        f"Forks: {forks_count}\n"
        f"Open Issues: {open_issues_count}\n"
        f"Pull Requests: {pull_requests_url}\n"
        f"Language: {data['language']}\n"
        f"Created: {data['created_at']}\n"
        f"Last Updated: {data['updated_at']}\n"
        f"Default Branch: {default_branch}\n"
        f"Visibility: {visibility}\n"
        f"Topics: {topics}\n"
        f"License: {license_info}"
    )

@github_agent.tool
async def get_repo_structure(ctx: RunContext[GitHubDeps], github_url: str) -> str:
    """Get the directory structure of a GitHub repository."""
    match = re.search(r'github\.com[:/]([^/]+)/([^/]+?)(?:\.git)?$', github_url)
    if not match:
        return "Invalid GitHub URL format"
    
    owner, repo = match.groups()
    headers = {'Authorization': f'token {ctx.deps.github_token}'} if ctx.deps.github_token else {}
    
    response = await ctx.deps.client.get(
        f'https://api.github.com/repos/{owner}/{repo}/git/trees/main?recursive=1',
        headers=headers
    )
    
    if response.status_code != 200:
        # Try with master branch if main fails
        response = await ctx.deps.client.get(
            f'https://api.github.com/repos/{owner}/{repo}/git/trees/master?recursive=1',
            headers=headers
        )
        if response.status_code != 200:
            return f"Failed to get repository structure: {response.text}"
    
    data = response.json()
    tree = data['tree']
    
    # Build directory structure
    structure = []
    for item in tree:
        if not any(excluded in item['path'] for excluded in ['.git/', 'node_modules/', '__pycache__/']):
            structure.append(f"{'📁 ' if item['type'] == 'tree' else '📄 '}{item['path']}")
    
    return "\n".join(structure)

class CLI:
    def __init__(self):
        self.messages: List[ModelMessage] = []
        self.deps = GitHubDeps(
            client=httpx.AsyncClient(),
            github_token=os.getenv('GITHUB_TOKEN'),
        )

    async def chat(self):
        print("GitHub Agent CLI (type 'quit' to exit)")
        print("Enter your message:")
        
        try:
            while True:
                user_input = input("> ").strip()
                if user_input.lower() == 'quit':
                    break

                # Run the agent with streaming
                result = await github_agent.run(
                    user_input,
                    deps=self.deps,
                    message_history=self.messages
                )

                # Store the user message
                self.messages.append(
                    ModelRequest(parts=[UserPromptPart(content=user_input)])
                )

                # Store intermediary messages like tool calls and responses
                filtered_messages = [msg for msg in result.new_messages() 
                                     if not (hasattr(msg, 'parts') and 
                                             any(part.part_kind == 'user-prompt' or part.part_kind == 'text' for part in msg.parts))]
                self.messages.extend(filtered_messages)

                print(result.data)

                # Add the final response from the agent
                self.messages.append(
                    ModelResponse(parts=[TextPart(content=result.data)])
                )
        finally:
            await self.deps.client.aclose()

async def main():
    cli = CLI()
    await cli.chat()

if __name__ == "__main__":
    asyncio.run(main())
