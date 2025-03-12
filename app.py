from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import httpx
import asyncio
import os
from typing import List, Dict, Any, Optional
import re
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.groq import GroqModel
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, UserPromptPart
from dataclasses import dataclass

# Define models for request and response
class ChatRequest(BaseModel):
    message: str
    github_token: Optional[str] = None
    groq_token: str
    history: Optional[List[Dict[str, str]]] = []

class ChatResponse(BaseModel):
    response: str
    tool_calls: Optional[List[Dict[str, Any]]] = []

# Create FastAPI app
app = FastAPI(title="GitHub Agent API")

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
    expose_headers=["*"], 
)

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

# Define tool functions for GitHub agent
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

async def get_file_content(ctx: RunContext[GitHubDeps], github_url: str, file_path: str) -> str:
    """Get the content of a specific file in a GitHub repository."""
    match = re.search(r'github\.com[:/]([^/]+)/([^/]+?)(?:\.git)?$', github_url)
    if not match:
        return "Invalid GitHub URL format"
    
    owner, repo = match.groups()
    headers = {'Authorization': f'token {ctx.deps.github_token}'} if ctx.deps.github_token else {}
    
    response = await ctx.deps.client.get(
        f'https://api.github.com/repos/{owner}/{repo}/contents/{file_path}',
        headers=headers
    )
    
    if response.status_code != 200:
        return f"Failed to get file content: {response.text}"
    
    data = response.json()
    if data.get('type') != 'file':
        return "The path does not point to a file"
    
    import base64
    content = base64.b64decode(data['content']).decode('utf-8')
    return f"File: {file_path}\n\n{content}"

async def get_issues(ctx: RunContext[GitHubDeps], github_url: str, state: str = "open") -> str:
    """Get the issues of a GitHub repository. State can be 'open', 'closed', or 'all'."""
    match = re.search(r'github\.com[:/]([^/]+)/([^/]+?)(?:\.git)?$', github_url)
    if not match:
        return "Invalid GitHub URL format"
    
    owner, repo = match.groups()
    headers = {'Authorization': f'token {ctx.deps.github_token}'} if ctx.deps.github_token else {}
    
    response = await ctx.deps.client.get(
        f'https://api.github.com/repos/{owner}/{repo}/issues?state={state}&per_page=10',
        headers=headers
    )
    
    if response.status_code != 200:
        return f"Failed to get issues: {response.text}"
    
    issues = response.json()
    if not issues:
        return f"No {state} issues found."
    
    result = []
    for issue in issues:
        # Skip pull requests
        if 'pull_request' in issue:
            continue
        
        result.append(
            f"#{issue['number']} - {issue['title']}\n"
            f"State: {issue['state']}\n"
            f"Created: {issue['created_at']}\n"
            f"URL: {issue['html_url']}\n"
        )
    
    return "\n".join(result) if result else f"No {state} issues found (excluding PRs)."

async def get_pull_requests(ctx: RunContext[GitHubDeps], github_url: str, state: str = "open") -> str:
    """Get the pull requests of a GitHub repository. State can be 'open', 'closed', or 'all'."""
    match = re.search(r'github\.com[:/]([^/]+)/([^/]+?)(?:\.git)?$', github_url)
    if not match:
        return "Invalid GitHub URL format"
    
    owner, repo = match.groups()
    headers = {'Authorization': f'token {ctx.deps.github_token}'} if ctx.deps.github_token else {}
    
    response = await ctx.deps.client.get(
        f'https://api.github.com/repos/{owner}/{repo}/pulls?state={state}&per_page=10',
        headers=headers
    )
    
    if response.status_code != 200:
        return f"Failed to get pull requests: {response.text}"
    
    prs = response.json()
    if not prs:
        return f"No {state} pull requests found."
    
    result = []
    for pr in prs:
        result.append(
            f"#{pr['number']} - {pr['title']}\n"
            f"State: {pr['state']}\n"
            f"Created: {pr['created_at']}\n"
            f"URL: {pr['html_url']}\n"
        )
    
    return "\n".join(result)

# Function to initialize GitHub agent with a specific Groq API key
def get_github_agent(api_key):
    groq_model = GroqModel('llama-3.3-70b-versatile', api_key=api_key)
    agent = Agent(
        groq_model,
        system_prompt=system_prompt,
        deps_type=GitHubDeps,
        retries=2
    )
    
    # Register tools
    agent.tool(get_repo_info)
    agent.tool(get_repo_structure)
    agent.tool(get_file_content)
    agent.tool(get_issues)
    agent.tool(get_pull_requests)
    
    return agent

# Endpoint to serve the HTML interface
@app.get("/", response_class=HTMLResponse)
async def get_html():
    with open("static/index.html", "r") as file:
        return file.read()

# API endpoint for chat
@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not request.groq_token:
        raise HTTPException(status_code=400, detail="Groq API token is required")
    
    # Initialize the GitHub agent with the provided token
    github_agent = get_github_agent(request.groq_token)
    
    # Convert history to the format expected by the agent
    message_history = []
    for msg in request.history:
        if msg["role"] == "user":
            message_history.append(ModelRequest(parts=[UserPromptPart(content=msg["content"])]))
        elif msg["role"] == "assistant":
            message_history.append(ModelResponse(parts=[TextPart(content=msg["content"])]))
    
    # Initialize dependencies
    deps = GitHubDeps(
        client=httpx.AsyncClient(),
        github_token=request.github_token,
    )
    
    try:
        # Run the agent
        result = await github_agent.run(
            request.message,
            deps=deps,
            message_history=message_history
        )
        
        # Extract tool calls
        tool_calls = []
        for msg in result.new_messages():
            if hasattr(msg, 'parts'):
                for part in msg.parts:
                    if hasattr(part, 'tool_call'):  # Check for tool call attribute
                        tool_calls.append({
                            "name": part.tool_call.function.name,  # Use the correct attribute
                            "args": part.tool_call.function.arguments
                        })
        
        return ChatResponse(
            response=result.data,
            tool_calls=tool_calls
        )
    except Exception as e:
        print(f"Error in /api/chat: {str(e)}")  # Log the error for debugging
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await deps.client.aclose()


# Run the app
if __name__ == "__main__":
    # Create static folder if it doesn't exist
    if not os.path.exists("static"):
        os.makedirs("static")
    
    # Mount static files
    app.mount("/static", StaticFiles(directory="static"), name="static")
    
    # Run the server
    uvicorn.run(app, host="0.0.0.0", port=8000)