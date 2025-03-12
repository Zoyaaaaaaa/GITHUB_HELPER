
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.groq import GroqModel
import os
import re
from dataclasses import dataclass
from dotenv import load_dotenv
import httpx

# Load environment variables from the .env file
load_dotenv()

# Initialize the Groq model with the API key
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("API key for Groq is missing!")

model = GroqModel('llama-3.3-70b-versatile', api_key=groq_api_key)
agent = Agent(model)

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

Do not ask the user questions before taking an action. Instead, always examine the repository using the provided tools before answering the user's question unless you already have the information.

When answering a question, always start your answer with the full repo URL in brackets and then provide your answer on the next line. For example:

[Using https://github.com/[repo URL from the user]]

Your answer should be clear and informative. If you cannot find the required information, explain why you were unable to do so, and suggest alternative ways to obtain the details.

Here are some example questions you can answer:
- "What is this repository about?"
- "Tell me more about the repository's purpose."
- "Show me the directory structure of the repository."
- "What is the README file about?"
- "What are the issues in this repository?"
- "What is the repository license?"
- "Who are the contributors?"
- "Has this repository had any recent commits?"
"""


# Initialize GitHub agent
github_agent = Agent(
    model,
    system_prompt=system_prompt,
    deps_type=GitHubDeps,
    retries=2
)


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


# Corrected instantiation of RunContext
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

# Main function to demonstrate the agent and GitHub queries
async def main():
    # Query with Groq Model
    result = await agent.run('What is the capital of France?')
    print(f"Capital of France: {result.data}")
    #> Paris

    # Using async stream with Groq model
    async with agent.run_stream('What is the capital of the UK?') as response:
        print(f"Capital of the UK: {await response.get_data()}")

    # Example of querying GitHub-related tools
    github_url = "https://github.com/Zoyaaaaaaa/DealDetective"
    github_token = os.getenv("GITHUB_TOKEN")  # Ensure you have a GitHub token
    
    async with httpx.AsyncClient() as client:
        deps = GitHubDeps(client=client, github_token=github_token)

        # Correctly pass retry, messages, and tool_name to RunContext
        context = RunContext(deps=deps, retry=2, messages=[], tool_name="get_repo_info")
        
        # Get repository information
        repo_info = await get_repo_info(context, github_url)
        print(repo_info)

        # Get repository structure
        repo_structure = await get_repo_structure(context, github_url)
        print(repo_structure)

        # Get specific file content from the repository
        # file_content = await get_file_content(context, github_url, "README.md")
        # print(file_content)

# Run the main async function
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
