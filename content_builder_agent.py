from deepagents import create_deep_agent
from langchain.tools import tool
from tavily import TavilyClient
import os
from google import genai
from deepagents.backends import FilesystemBackend
import yaml
import sys
from langchain.messages import HumanMessage

from dotenv import load_dotenv
load_dotenv()

from pathlib import Path
EXAMPLE_DIR = Path(__file__).parent

@tool
def web_search(query, max_results=3,topic="general"):
    '''Search the web for information on the query
    args:
        query: the search prompt
        max_results: number of results to return
        topic: "general" for common queries, "news" for current excerpts'''
    api_key=os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return {"error": "TAVILY_API_KEY not set"}
    return TavilyClient(api_key=api_key).search(query, max_results=max_results, topic=topic)

@tool
def generate_cover(prompt, slug):
    '''generate the cover image for a social media post.
    args:
        prompt: detailed description of the image to generate
        slug: image save to blogs/<slug>/hero.png'''
    client=genai.Client()
    response=client.models.generate_content(
        model="gemini-2.5-flash-image",
        contents=[prompt]
    )
    output_path = EXAMPLE_DIR / "blogs" / slug / "hero.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    response.parts[0].as_image().save(output_path)
    return f"Image saved to {output_path}"

@tool
def generate_social_image(prompt, platform, slug):
    '''generate image for a social media post.
    args:
        prompt: detailed description of the image to generate
        platform: either "linkedin" or "tweets"
        slug: image save to <platform>/<slug>/image.png'''
    client=genai.Client()
    response=client.models.generate_content(
        model="gemini-2.5-flash-image",
        contents=[prompt]
    )
    output_path = EXAMPLE_DIR / "blogs" / slug / "hero.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    response.parts[0].as_image().save(output_path)
    return f"Image saved to {output_path}"

def load_subagents(config_path: Path) -> list:
    """Load subagent definitions from YAML and wire up tools.

    Unlike `memory` and `skills`, deep agents do not load subagents from files by default.
    This helper externalizes configuration so you can edit YAML without changing Python code.
    """
    available_tools = {
        "web_search": web_search,
    }
    with open(config_path) as f:
        config = yaml.safe_load(f)
    subagents = []
    for name, spec in config.items():
        subagent = {
            "name": name,
            "description": spec["description"],
            "system_prompt": spec["system_prompt"],
        }
        if "model" in spec:
            subagent["model"] = spec["model"]
        if "tools" in spec:
            subagent["tools"] = [available_tools[t] for t in spec["tools"]]
        subagents.append(subagent)
    return subagents

agent=create_deep_agent(
    model="deepseek-v4-flash",
    tools=[web_search, generate_cover, generate_social_image],
    memory=["./AGENTS.md"],
    skills=["./skills/"],
    subagents=load_subagents(EXAMPLE_DIR / "subagents.yaml"),
    backend=FilesystemBackend(root_dir=EXAMPLE_DIR),
)

task = (
        " ".join(sys.argv[1:])
        if len(sys.argv) > 1
        else "Write a blog post about how AI agents are transforming software development"
    )

result=agent.invoke(
        {"messages": [HumanMessage(content=task)]},
        config={"configurable": {"thread_id": "content-builder-demo"}}
        #没有checkpointer记忆就停留在这次的脚本运行中；有的话就是永久存档
        )

for msg in result.get("messages", []):
    if hasattr(msg, "content") and msg.content:
        print(msg.content)
        

