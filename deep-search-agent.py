from deepagents import create_deep_agent
from langchain_tavily import TavilySearch
from langchain.tools import tool
from langchain.messages import HumanMessage

from dotenv import load_dotenv
load_dotenv()

tavily=TavilySearch(
    max_result=3,
    topic="general",
    include_raw_content=True #官方推荐使用 Tavily 来获取完整的网页内容
)

@tool
def web_search(query, max_results=1, topic="general"):
    '''
    search the web for information on the query.
    args:
        query: what the user want to search on.
        max_results: Maximum number of results to return
        topic: Topic filter - 'general', 'news', or 'finance'
    '''
    return tavily.invoke(query)

prompt='''
    role:
        You are a search agent.
    requirement:
        1. Plan research using a todo list
        2. Delegate focused research tasks to sub-agents with isolated context
        3. Assess search results and plan next steps as you gather information
        4. Synthesize findings with proper citations into a final report
    subagent delegation:
        Start with 1 sub-agent** for most queries,
        ONLY parallelize when the query EXPLICITLY requires comparison or has clearly independent aspects.
    '''

subagent_prompt='''
    You are to gather information on the user's topiv using the tools.
    Follow these steps:
        1. **Read the question carefully** - What specific information does the user need?
        2. **Start with broader searches** - Use broad, comprehensive queries first
        3. **After each search, pause and assess** - Do I have enough to answer? What's still missing?
        4. **Execute narrower searches as you gather information** - Fill in the gaps
        5. **Stop when you can answer confidently** - Don't keep searching for perfection
    '''

research_subagent={
    "name": "research_agent",
    "description": "Delegate one research topic at a time to the sub-agent",
    "system_prompt": subagent_prompt,
    "model": "deepseek-v4-flash",
    "tools": [web_search]
}

agent=create_deep_agent(
    model="deepseek-v4-flash",
    tools=[web_search],
    system_prompt=prompt,
    subagents=[research_subagent]
)

result = agent.invoke(
    {"messages": HumanMessage("What are the main differences between RAG and fine-tuning for LLM applications?")}
    )

messages = result.get("messages", [])

if messages:
    final_message = messages[-1]
    print(final_message.content)


