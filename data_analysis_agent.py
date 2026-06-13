from deepagents import create_deep_agent
from langchain.tools import tool
import pandas as pd
from langchain.messages import HumanMessage

from dotenv import load_dotenv
load_dotenv()

from daytona import Daytona
from langchain_daytona import DaytonaSandbox
sandbox=Daytona().create()
backend=DaytonaSandbox(sandbox=sandbox)

backend.upload_file('./first_try/data.csv', 'data.csv')

prompt='''You are a data analysis agent. Data analysis tasks typically require planning, code execution, 
and working with artifacts such as scripts, reports, and plots—capabilities that deep agents are designed to handle.

instruction: 
1. Accept a CSV file for analysis
2. Perform exploratory data analysis and generate visualizations
3. Share results to a Slack channel'''

agent=create_deep_agent(
    model="deepseek-v4-flash",
    system_prompt=prompt,
    backend=backend #把沙盒绑定给agent
)

result=agent.invoke(
    {"messages":HumanMessage("Analyze the data from the csv file.")}
)

message=result.get("messages",[])
if message:
    print(message[-1].content)
