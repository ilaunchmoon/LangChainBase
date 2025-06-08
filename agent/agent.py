from langchain.prompts import (
    ChatPromptTemplate,         
    PromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,                # 用于历史对话消息的占位符
)
from dotenv import load_dotenv
from datetime import datetime
from langchain.tools import tool 
from pydantic import BaseModel, Field
from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor                  # agent的执行函数
from langchain_openai import ChatOpenAI                     # agent的所使用的llm

# 加载环境变量: 注意是OpenAI key
load_dotenv()

system_message = """你是一个有用的AI助手。你可以回答问题并执行工具来获取信息。
你有两个工具可用：
1. get_curr_time - 获取当前时间
2. get_weather_tool - 获取指定位置的天气

如果你需要获取当前时间, 请使用get_curr_time工具。
如果你需要获取天气信息, 请使用get_weather_tool工具。
如果你不需要使用任何工具，可以直接回答用户的问题。"""


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_message),
        MessagesPlaceholder(variable_name='chat_history', optional=True),       # 用于历史对话消息的占位符   
        HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['input'], template='{input}')),  # 输入用户提示词
        MessagesPlaceholder(variable_name='agent_scratchpad')                                              # agent的输出消息
    ]
)


# 定义查询天气所在地址的类
class WeatherSchema(BaseModel):
    locations:str = Field(description="城市或县区, 比如上海市, 深圳罗湖区等")

@tool("get_weather_tool", args_schema=WeatherSchema)
def get_curr_weather_tool(location:str):
    """
        查询天气的工具
    """
    return f"{location}是晴天"

@tool("get_curr_time")
def get_curr_time():
    """
        获取当前时间的工具
    """
    curr_datatime = datetime.now()

    # 格式化时间
    format_time = curr_datatime.strftime('%Y-%m-%d %H:%M:%S')
    return f"当地时间: {format_time}"


# 定义需要调用的工具集合
tools = [get_curr_weather_tool, get_curr_time]

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# 定义agent: 必须输入三项内容: llm、工具集、提示词
agent = create_tool_calling_agent(llm, tools=tools, prompt=prompt)

# 定义agent的执行器, 必须输入agent实例, 和agent需要使用的工具集合, verbose代表开启监控日志
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 执行agent
response = agent_executor.invoke({"input": "现在几点, 上海的天气如何"})
print(response)

