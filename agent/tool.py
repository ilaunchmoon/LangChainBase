"""
    Agent之使用工具
        
        1. 调用LangChain内置集成工具
        2. LLM function-calling
        3. 定义和调用自定义工具的三种方式
        4. LLM绑定需要调用的工具
        5. 一个完整的Agent示例, 见agent.py

"""

#  1. 调用LangChain内置集成工具, 以调用维基百科搜索工具为例
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun


# top_k_results=1: 限制只返回相关性最高的 1 个搜索结果
# doc_content_chars_max=100: 限制每个结果的内容最多返回 100 个字符，用于控制返回内容的长度
api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100)

# 创建查询工具
# 将配置好的api_wrapper注入到WikipediaQueryRun中，创建一个可执行的工具实例
# 该工具可通过调用tool.run("查询关键词")来执行 Wikipedia 搜索
tool = WikipediaQueryRun(api_wrapper=api_wrapper)
print(tool)

"""
    执行print的输出

    {'name': tool.name,                     工具的名称(如wikipedia),用于 Agent 在决策时识别工具
    'description': tool.description,        工具的描述信息(如查询Wikipedia获取信息)，帮助 Agent 理解工具的用途
    'input': tool.args,                     工具所需的输入参数(如query)，即用户需要提供的搜索关键词
    'return_direct': tool.return_direct,    是否直接返回结果(默认为False)，控制 Agent 是否继续处理工具的输出
    }
"""


# 2. LLM function-calling
from openai import OpenAI
from datetime import datetime
import json
import os

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"), # 如果您没有配置环境变量，请在此处用您的API Key进行替换
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 填写DashScope SDK的base_url
)

# 定义工具列表，模型在选择使用哪个工具时会参考工具的name和description
# 由于LLM调用工具的时候会参考工具的name和description, 所以都需要给工具添加名字和描述, 方便llm快速精准的调用
tools = [
    # 工具1 获取当前时刻的时间
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "当你想知道现在的时间时非常有用。",
            "parameters": {}  # 因为获取当前时间无需输入参数，因此parameters为空字典
        }
    },  
    # 工具2 获取指定城市的天气
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "当你想查询指定城市的天气时非常有用。",
            "parameters": {  # 查询天气时需要提供位置，因此参数设置为location
                        "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "城市或县区，比如北京市、杭州市、余杭区等。"
                    }
                }
            },
            "required": [
                "location"
            ]
        }
    }
]

# 模拟天气查询工具。返回结果示例：“北京今天是晴天。”
def get_current_weather(location):
    return f"{location}今天是雨天。 "

# 查询当前时间的工具。返回结果示例：“当前时间：2024-04-15 17:15:18。“
def get_current_time():
    # 获取当前日期和时间
    current_datetime = datetime.now()
    # 格式化当前日期和时间
    formatted_time = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    # 返回格式化后的当前时间
    return f"当前时间：{formatted_time}。"


# 封装模型响应函数
def get_response(messages):
    completion = client.chat.completions.create(
        model="qwen-plus",
        messages=messages,
        tools=tools
        )
    return completion.model_dump()

def call_with_messages():
    print('\n')
    messages = [
            {
                "content": "杭州和北京天气怎么样？现在几点了？",
                "role": "user"
            }
    ]
    print("-"*60)
    # 模型的第一轮调用
    i = 1
    first_response = get_response(messages)
    assistant_output = first_response['choices'][0]['message']
    print(f"\n第{i}轮大模型输出信息：{first_response}\n")
    if  assistant_output['content'] is None:
        assistant_output['content'] = ""
    messages.append(assistant_output)
    # 如果不需要调用工具，则直接返回最终答案
    if assistant_output['tool_calls'] == None:  # 如果模型判断无需调用工具，则将assistant的回复直接打印出来，无需进行模型的第二轮调用
        print(f"无需调用工具，我可以直接回复：{assistant_output['content']}")
        return
    # 如果需要调用工具，则进行模型的多轮调用，直到模型判断无需调用工具
    while assistant_output['tool_calls'] != None:
        # 如果判断需要调用查询天气工具，则运行查询天气工具
        if assistant_output['tool_calls'][0]['function']['name'] == 'get_current_weather':
            tool_info = {"name": "get_current_weather", "role":"tool"}
            # 提取位置参数信息
            arguments = assistant_output['tool_calls'][0]['function']['arguments']
            if 'properties' in arguments:
                location = arguments['properties']['location']
            else:
                location = arguments['location']
            tool_info['content'] = get_current_weather(location)
        # 如果判断需要调用查询时间工具，则运行查询时间工具
        elif assistant_output['tool_calls'][0]['function']['name'] == 'get_current_time':
            tool_info = {"name": "get_current_time", "role":"tool"}
            tool_info['content'] = get_current_time()
        print(f"工具输出信息：{tool_info['content']}\n")
        print("-"*60)
        messages.append(tool_info)
        assistant_output = get_response(messages)['choices'][0]['message']
        if  assistant_output['content'] is None:
            assistant_output['content'] = ""
        messages.append(assistant_output)
        i += 1
        print(f"第{i}轮大模型输出信息：{assistant_output}\n")
    print(f"最终答案：{assistant_output['content']}")


call_with_messages()


# 3. 定义和调用自定义工具的三种方式
# 3.1 使用@tool的方式定义工具
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool, StructuredTool, tool

@tool
def get_current_weather(location: str) -> str:
    """当你想查询指定城市的天气时非常有用。"""
    return f"{location}今天是雨天。"


# 3.2 使用BaseModel类来自定义工具类型
class InputSchema(BaseModel):
    location: str = Field(description="城市或县区，比如北京市、杭州市、余杭区等。")

@tool("get_current_weather", args_schema=InputSchema)
def get_current_weather(location: str):
    """当你想查询指定城市的天气时非常有用。"""
    return f"{location}今天是雨天。"


# 3.3 使用继承了BaseModel类的子类来定义
from typing import Optional, Type
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)

class InputSchema(BaseModel):
    location: str = Field(description="城市或县区，比如北京市、杭州市、余杭区等。")

class GetCurrentWeatherTool(BaseTool):
    name = "get_current_weather"
    description = "当你想查询指定城市的天气时非常有用。"
    args_schema: Type[BaseModel] = InputSchema

    def _run(
        self, location: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        return f"{location}今天是雨天。"
    
# 3.4 使用自定义类结合StructuredTool来定义
class InputSchema(BaseModel):
    location: str = Field(description="城市或县区，比如北京市、杭州市、余杭区等。")

def get_current_weather_func(location: str):
    """当你想查询指定城市的天气时非常有用。"""
    return f"{location}今天是雨天。"

get_current_weather = StructuredTool.from_function(
    func=get_current_weather_func,
    name="get_current_weather",
    description="当你想查询指定城市的天气时非常有用。",
    args_schema=InputSchema
)


#  4. 需要调用的工具绑定到llm, 方便llm和agent进行调用
#  4.1 将tool工具以参数的形式输入llm, 来实现绑定
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

model = ChatOpenAI(model='gpt-4')
messages = [HumanMessage(content="杭州和北京天气怎么样？现在几点了？")]
assistant_output = model.invoke(messages, tools=tools)
print(assistant_output)


# 4.1 使用model.bind_tools()的方式来绑定工具
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from tools.function_calling import convert_to_openai_tool

class WeatherSchema(BaseModel):
    location: str = Field(description="城市或县区，比如北京市、杭州市、余杭区等。")

@tool("get_current_weather", args_schema=WeatherSchema)
def get_current_weather(location: str):
    """当你想查询指定城市的天气时非常有用。"""
    return f"{location}今天是雨天。"


# 查询当前时间的工具。返回结果示例：“当前时间：2024-04-15 17:15:18。“
@tool("get_current_time")
def get_current_time():
    """当你想知道现在的时间时非常有用。"""
    # 获取当前日期和时间
    current_datetime = datetime.now()
    # 格式化当前日期和时间
    formatted_time = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    # 返回格式化后的当前时间
    return f"当前时间：{formatted_time}。"

functions = [get_current_weather, get_current_time]
tools = [convert_to_openai_tool(t) for t in functions]
model_with_tools = model.bind_tools(functions)                      # 使用bind_tools()的方式绑定, functions就是被绑定的工具
messages = [HumanMessage(content="杭州和北京天气怎么样？现在几点了？")]
assistant_output = model_with_tools.invoke(messages)
