"""
    关于Model I/O部分: langchain中提示词模版

    1. 字符串形式的提示词模版的形式
    2. 带有角色的对话式的提示词模版形式, 硬编码形式
    3. 带有角色的对话式的提示词模版形式, 使用角色类的非硬编码形式
    4. 读取提示词文件的形式创建提示词模版
    5. 使用消息模版和消息占位符预先设定提示词的槽位
    6. few-shot提示词模版
"""

# 1. 字符串形式的提示词模版的形式
from langchain.prompts import PromptTemplate        
prompt_template = PromptTemplate.from_template(
    "Tell me a {object} joke about {content}"
)
print(prompt_template.format(object="小明", content="高中学习"))


# 2. 带有角色的对话式的提示词模版形式, 硬编码形式
# from langchain.prompts import ChatPromptTemplate          # 旧版本导入方式, 不建议使用
from langchain_core.prompts import ChatPromptTemplate       # 新版本导入方式

chat_prompt1 = ChatPromptTemplate.from_messages(            # 直接从消息列表创建模板, 则使用from_messages
    [
        ("system", "You are a powerful AI bot. Your name is {name}"),       # 系统全局信息
        ("human", "Hello, how are you doing"),                              # 用户输入信息
        ("ai", "fine, thanks"),                                             # llm回复信息
        ("human", "{user_input}")                                           # 用户输入信息
    ]
)

print(chat_prompt1.format_messages(name="bot", user_input="你的名字是什么？你能为我做什么, 不能做什么"))

# 可转为字典的方式来获取仅含提示词的信息: 
formatted_messages = chat_prompt1.format_messages(name="bot", user_input="你的名字是什么？你能为我做什么, 不能做什么")
for msg in formatted_messages:
    print(f"{msg.text}: {msg.content}")


# 转换为字典（调试时常用）
from langchain_core.messages import messages_to_dict
messages_dict = messages_to_dict(formatted_messages)
print(messages_dict[0]["data"]["content"])              # 注意必须先到 “data”然后再到content

# 或者通过使用索引方式来获取
print(formatted_messages[0].content)        # 第0个位置是 ("system", "You are a powerful AI bot. Your name is {name}")
print(formatted_messages[1].content)        # 第1个位置是 ("human", "Hello, how are you doing")
print(formatted_messages[2].content)        # 第2个位置是 ("ai", "fine, thanks")
print(formatted_messages[3].content)        # 第3个位置是("human", "{user_input}")


chat_prompt2= ChatPromptTemplate.from_template(             # 使用字符串的形式创建加占位符的形式创建: 使用from_template()
    "You are a weather expert. What's the weather like in {city} today?"
)

# format_messages和 format_prompt的去呗
"""
    format_messages()

        输出格式：返回一个 Message 对象列表（如 SystemMessage、HumanMessage 等）
        适用场景：直接用于聊天模型如 hatOpenAI、ChatAnthropic的输入
        特点：保持消息的结构化, 明确区分不同角色 system/human/ai

    format_prompt()

        输出格式：返回一个 PromptValue 对象（包含 to_messages() 和 to_string() 方法）
        适用场景：更通用，可根据需要转换为消息列表或字符串
        特点：灵活性更高，支持多类型输出

"""
print(chat_prompt2.format_messages(city="shanghai"))
print(chat_prompt2.format_prompt(city="changzhou"))



# 3. 带有角色的对话式的提示词模版形式, 使用角色类的非硬编码形式
from langchain_core.prompts import HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage
chat_mess = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content=("你是功能强大的AI助手, 可以用于写作和润色")),    # 系统角色消息
        HumanMessagePromptTemplate.from_template("{text}")                # 用户角色消息
    ]
)

# 使用format_messages()方法格式化模板
chat_message1 = chat_mess.format_messages(text="你能做什么")
print(chat_message1)




# 4. 读取提示词文件的形式创建提示词模版
from langchain.prompts import PromptTemplate
chat_prompt_file = PromptTemplate.from_file("/Users/icur/CursorProjects/LangChainBase/prompt_test.txt")
print(chat_prompt_file.format(score="650"))


# 4.1 使用其他加载方式
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

# 从文件加载用户提示模板
loader = TextLoader("/Users/icur/CursorProjects/LangChainBase/prompt_test.txt")
load_chat_prompt_file1 = loader.load()
chat_prompt_file1_content = load_chat_prompt_file1[0].page_content

# 创建聊天提示模板
chat_prompt_file1_message = ChatPromptTemplate.from_messages(
    [
        HumanMessagePromptTemplate.from_template(chat_prompt_file1_content)
    ]
)

# 使用 format_messages() 填充模板变量
formatted_messages = chat_prompt_file1_message.format_messages(score="700")
print(formatted_messages)
print(formatted_messages[0].content)




# 5. 使用消息模版和消息占位符预先设定提示词的槽位
from langchain.prompts import ChatMessagePromptTemplate

mess_prompts = "Say topic about {topic}"
chat_mess_prompts = ChatMessagePromptTemplate.from_template(
    role="Bob",
    template=mess_prompts
)
print(chat_mess_prompts.format(topic="小明"))


from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder             # 消息占位符
)

from langchain_core.messages import AIMessage, HumanMessage     # 使用AI和用户message

human_prompt_message = "Summarize our conversation so far in {word_count} words."
human_message_template = HumanMessagePromptTemplate.from_template(human_prompt_message)
chat_prompt = ChatPromptTemplate.from_messages(
    [
        MessagesPlaceholder(variable_name="conversation"),
        human_message_template
    ]
)

human_message = HumanMessage(content="What is the best way to learn programming?")
ai_message = AIMessage(
    content="""\
1. Choose a programming language: Decide on a programming language that you want to learn.

2. Start with the basics: Familiarize yourself with the basic programming concepts such as variables, data types and control structures.

3. Practice, practice, practice: The best way to learn programming is through hands-on experience\
"""
)

chat_prompt.format_prompt(
    conversation=[human_message, ai_message], word_count="11"
).to_messages()


# 6. few-shot提示词模版
from langchain_core.prompts import (
    ChatPromptTemplate, 
    FewShotChatMessagePromptTemplate
)
examples = [
    {"input": "2 + 3", "output": "5"},
    {"input": "2 + 4", "output": "6"},
    {"input": "2 + 5", "output": "7"},
    {"input": "2 + 6", "output": "8"}
]

example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}")
    ]
)

few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples
)
print(few_shot_prompt.format())
