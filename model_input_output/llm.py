"""
    关于Model I/O部分: langchain调用llm

    1. 使用llm阻塞输出模式
    2. 使用llm流式输出模式
    3. 自定义调用llm
    4. 缓存
    5. 记录消耗的token
"""

# 1. 使用llm阻塞输出模式
from langchain_openai import ChatOpenAI     # ChatOpenAI只能够调用OpenAI系列的llm

llm = ChatOpenAI(model="gpt-4")
response = llm.invoke("你好, 你能为我做什么, 10字以内")
print(response.content)

#  2. 使用llm流式输出模式
llm = ChatOpenAI(model="gpt-4")
for chunk in llm.stream("什么是曲率引擎, 40个字以内回答"):
    # print(chunk, end="\n", flush=True)
    print(chunk.content, flush=True)      # 输出方式: token by token


# 3. 自定义llm
"""
    自定义llm是如下几个方面的自定义:

        模型提供商自定义
            OpenAI、Anthropic、Hugging Face 等第三方 API
            本地部署的开源模型（如 LLaMA、Alpaca 等）
            专有或企业内部的 LLM 服务

        输入处理自定义
            文本格式化
            添加系统指令
            实现特定的提示模板
            支持特殊的输入格式（如 JSON、Markdown 等）

        输出处理自定义
            提取特定格式的内容
            过滤或修改输出
            转换为特定的数据结构
            实现自定义的错误处理逻辑

        流式输出自定义
            按字符、词、句子或自定义规则拆分输出
            添加额外的元数据或上下文信息
            实现自定义的进度跟踪
        
        参数映射自定义
            将 LangChain 的标准参数映射到特定 API 的参数
            emperature、top_p 等通用参数的转换
            支持特定 API 的专有参数
            实现参数验证和默认值设置

        错误处理自定义
            与缓存系统集成
            添加自定义的回调函数
            集成监控和日志系统
            实现自定义的成本计算逻辑

        集成功能自定义
            与缓存系统集成
            添加自定义的回调函数
            集成监控和日志系统
            实现自定义的成本计算逻辑
"""
from typing import Any, Dict, Iterator, List, Mapping, Optional
from langchain_core.callbacks.manager import CallbackManagerForLLMRun   # 用于管理回调函数，监控模型运行过程
from langchain_core.language_models.llms import LLM                     # LLM是 LangChain 中所有 LLM 模型的基类，自定义模型需要继承它
from langchain_core.outputs import GenerationChunk                      # 用于流式输出时表示一个生成块


class CustomLLM(LLM):
    """A custom chat model that echoes the first `n` characters of the input.

    When contributing an implementation to LangChain, carefully document
    the model including the initialization parameters, include
    an example of how to initialize the model and include any relevant
    links to the underlying models documentation or API.

    响应输入的第一个“n”字符的自定义聊天模型, 自定义LLM必须要n
    在向LangChain贡献实现时,请仔细编写文档
    该模型包括初始化参数,包括一个如何初始化模型并包含任何相关的示例链接到底层模型文档或API

    Example:

        .. code-block:: python

            model = CustomChatModel(n=2)
            result = model.invoke([HumanMessage(content="hello")])
            result = model.batch([[HumanMessage(content="hello")],
                                 [HumanMessage(content="world")]])
    """

    n: int
    """The number of characters from the last message of the prompt to be echoed."""

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Run the LLM on the given input.

        Override this method to implement the LLM logic.

        Args:
            prompt: The prompt to generate from.
            stop: Stop words to use when generating. Model output is cut off at the
                first occurrence of any of the stop substrings.
                If stop tokens are not supported consider raising NotImplementedError.
            run_manager: Callback manager for the run.
            **kwargs: Arbitrary additional keyword arguments. These are usually passed
                to the model provider API call.

        Returns:
            The model output as a string. Actual completions SHOULD NOT include the prompt.

            在给定的输入上运行LLM, 自定义的LLM必须重写此方法以实现LLM逻辑, _call方法是 LLM 的核心执行逻辑，必须被重写
            此实现简单返回输入的前n个字符, 不支持stop参数时, 会抛出异常

            参数:
                prompt: 要从中生成的提示符
                停止：生成时使用的停止词, 模型输出在任何停止子字符串的第一次出现如果不支持停止令牌,请考虑引发NotImplementedError
                run_manager: 运行的回调管理器
                **kwargs: 任意附加关键字参数。这些通常是通过的,到模型提供程序API调用

            返回:
                模型输出为字符串。实际完成不应该包含提示符            
        """
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        return prompt[: self.n]

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        """Stream the LLM on the given prompt.

        This method should be overridden by subclasses that support streaming.

        If not implemented, the default behavior of calls to stream will be to
        fallback to the non-streaming version of the model and return
        the output as a single chunk.

        Args:
            prompt: The prompt to generate from.
            stop: Stop words to use when generating. Model output is cut off at the
                first occurrence of any of these substrings.
            run_manager: Callback manager for the run.
            **kwargs: Arbitrary additional keyword arguments. These are usually passed
                to the model provider API call.

        Returns:
            An iterator of GenerationChunks.

        实现流式输出功能，逐字符生成结果, 对每个字符创建GenerationChunk对象, 使用run_manager通知新 token 生成, 通过yield关键字实现迭代器, 支持实时输出
        在给定的提示符上流式传输LLM, 这个方法应该被支持流的子类覆盖, 在自定义LLM的类中必须重写该方法
        如果没有实现，调用流的默认行为将是回退到模型的非流版本并返回作为单个块的输出

        参数:
            prompt: 要从中生成的提示符
            停止: 生成时使用的停止词。模型输出在, 这些子字符串中的任何一个首次出现
            run_manager: 运行的回调管理器。
            **kwargs: 任意附加关键字参数。这些通常是通过的到模型提供程序API调用

        返回:
            GenerationChunks的迭代器


        """
        for char in prompt[: self.n]:
            chunk = GenerationChunk(text=char)
            if run_manager:
                run_manager.on_llm_new_token(chunk.text, chunk=chunk)

            yield chunk

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """
            Return a dictionary of identifying parameters.
            返回模型标识参数，用于监控和成本计算, 返回一个用于标识参数的字典,
        """
        return {
            # The model name allows users to specify custom token counting
            # rules in LLM monitoring applications (e.g., in LangSmith users
            # can provide per token pricing for their model and monitor
            # costs for the given LLM.)
            "model_name": "CustomChatModel",
        }

    @property
    def _llm_type(self) -> str:
        """
            Get the type of language model used by this chat model. Used for logging purposes only.
            获取此聊天模型使用的语言模型类型, 仅用于记录目的
            返回模型类型，用于日志记录
        """
        return "custom"
    



# 如下是基于gpt-4实现的自定义LLM
import os
import requests
import asyncio
from typing import Any, Dict, Iterator, List, Mapping, Optional, Union, Tuple
from urllib.parse import urljoin
from functools import partial

from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk
from langchain_core.pydantic_v1 import BaseModel, Extra, Field, root_validator
from langchain_core.callbacks.manager import CallbackManagerForLLMRun


class GPT4LLM(LLM, BaseModel):
    """OpenAI GPT-4 LLM 自定义实现"""
    
    model_name: str = "gpt-4"
    """GPT模型名称"""
    
    api_base: str = "https://api.openai.com/v1"
    """OpenAI API基础URL"""
    
    api_key: Optional[str] = None
    """OpenAI API密钥"""
    
    temperature: float = 0.7
    """生成温度，控制输出的随机性"""
    
    max_tokens: Optional[int] = None
    """最大生成token数"""
    
    top_p: float = 0.95
    """核采样概率"""
    
    n: int = 1
    """生成候选数量"""
    
    timeout: float = 60.0
    """请求超时时间(秒)"""
    
    batch_size: int = 5
    """批量处理的批次大小"""
    
    class Config:
        """模型配置"""
        extra = Extra.forbid
        
    def __init__(self, **kwargs: Any) -> None:
        """初始化模型"""
        super().__init__(**kwargs)
        
    @root_validator(pre=True)
    def validate_environment(cls, values: Dict) -> Dict:
        """验证环境变量和配置"""
        api_key = values.get("api_key") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API密钥未提供, 请设置api_key参数或OPENAI_API_KEY环境变量")
        values["api_key"] = api_key
        return values
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """执行LLM调用, 返回生成的文本"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "top_p": self.top_p,
            "n": self.n,
        }
        
        if self.max_tokens:
            payload["max_tokens"] = self.max_tokens
            
        if stop:
            payload["stop"] = stop
            
        # 合并额外参数
        payload.update(kwargs)
        
        response = requests.post(
            urljoin(self.api_base, "/chat/completions"),
            headers=headers,
            json=payload,
            timeout=self.timeout
        )
        
        if response.status_code != 200:
            raise ValueError(f"OpenAI API调用失败: {response.status_code}, {response.text}")
            
        return response.json()["choices"][0]["message"]["content"]
    
    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        """流式输出LLM生成结果"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "top_p": self.top_p,
            "n": self.n,
            "stream": True,
        }
        
        if self.max_tokens:
            payload["max_tokens"] = self.max_tokens
            
        if stop:
            payload["stop"] = stop
            
        # 合并额外参数
        payload.update(kwargs)
        
        response = requests.post(
            urljoin(self.api_base, "/chat/completions"),
            headers=headers,
            json=payload,
            stream=True,
            timeout=self.timeout
        )
        
        if response.status_code != 200:
            raise ValueError(f"OpenAI API调用失败: {response.status_code}, {response.text}")
            
        # 解析流式响应
        for line in response.iter_lines():
            if line:
                line = line.decode("utf-8")
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        chunk_data = eval(data)
                        content = chunk_data["choices"][0]["delta"].get("content", "")
                        if content:
                            chunk = GenerationChunk(text=content)
                            if run_manager:
                                run_manager.on_llm_new_token(chunk.text, chunk=chunk)
                            yield chunk
                    except Exception as e:
                        print(f"解析流式响应错误: {e}")
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """返回模型标识参数"""
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "n": self.n,
            "timeout": self.timeout,
            "batch_size": self.batch_size,
        }
    
    @property
    def _llm_type(self) -> str:
        """返回LLM类型"""
        return "openai-gpt4"
    
    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """异步执行LLM调用"""
        import aiohttp
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "top_p": self.top_p,
            "n": self.n,
        }
        
        if self.max_tokens:
            payload["max_tokens"] = self.max_tokens
            
        if stop:
            payload["stop"] = stop
            
        # 合并额外参数
        payload.update(kwargs)
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                urljoin(self.api_base, "/chat/completions"),
                headers=headers,
                json=payload,
                timeout=self.timeout
            ) as response:
                if response.status != 200:
                    raise ValueError(f"OpenAI API调用失败: {response.status}, {await response.text()}")
                    
                result = await response.json()
                return result["choices"][0]["message"]["content"]
    
    async def _astream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        """异步流式输出LLM生成结果"""
        import aiohttp
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "top_p": self.top_p,
            "n": self.n,
            "stream": True,
        }
        
        if self.max_tokens:
            payload["max_tokens"] = self.max_tokens
            
        if stop:
            payload["stop"] = stop
            
        # 合并额外参数
        payload.update(kwargs)
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                urljoin(self.api_base, "/chat/completions"),
                headers=headers,
                json=payload,
                timeout=self.timeout
            ) as response:
                if response.status != 200:
                    raise ValueError(f"OpenAI API调用失败: {response.status}, {await response.text()}")
                    
                # 异步解析流式响应
                async for line in response.content:
                    if line:
                        line = line.decode("utf-8").strip()
                        if line.startswith("data: "):
                            data = line[6:]
                            if data == "[DONE]":
                                break
                            try:
                                chunk_data = eval(data)
                                content = chunk_data["choices"][0]["delta"].get("content", "")
                                if content:
                                    chunk = GenerationChunk(text=content)
                                    if run_manager:
                                        await run_manager.on_llm_new_token(chunk.text, chunk=chunk)
                                    yield chunk
                            except Exception as e:
                                print(f"解析异步流式响应错误: {e}")
    
    def batch(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> List[str]:
        """批量处理多个提示"""
        results = []
        
        # 分批处理
        for i in range(0, len(prompts), self.batch_size):
            batch_prompts = prompts[i:i + self.batch_size]
            
            # 对每个提示调用模型
            batch_results = [
                self._call(prompt, stop=stop, run_manager=run_manager, **kwargs)
                for prompt in batch_prompts
            ]
            
            results.extend(batch_results)
            
        return results
    
    async def abatch(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> List[str]:
        """异步批量处理多个提示"""
        results = []
        
        # 分批处理
        for i in range(0, len(prompts), self.batch_size):
            batch_prompts = prompts[i:i + self.batch_size]
            
            # 异步对每个提示调用模型
            tasks = [
                self._acall(prompt, stop=stop, run_manager=run_manager, **kwargs)
                for prompt in batch_prompts
            ]
            
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)
            
        return results



# 4. 缓存  5. 记录消耗的token
from langchain.globals import set_llm_cache
# from langchain.cache import InMemoryCache 旧版本使用
from langchain_community.cache import InMemoryCache
from langchain.callbacks.base import BaseCallbackHandler
# from langchain.callbacks import get_openai_callback 旧版本导入
from langchain_community.callbacks.manager import get_openai_callback

# 使用OpenAI回调处理器跟踪token使用情况
with get_openai_callback() as cb:
    result = llm.invoke("解释量子计算的基本原理, 20字以内")
    print(cb)  # 打印token使用情况




