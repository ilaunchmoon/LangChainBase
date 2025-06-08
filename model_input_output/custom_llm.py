from typing import Any, AsyncIterator, Dict, Iterator, List, Optional
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel, SimpleChatModel
from langchain_core.messages import AIMessageChunk, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import run_in_executor


class CustomChatModelAdvanced(BaseChatModel):
    """A custom chat model that echoes the first `n` characters of the input.

    When contributing an implementation to LangChain, carefully document
    the model including the initialization parameters, include
    an example of how to initialize the model and include any relevant
    links to the underlying models documentation or API.

    Example:

        .. code-block:: python

            model = CustomChatModel(n=2)
            result = model.invoke([HumanMessage(content="hello")])
            result = model.batch([[HumanMessage(content="hello")],
                                 [HumanMessage(content="world")]])
    """


    # model_name 用于存储模型名称，n 表示要回显的字符数
    model_name: str
    """The name of the model"""
    n: int
    """
        The number of characters from the last message of the prompt to be echoed.
        n它决定了模型将回显输入消息内容的前多少个字符    
        当调用 model.invoke([HumanMessage(content="Meow!")]) 时，由于 n=3, 模型会返回 "Meo"（即 "Meow!" 的前 3 个字符）
        如果 n 设置为其他值(如n=2)则会返回 "Me"
    """

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Override the _generate method to implement the chat model logic.

        This can be a call to an API, a call to a local model, or any other
        implementation that generates a response to the input prompt.
        

        Args:
            messages: the prompt composed of a list of messages.
            stop: a list of strings on which the model should stop generating.
                  If generation stops due to a stop token, the stop token itself
                  SHOULD BE INCLUDED as part of the output. This is not enforced
                  across models right now, but it's a good practice to follow since
                  it makes it much easier to parse the output of the model
                  downstream and understand why generation stopped.
            run_manager: A run manager with callbacks for the LLM.


        模型的核心实现，负责生成回复。它接收消息列表、停止词列表和回调管理器，返回一个 ChatResult 对象
        实现逻辑是取消息列表中的最后一条消息，提取其内容的前 n 个字符，创建一个 AI 消息，并将其包装在 ChatResult 中返回


        """
        # Replace this with actual logic to generate a response from a list
        # of messages.
        last_message = messages[-1]
        tokens = last_message.content[: self.n]
        message = AIMessage(
            content=tokens,
            additional_kwargs={},  # Used to add additional payload (e.g., function calling request)
            response_metadata={  # Use for response metadata
                "time_in_seconds": 3,
            },
        )
        ##

        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream the output of the model.

        This method should be implemented if the model can generate output
        in a streaming fashion. If the model does not support streaming,
        do not implement it. In that case streaming requests will be automatically
        handled by the _generate method.

        Args:
            messages: the prompt composed of a list of messages.
            stop: a list of strings on which the model should stop generating.
                  If generation stops due to a stop token, the stop token itself
                  SHOULD BE INCLUDED as part of the output. This is not enforced
                  across models right now, but it's a good practice to follow since
                  it makes it much easier to parse the output of the model
                  downstream and understand why generation stopped.
            run_manager: A run manager with callbacks for the LLM.

        
        实现了流式输出功能。它逐字符生成响应，并通过生成器逐个返回 ChatGenerationChunk 对象
        每次生成一个字符时，如果有回调管理器，会调用 on_llm_new_token 方法记录生成的标记。最后，它还添加了一个包含响应元数据的空块

        """
        last_message = messages[-1]
        tokens = last_message.content[: self.n]

        for token in tokens:
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=token))

            if run_manager:
                # This is optional in newer versions of LangChain
                # The on_llm_new_token will be called automatically
                run_manager.on_llm_new_token(token, chunk=chunk)

            yield chunk

        # Let's add some other information (e.g., response metadata)
        chunk = ChatGenerationChunk(
            message=AIMessageChunk(content="", response_metadata={"time_in_sec": 3})
        )
        if run_manager:
            # This is optional in newer versions of LangChain
            # The on_llm_new_token will be called automatically
            run_manager.on_llm_new_token(token, chunk=chunk)
        yield chunk

    @property
    def _llm_type(self) -> str:
        """
            Get the type of language model used by this chat model.
            返回模型的类型字符串，用于标识这是一个高级回显聊天模型
        """
        return "echoing-chat-model-advanced"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """
            Return a dictionary of identifying parameters.

            This information is used by the LangChain callback system, which
            is used for tracing purposes make it possible to monitor LLMs.

            返回模型的标识参数，这里只包含模型名称。这些参数用于 LangChain 的回调系统，帮助跟踪和监控 LLM 的使用情况
        """
        return {
            # The model name allows users to specify custom token counting
            # rules in LLM monitoring applications (e.g., in LangSmith users
            # can provide per token pricing for their model and monitor
            # costs for the given LLM.)
            "model_name": self.model_name,
        }
    


model = CustomChatModelAdvanced(n=3, model_name="my_custom_model")

model.invoke(
    [
        HumanMessage(content="hello!"),
        AIMessage(content="Hi there human!"),
        HumanMessage(content="Meow!"),
    ]
)

# 输入也支持字符串，可以等同于`[HumanMessage(content="cat vs dog")]`
for chunk in model.stream("cat vs dog"):
    print(chunk.content, end="|")


