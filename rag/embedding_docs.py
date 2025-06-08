"""
    关于RAG部分: langchain进行文档嵌入操作embedding

    1. 文档嵌入操作embedding
    2. 使用 LangChain 的缓存机制优化文本嵌入过程，避免重复调用 API
    3. 使用内存缓存文本嵌入
    4. 使用redis数据库缓存文本嵌入

"""

# 1. 文档嵌入操作embedding
from langchain_community.embeddings import OpenAIEmbeddings

# 初始化嵌入模型
embeddings_model = OpenAIEmbeddings(
    model="text-embedding-ada-002",  # 可选参数，默认即为 ada-002
    chunk_size=1000  # 批量处理大小，提高效率
)

# 嵌入文本
texts = [
    "你好吗",
    "你的名字是什么",
    "我的肚子好痛啊",
    "肠胃不舒服",
    "我在吃东西"
]

embeddings = embeddings_model.embed_documents(texts)

# 检查维度
print(f"向量数量: {len(embeddings)}")
print(f"单个向量维度: {len(embeddings[0])}")  # 1536 维


# 2. 缓存机制优化文本嵌入过程
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
import os

# (1). 创建存储后端（直接使用当前目录作为根目录）
env = os.environ.get("ENV", "dev")
store = LocalFileStore(f"./{env}/")  # 修改为当前目录下的环境子目录

# (2). 初始化基础嵌入模型
embeddings_model = OpenAIEmbeddings(model="text-embedding-ada-002")

# (3). 创建带缓存的嵌入器（自定义命名空间）
cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    embeddings_model,
    store=store,
    namespace="my_knowledge_base_v1.2"
)

# (4). 使用缓存嵌入器
texts = ["你好吗", "今天天气如何"]
embeddings = cached_embedder.embed_documents(texts)


#  3. 使用内存缓存文本嵌入
from langchain.storage import InMemoryByteStore
store = InMemoryByteStore()
cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    embeddings_model, store, namespace=embeddings_model.model_name
)
print(cached_embedder.embed_documents)


#  4. 使用redis数据库缓存文本嵌入
from langchain.storage import RedisStore
store = RedisStore()
cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    embeddings_model, store, namespace=embeddings_model.model_name
)
print(cached_embedder.embed_documents)



