"""
    关于RAG部分: langchain进行RAG的召回, 即整个RAG系统的流程

    1. 加载文档, 进行分块chunk
    2. 使用文本嵌入将文本块进行文本嵌入
    3. 使用向量数据库将文本嵌入进行存放
    4. 查询出与用户查询最相似的部分
    5. 将用户的查询和检索到最相似的文档一起输入到LLLM
    6. LLM输出用户查询的答案
"""

"""
    1. 加载文档, 进行分块chunk
    2. 使用文本嵌入将文本块进行文本嵌入
    3. 使用向量数据库将文本嵌入进行存放
"""
from langchain_community.document_loaders import TextLoader         # 文本文档加载器
from langchain_text_splitters import CharacterTextSplitter          # 按字符分割的文本分割器
from langchain_community.embeddings import OpenAIEmbeddings         # 使用OpenAIEmbeddings进行文本嵌入
from langchain_community.vectorstores import FAISS                  # 使用FAISS向量数据库

# 1. 加载文档, 进行分块chunk
raw_documents = TextLoader("/Users/icur/CursorProjects/LangChainBase/prompt_test.txt").load()
text_splitter = CharacterTextSplitter(
    separator='\n\n\n',
    chunk_size=50,
    chunk_overlap=4
)

# 分割文档
documents = text_splitter.split_documents(raw_documents)

# 使用嵌入模型嵌入分割文本后, 再存入到向量数据库中
db = FAISS.from_documents(documents=documents, embedding=OpenAIEmbeddings()) 

# 1.1 使用向量数据库的相似度检索, 检索出最为相似的2个
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 2})
# 检索出与用户查询最为相似的部分, 此处仅仅是依据向量数据库的相似检索, 目前还没有将检索到的结果与用户查询输入llm
retriever.invoke("哪里可以了解高考成绩")

# 1.2  MultiQueryRetriever 会利用 LLM（大语言模型）自动生成多个相关查询变体，然后合并所有查询的结果，并去重得到最终的文档集合 
# 属于是在检索前扩展查询的一种方式
# 而上面普通的检索器通常只使用单一查询来获取文档
from langchain.retrievers.multi_query import MultiQueryRetriever        # 结合llm来进行多查询检索
from langchain_openai import ChatOpenAI     
llm = ChatOpenAI(model="gpt-4")
retriever_from_llm = MultiQueryRetriever.from_llm(
    retriever=retriever,            # 上面向量数据库的普通检索器
    llm=llm,                        # 用于生成查询变体的llm
    include_original=True           # 是否包含原始查询
)

unique_docs = retriever_from_llm.invoke("哪里可以查到高考成绩")
print(unique_docs)


# 1.3 ContextualCompressionRetriever 是在检索后优化结果, 适用于需要精确回答或减少冗余信息的场景
# 而 MultiQueryRetriever通过生成多个查询变体来提高检索的召回率，获取更全面的文档集合
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# 创建基于LLM的文档压缩器
compressor = LLMChainExtractor.from_llm(llm)

# 创建上下文压缩检索器
compressor_retrievers = ContextualCompressionRetriever(
    base_compressor=compressor,     # 设置压缩文档的组件
    base_retriever=retriever        # 基础检索器
)

compressor_docs = compressor_retrievers.invoke("哪里可以查到高考成绩")
print(compressor_docs)


"""
    注意⚠️：
        目标不同
            MultiQueryRetriever: 通过生成多个查询变体来提高检索的召回率, 获取更全面的文档集合
            ContextualCompressionRetriever: 专注于提高检索结果的质量, 筛选出最相关的内容

        适用场景不同
            MultiQueryRetriever: 适用于模糊查询或需要全面信息的场景
            ContextualCompressionRetriever: 适用于需要精确回答或减少冗余信息的场景

        处理结果不同
            MultiQueryRetriever: 返回更多但可能包含冗余的文档
            ContextualCompressionRetriever: 返回更少但高度相关的文档

        一般会将如上两种方式结合起来使用:

            这两种技术并非互斥，实际上可以组合使用：
                先用 MultiQueryRetriever 扩展查询并获取更全面的文档集合
                再用 ContextualCompressionRetriever 对结果进行压缩和筛选
""" 

# 1.4 使用 LLMChainFilter 作为过滤器与 ContextualCompressionRetriever 结合使用
# LLMChainExtractor：从文档中提取相关内容片段，保留关键信息
# LLMChainFilter：直接过滤掉不相关的文档，保留完整文档
"""
    用户提出查询："哪里可以了解高考成绩"
    基础检索器获取初步文档集合（可能包含相关和不相关的文档）
    LLMChainFilter 使用 LLM 评估每个文档与查询的相关性
    只保留被判定为相关的文档
    返回筛选后的文档集合
"""
from langchain.retrievers.document_compressors import LLMChainFilter

# 创建基于LLM的文档过滤器
_filter = LLMChainFilter.from_llm(llm)

# 创建上下文压缩检索器
compression_retriever = ContextualCompressionRetriever(
    base_compressor=_filter,  # 使用过滤器作为压缩器
    base_retriever=retriever  # 基础检索器
)

# 使用压缩检索器执行查询
compressed_docs = compression_retriever.invoke(
    "哪里可以了解高考成绩"
)
compressed_docs

# 1.5 使用 EmbeddingsFilter 作为过滤器与 ContextualCompressionRetriever 结合使用
"""
    用户提出查询：
        "What did the president say about Ketanji Jackson Brown"
        (总统对 Ketanji Jackson Brown 说了什么？)

    基础检索器获取初步文档集合：
        使用底层检索器（如向量数据库）获取可能相关的文档集合，这些文档可能包含相关和不相关内容

    嵌入向量转换：
        EmbeddingsFilter 将查询文本和所有检索到的文档转换为高维向量表示。
        (例如，将文本转换为 768 维的向量)

    相似度计算：
        计算每个文档向量与查询向量的余弦相似度，得到相似度分数
        (余弦相似度范围为 -1 到 1, 值越高表示越相似)
    
    阈值过滤：
        只保留相似度分数 ≥ 0.76 的文档，过滤掉低于阈值的文档
        (例如：相似度 0.82 的文档保留, 0.65 的文档丢弃)

    返回筛选后的文档集合：
        最终返回与查询高度相关的文档集合，且保留文档的原始完整性

"""
from langchain.retrievers.document_compressors import EmbeddingsFilter

# 初始化OpenAI嵌入模型
embeddings = OpenAIEmbeddings()

# 创建基于嵌入向量的文档过滤器
embeddings_filter = EmbeddingsFilter(
    embeddings=embeddings,                  # 使用的嵌入模型
    similarity_threshold=0.76               # 相似度阈值，低于此值的文档将被过滤
)

# 创建上下文压缩检索器
compression_retriever = ContextualCompressionRetriever(
    base_compressor=embeddings_filter,      # 使用嵌入过滤器作为压缩器
    base_retriever=retriever                # 基础检索器
)

# 使用压缩检索器执行查询
compressed_docs = compression_retriever.invoke(
    "What did the president say about Ketanji Jackson Brown"
)
compressed_docs


# 1.6 使用DocumentCompressorPipeline组合多个过滤器, 来实现
# 
from langchain.retrievers.document_compressors import DocumentCompressorPipeline        # 用于组合多个不同类型的过滤器的管道
from langchain_community.document_transformers import EmbeddingsRedundantFilter         # 用于冗余语义过滤的过滤器

# 创建冗余过滤器: 基于语义冗余来过滤
redundant_filter = EmbeddingsRedundantFilter(
    embeddings=embeddings
)
# 创建相关性过滤器: 用于过滤出与查询相似
relevant_filter = EmbeddingsFilter(
    embeddings=embeddings,
    similarity_threshold=0.76           # 相似度阈值
)

# 使用DocumentCompressorPipeline组合多个过滤器
pipeline_compressor = DocumentCompressorPipeline(
    transformers=[redundant_filter, relevant_filter] # 先进行冗余过滤, 再进行相似度过滤 
)

# 创建上下文检索器: 用于结合基础检索和高级过滤, 实现检索与压缩的解耦
"""
    传统检索器的局限性：只负责 "找文档"，不负责 "优化文档"，导致结果质量参差不齐
    ContextualCompressionRetriever: 通过 **"检索 + 压缩" 的组合拳 **
        信息过载：减少冗余和无关内容
        上下文缺失：动态筛选最相关的文档
        模块化不足：解耦检索与压缩逻辑
        扩展性受限：支持灵活替换压缩策略

"""
compression_retriever = ContextualCompressionRetriever(
    base_compressor=pipeline_compressor,        # 使用组合过滤器
    base_retriever=retriever                    # 使用基础检索器
)

# 调用
compression_retriever_doc = compression_retriever.invoke(
    "What did the president say about Ketanji Jackson Brown"
)


# 1.7 集成检索器 (EnsembleRetriever) 结合两种不同的检索策略：BM25 检索 和 向量检索

from langchain_community.retrievers import BM25Retriever

# 提取文档文本内容
doc_list = [doc.page_content for doc in documents]

# 创建BM25检索器
bm25_retriever = BM25Retriever.from_texts(
    doc_list, 
    metadatas=[{"source": f"BM25"}] * len(doc_list)
)
bm25_retriever.k = 2  # 设置返回文档数量

from langchain.retrievers import EnsembleRetriever

# 初始化集成检索器
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, retriever],  # 组合两种检索器: bm25_retriever和基本向量数据库中的检索方法
    weights=[0.5, 0.5]                       # 权重均等
)

# 执行查询
docs = ensemble_retriever.invoke("哪里可以了解高考成绩")
print(docs)


# 1.7  使用多向量检索器 (MultiVectorRetriever)的高级检索策略，通过 **"子向量索引 + 父文档检索"** 的机制优化检索效果
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryByteStore

import uuid
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 为每个原始文档生成唯一ID
doc_ids = [str(uuid.uuid4()) for _ in documents]
id_key = "doc_id"  # 元数据中存储父文档ID的键

# 文本分割器：将文档拆分为小片段, 子文档
child_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,    # 每个片段约100个字符
    chunk_overlap=20   # 片段间重叠20个字符
)

# 处理每个文档
sub_docs = []
for i, doc in enumerate(documents):
    _id = doc_ids[i]                  # 获取父文档ID
    _sub_docs = child_text_splitter.split_documents([doc])  # 分割文档
    for _doc in _sub_docs:
        _doc.metadata[id_key] = _id   # 为每个子片段添加父文档ID
    sub_docs.extend(_sub_docs)

print(f"原始文档数: {len(documents)}, 子片段数: {len(sub_docs)}")

# 创建向量存储：为子片段构建索引
vectorstore = FAISS.from_documents(sub_docs, OpenAIEmbeddings())

# 创建内存存储：存储原始文档
store = InMemoryByteStore()

# 初始化多向量检索器
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,  # 子片段向量索引
    byte_store=store,         # 原始文档存储
    id_key=id_key,            # 元数据键名
)

# 将原始文档存入存储层
retriever.docstore.mset(list(zip(doc_ids, documents)))


# 1.8 使用Parent Document Retriever
# 需要完整上下文和检索结果需要易于理解和解释时场景
"""
    接收原始文档
    自动分割为子片段
    为子片段生成向量并索引
    存储原始父文档
    检索时通过子片段匹配找到父文档

"""
from langchain.retrievers import ParentDocumentRetriever  # 核心检索器类，实现子向量索引与父文档检索
from langchain_community import Chroma                    # 向量数据库，用于存储和检索子片段的向量表示
from langchain.storage import InMemoryStore               # 内存存储，用于快速访问父文档

# 子文档分割器：将文档拆分为小片段
child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,    # 每个片段约100个字符
    chunk_overlap=20   # 片段间重叠20个字符
)

# 子片段向量存储：使用Chroma数据库
vectorstore = Chroma(
    collection_name="full_documents",  # 集合名称
    embedding_function=OpenAIEmbeddings()  # 嵌入模型
)

# 父文档存储：使用内存存储
store = InMemoryStore()

# 初始化父文档检索器
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,      # 子片段向量索引
    docstore=store,               # 父文档存储
    child_splitter=child_splitter,  # 文本分割器
)

# 添加文档：自动处理分割和索引
retriever.add_documents(documents, ids=None)

# 1.9 基于摘要的多向量检索器 (Summary-based MultiVectorRetriever)，它结合了大语言模型生成的摘要和向量检索技术
"""
    流程
        用户提交查询
        查询与摘要向量进行相似度匹配
        找到最相关的摘要文档
        通过摘要的元数据(doc_id)定位原始文档
        返回完整的原始文档
    
    适用场景
        大型知识库, 文档数量庞大，需要快速筛选, 摘要帮助用户快速定位相关内容, 快速缩小搜索范围
"""
import uuid
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


chain = (
    {"doc": lambda x: x.page_content}
    | ChatPromptTemplate.from_template("概括以下内容:\n\n{doc}")
    | OpenAIEmbeddings(max_retries=0)
    | StrOutputParser()
)

summaries = chain.batch(documents, {"max_concurrency": 5})
doc_ids = [str(uuid.uuid4()) for _ in documents]
id_key = "doc_id"

summary_docs = [
    Document(page_content=s, metadata={id_key: doc_ids[i]})
    for i, s in enumerate(summaries)
]

# The vectorstore to use to index the child chunks
vectorstore = FAISS.from_documents(summary_docs, OpenAIEmbeddings())
# The storage layer for the parent documents
store = InMemoryByteStore()
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    byte_store=store,
    id_key=id_key,
)
retriever.docstore.mset(list(zip(doc_ids, documents)))


# 1.10 基于假设性问题(HypotheticalQueries)的多向量检索器
"""
    基于假设性问题(HypotheticalQueries) 通过 "问题索引 + 全文返回" 的方式，实现了检索精度和用户体验的双重提升，特别适合需要智能问答和知识探索的应用场景
    Hypothetical Queries(假设性问题)是一种检索增强技术，核心思想是：


    为文档生成代表性问题：
        使用 LLM 为每个文档生成多个相关问题
        这些问题代表了文档的核心内容和潜在查询

    基于问题而非文档内容索引：
        不为原始文档构建向量索引
        而是为生成的问题构建向量索引

    检索流程：
        用户查询 → 匹配相似问题 → 通过问题找到原始文档

    最终返回完整的原始文档


"""
from langchain_core.messages import AIMessage
from langchain_core.exceptions import OutputParserException

def custom_parse(ai_message: AIMessage) -> str:
    """解析AI生成的消息, 提取多个问题"""
    if '\n\n' in ai_message.content:
        return ai_message.content.split('\n\n')  # 按双换行符分割
    elif '\n' in ai_message.content:
        return ai_message.content.split('\n')    # 按单换行符分割
    else:
        raise OutputParserException("Badly formed question!")  # 格式错误时抛出异常

# 创建问题生成链
chain = (
    {"doc": lambda x: x.page_content}  # 提取文档内容
    | ChatPromptTemplate.from_template("为下面内容生成3个合适的提问问题: \n\n{doc}\n\n#限制\n生成的3个问题使用两个换行符, 即```\n\n```符号进行隔开")  # 提示模板
    | ChatOpenAI(max_retries=0)  # 使用通义千问生成问题
    | custom_parse  # 自定义解析器
)

# 批量为文档生成假设性问题（并发数为5）
hypothetical_questions = chain.batch(documents, {"max_concurrency": 5})
hypothetical_questions[0]  # 查看第一个文档的问题

# 为每个文档生成唯一ID
doc_ids = [str(uuid.uuid4()) for _ in documents]
id_key = "doc_id"  # 元数据中存储父文档ID的键

# 创建问题文档对象
question_docs = []
for i, question_list in enumerate(hypothetical_questions):
    question_docs.extend(
        [Document(page_content=s, metadata={id_key: doc_ids[i]}) for s in question_list]
    )

# 为问题构建向量索引
vectorstore = FAISS.from_documents(question_docs, OpenAIEmbeddings())

# 创建内存存储：用于存储原始文档
store = InMemoryByteStore()

# 初始化多向量检索器
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,  # 问题向量索引
    byte_store=store,         # 原始文档存储
    id_key=id_key,            # 元数据键名
)

# 将原始文档存入存储层
retriever.docstore.mset(list(zip(doc_ids, documents)))

# 1.11 基于SelfQueryRetriever，它是一种能够自动解析用户查询并结合向量检索和元数据过滤的高级检索器
"""
    SelfQueryRetriever 通过 "智能解析 + 组合检索" 的方式，实现了语义理解和结构化过滤的结合，特别适合既有文本内容又有丰富元数据的场景
    相比之下, Hypothetical Queries 更侧重于通过生成假设性问题来扩展文档的语义表示，适用于非结构化数据和需要扩展查询覆盖的场景


    工作流程:
        用户提交查询
        LLM 解析查询，提取语义内容和元数据条件
        组合向量检索和元数据过滤
        返回符合条件的文档

"""
from langchain_community import Chroma 
from langchain_core.documents import Document

# 创建示例文档
docs = [
    Document(
        page_content="A bunch of scientists bring back dinosaurs and mayhem breaks loose",
        metadata={"year": 1993, "rating": 7.7, "genre": "science fiction"},
    ),
    Document(
        page_content="Leo DiCaprio gets lost in a dream within a dream within a dream within a ...",
        metadata={"year": 2010, "director": "Christopher Nolan", "rating": 8.2},
    ),
    Document(
        page_content="A psychologist / detective gets lost in a series of dreams within dreams within dreams and Inception reused the idea",
        metadata={"year": 2006, "director": "Satoshi Kon", "rating": 8.6},
    ),
    Document(
        page_content="A bunch of normal-sized women are supremely wholesome and some men pine after them",
        metadata={"year": 2019, "director": "Greta Gerwig", "rating": 8.3},
    ),
    Document(
        page_content="Toys come alive and have a blast doing so",
        metadata={"year": 1995, "genre": "animated"},
    ),
    Document(
        page_content="Three men walk into the Zone, three men walk out of the Zone",
        metadata={
            "year": 1979,
            "director": "Andrei Tarkovsky",
            "genre": "thriller",
            "rating": 9.9,
        },
    ),
]

# 创建向量存储
vectorstore = Chroma.from_documents(docs, OpenAIEmbeddings())

from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever

# 定义元数据字段信息
metadata_field_info = [
    AttributeInfo(
        name="genre",
        description="The genre of the movie. One of ['science fiction', 'comedy', 'drama', 'thriller', 'romance', 'action', 'animated']",
        type="string",
    ),
    AttributeInfo(
        name="year",
        description="The year the movie was released",
        type="integer",
    ),
    AttributeInfo(
        name="director",
        description="The name of the movie director",
        type="string",
    ),
    AttributeInfo(
        name="rating", description="A 1-10 rating for the movie", type="float"
    ),
]

# 文档内容描述
document_content_description = "Brief summary of a movie"

# 初始化语言模型
llm = ChatOpenAI(model="gpt-4")

# 创建SelfQueryRetriever
retriever = SelfQueryRetriever.from_llm(
    llm,
    vectorstore,
    document_content_description,
    metadata_field_info,
)
