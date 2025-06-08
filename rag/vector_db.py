"""
    关于RAG部分: langchain中向量数据库存储

    1. 加载源数据文档
    2. 使用向量数据库存放 
    3. 相似度检索:  相似度指标为: Maximum marginal relevance search (MMR)
    
"""

# 1. 加载源数据文档
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
raw_documents = TextLoader('/Users/icur/CursorProjects/LangChainBase/prompt_test.txt').load()
text_splitter = CharacterTextSplitter(separator='\n\n\n', chunk_size=50, chunk_overlap=4)
documents = text_splitter.split_documents(raw_documents)
print(documents[-1])

# 2. 使用向量数据库存放分割的文档, 存放的时候一定要使用文本嵌入模型将分割的文档进行嵌入之后, 向量数据库才可以存放文本嵌入的向量
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

db = FAISS.from_documents(documents, OpenAIEmbeddings())

# 3. 相似度检索
query = "哪里可以了解高考成绩"
docs = db.similarity_search(query)
print(docs[0].page_content)
embedding_vector = OpenAIEmbeddings().embed_query(query)
docs = db.similarity_search_by_vector(embedding_vector)
print(docs[0].page_content)

# 3. 相似度检索, 使用 Maximum marginal relevance search (MMR)指标
query = "哪里可以了解高考成绩"
docs = db.max_marginal_relevance_search(query)
print(docs)
