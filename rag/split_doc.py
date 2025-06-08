"""
    关于RAG部分: langchain中分割文档

    1. 文本文档分割CharacterTextSplitter, 它是一个按照字符进行分割的文本文档分割器, 它只适合纯文本文档：如.txt、.md、.py等代码文件的分割, 它分割仅仅是通过分割出来的chunk大小来进行分割的
    2. 文本文档递归分割 Recursively split by character, 它是一个递归按照字符进行分割的文本文档分割器,  它分割出来的chunk是语义完整的, 它优先在语义边界(如段落、句子、单词)处分割
    3. 分割Split code, 即专门用于分割编程代码的分割器
    4. 分割MarkdownHeaderTextSplitter类, 用于markdown文件的分割
    5. 基于语义相似性将长文档分割成连贯的文本块

"""

# 1. 文本文档分割CharacterTextSplitter
from langchain_text_splitters import CharacterTextSplitter

text_splitter = CharacterTextSplitter(
    separator="",                       # 分割符为空, 即只按照字符进行分割, 不使用任何分割符进行分割
    chunk_size=35,                      # 每35个字符为一个chunck
    chunk_overlap=4                     # 每个chunk之间重叠的部分为 4 字符
)

text = "文本文档分割CharacterTextSplitter, 它是一个按照字符进行分割的文本文档分割器, 它只适合纯文本文档：如.txt、.md、.py等代码文件的分割, 它分割仅仅是通过分割出来的chunk大小来进行分割的"
text_chunck = text_splitter.create_documents([text])  # 使用CharacterTextSplitter的分割函数create_documents()进行分割
# print(text_chunck)


text_splitter_with_space = CharacterTextSplitter(
    separator="\n",                     # 分割符"\n", 只按照"\n"进行分割
    chunk_size=35,                      # 每35个字符为一个chunck
    chunk_overlap=4                     # 每个chunk之间重叠的部分为 4 字符
)

text = "文本文档分割CharacterTextSplitter, 它是一个按照字符进行分割的文本文档分割器, 它只适合纯文本文档：如.txt、.md、.py等代码文件的分割, 它分割仅仅是通过分割出来的chunk大小来进行分割的"
text_chunck = text_splitter_with_space.create_documents([text])  # 使用CharacterTextSplitter的分割函数create_documents()进行分割
# print(text_chunck)



# 2. 文本文档递归分割 Recursively split by character
from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter_with_Recursive = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", " ", ""],  # 分割符指定这几个 "\n\n", "\n", " ", "" 进行分割, 注意separators参数和上面CharacterTextSplitter中的参数separator不一致
    chunk_size=35,                      # 每35个字符为一个chunck
    chunk_overlap=4                     # 每个chunk之间重叠的部分为 4 字符
)

text = "文本文档分割CharacterTextSplitter, 它是一个按照字符进行分割的文本文档分割器, 它只适合纯文本文档：如.txt、.md、.py等代码文件的分割, 它分割仅仅是通过分割出来的chunk大小来进行分割的"
text_chunck = text_splitter_with_Recursive.create_documents([text])  # 使用CharacterTextSplitter的分割函数create_documents()进行分割
print(text_chunck)


# 3. 分割Split code, 即专门用于分割编程代码的分割器
# 注意: 代码是一种具有规范结构, 强烈上下文语义关联的语言, 所以一般都是使用RecursiveCharacterTextSplitter进行分割, 因为它能保证完整语义
#      并且chunk_overlap一般设置为0, 因为代码中的 函数、类、模块通常是相对独立的单元，不需要依赖上下文即可理解, 没有必要进行重叠
from langchain_text_splitters import (
    Language,
    RecursiveCharacterTextSplitter
)

Python_code = """
    def hello_world():
        print("Hello world)
    
    # call func
    hello_world()
"""
python_code_spliter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, 
    chunk_size = 50,
    chunk_overlap=0
)
python_docs = python_code_spliter.create_documents([Python_code])
print(python_docs)

# 4. 分割Markdown文件, 仅仅按照Markdown文件的标题进行分割
from langchain_text_splitters import MarkdownHeaderTextSplitter  # 专门用于按 Markdown 标题结构分割文档
markdown_doc = " # Intro \n\n    ## History \n\n Markdown[9] is a lightweight markup language for creating formatted text using a plain-text editor. John Gruber created Markdown in 2004 as a markup language that is appealing to human readers in its source code form.[9] \n\n Markdown is widely used in blogging, instant messaging, online forums, collaborative software, documentation pages, and readme files. \n\n ## Rise and divergence \n\n As Markdown popularity grew rapidly, many Markdown implementations appeared, driven mostly by the need for \n\n additional features such as tables, footnotes, definition lists,[note 1] and Markdown inside HTML blocks. \n\n #### Standardization \n\n From 2012, a group of people, including Jeff Atwood and John MacFarlane, launched what Atwood characterised as a standardisation effort. \n\n ## Implementations \n\n Implementations of Markdown are available for over a dozen programming languages."
headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3")
]

markdown_doc_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
md_header_splits = markdown_doc_splitter.split_text(markdown_doc)
print(md_header_splits)

# 4.1 分割Markdown文件, 在 Markdown 标题分割的基础上，进一步使用递归字符分割器进行细粒度分割的流程
markdown_splitter = MarkdownHeaderTextSplitter(                  # 先按照Markdown文本的标题进行分割
    headers_to_split_on=headers_to_split_on, strip_headers=False
)

md_header_splits = markdown_splitter.split_text(markdown_doc)   

# Char-level splits, 在按照标题分割的基础上, 再进行细粒度的分割
from langchain_text_splitters import RecursiveCharacterTextSplitter

chunk_size = 250
chunk_overlap = 30
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size, chunk_overlap=chunk_overlap
)

# Split
splits = text_splitter.split_documents(md_header_splits)
print(splits)


# 5. 基于语义相似性将长文档分割成连贯的文本块
import os
from langchain_experimental.text_splitter import SemanticChunker            # 基于语义相似度来进行分割
from langchain_community.embeddings import OpenAIEmbeddings                 # 将文本进行词嵌入

def load_document(
    file_path: str,
    encoding: str = "utf-8",
    errors: str = "strict",
    chunk_size: int = 1024 * 1024,  # 1MB chunks for large files
) -> str:
    """
    加载文本文件内容并返回
    
    Args:
        file_path: 文件路径
        encoding: 文件编码，默认为utf-8
        errors: 编码错误处理方式，默认为"strict"
        chunk_size: 大文件读取的块大小，单位为字节
        
    Returns:
        文件内容字符串
        
    Raises:
        FileNotFoundError: 文件不存在
        PermissionError: 没有权限读取文件
        UnicodeDecodeError: 文件编码与指定编码不匹配
    """

    with open(file_path, 'r', encoding=encoding, errors=errors) as f:
        # 对于小文件，可以直接读取
        if os.path.getsize(file_path) < chunk_size:
            return f.read()
            
        # 对于大文件，分块读取并拼接
        content = []
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            content.append(chunk)
        return ''.join(content)


doc = load_document("/Users/icur/CursorProjects/LangChainBase/prompt_test.txt")
text_splitter1 = SemanticChunker(OpenAIEmbeddings())        # 进行词嵌入操作
docs = text_splitter1.create_documents([doc])
print(docs)

