"""
    关于RAG部分: langchain中加载文档

    1. 文本文档加载器
    2. csv文档加载器
    3. PDF文档加载器
    4. 自定义加载各种类型文档的类
"""

# 1. 文本文档加载器
from langchain_community.document_loaders import TextLoader         # 文本文档加载器

loader = TextLoader("/Users/icur/CursorProjects/LangChainBase/prompt_test.txt")
text_doc = loader.load()
#print(text_doc)



# 2. csv文档加载器
from langchain_community.document_loaders import CSVLoader          # csv文档加载器
csv_doc_loader = CSVLoader("/Users/icur/CursorProjects/LangChainBase/test_csv.csv")
csv_doc = csv_doc_loader.load()
#print(csv_doc)

# 使用参数配合加载csv格式的文件
"""
    csv_args
        传递给 Python 内置csv.reader的参数
        控制 CSV 文件的解析方式
        常见参数包括：
        delimiter: 字段分隔符（默认为逗号,)
        quotechar: 引号字符（默认为双引号")
        fieldnames: 自定义列名（当 CSV 没有表头时使用）
        skipinitialspace: 是否跳过分隔符后的空白(默认为False)
        escapechar: 转义字符（如\)
        lineterminator: 行终止符(如\n)

    source_column
        指定哪一列作为文档的 source 标识
        示例中使用id列作为 source

"""
loader = CSVLoader(
    file_path='/Users/icur/CursorProjects/LangChainBase/test_csv.csv', 
    csv_args={
    'delimiter': ',',
    'quotechar': '"',
    'fieldnames': ['id', 'name', 'degree']
    }, 
    source_column='id'
)

csv_doc = loader.load()
print(csv_doc)

# 3. PDF文档加载器: 需要安装 pip install pypdf
from langchain_community.document_loaders import PyPDFLoader
pdf_loader = PyPDFLoader("/Users/icur/Downloads/2nd_place_Guanshuo.pdf")
pages = pdf_loader.load()
print(pages)


# 自定义加载各种类型文档的类
from typing import AsyncIterator, Iterator                  # 异步或同步迭代器
from langchain_core.document_loaders import BaseLoader      # LangChain的文档加载基类
from langchain_core.documents import Document               # LangChain的文档类, 包含内容和元数据

# 自定义按行读取文本文档中的每一行
# 
class TextDocCustomLoader(BaseLoader):
    def __init__(self, file_path:str) -> None:
        super().__init__()
        self.file_path = file_path                          # 文档路径

    def lazy_load(self) -> Iterator[Document]:              # 同步调用读取文件
        with open(self.file_path, encoding="utf-8") as f:
            line_num = 0
            for line in f:
                if not line.strip():
                    continue
                yield Document(
                    page_content=line,
                    metadata={
                                "line_num": line_num,
                                "source": self.file_path
                            }
                )
                line_num += 1
    
    async def alazy_load(self) -> AsyncIterator[Document]:  # 异步调用读取文件: 需要按照 pip install aiofiles
        async with open(self.file_path, encoding="utf-8") as f:
            line_num = 0
            async for line in f:
                if not line.strip():
                    continue
                yield Document(
                    page_content=line,
                    metadata={
                                "line_num": line_num,
                                "source": self.file_path
                    }
                )
                line_num += 1


# 使用上面的类进行同步调用读取文件
text_custom_load_file = TextDocCustomLoader("/Users/icur/CursorProjects/LangChainBase/prompt_test.txt")
for doc in text_custom_load_file.lazy_load():
    print(type(doc), '|', doc)


# 使用上面的类进行异步调用读取文件
async def read_text_doc_with_async():
    text_custom_async_load_file = TextDocCustomLoader("/Users/icur/CursorProjects/LangChainBase/prompt_test.txt")
    async for doc in text_custom_async_load_file.alazy_load():
        print(type(doc), '|', doc)


read_text_doc_with_async()




#  





