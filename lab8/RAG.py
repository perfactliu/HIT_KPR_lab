import torch  # filepath: example.py
# 核心 LangChain 组件
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA, LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
# LangChain 社区组件
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings

# Hugging Face 组件
from transformers import pipeline

# 标准库
import os
import urllib.request
import zipfile

# 知识库构建
# RAG系统的基础是高质量的知识库。本实现中，我们使用一个包含亚洲各目的地信息的数据集作为示例：

extract_folder = "asia_documents"

# 文档处理与分块策略
# RAG系统的关键步骤是文档的处理与分块。这一阶段包括：
# 文档加载
# 将文档分割为适当大小的块
# 为每个文本块生成向量表示

 # filepath: example.py
# 从文件夹加载所有文本文件
documents = []
for filename in os.listdir(extract_folder):
    if filename.endswith(".txt"):
        file_path = os.path.join(extract_folder, filename)
        loader = TextLoader(file_path)
        documents.extend(loader.load())

print(f"Loaded {len(documents)} documents")  # 加载了 {len(documents)} 个文档

# 将文档拆分成更小的块，以便更好地检索
from langchain.text_splitter import CharacterTextSplitter

text_splitter = CharacterTextSplitter(
    chunk_size=500,  # 每个块的字符数
    chunk_overlap=100  # 块之间的重叠以保持上下文
)
docs = text_splitter.split_documents(documents)

print(f"Created {len(docs)} document chunks")  # 创建了 {len(docs)} 个文档块

# 向量数据库构建
# RAG系统的核心是向量数据库，它实现了基于语义的高效搜索。本实现采用FAISS作为向量存储引擎，结合句子转换模型构建嵌入表示：

 # filepath: example.py
# 初始化嵌入模型
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"# 速度和质量的良好平衡
)

# 从我们的文档块创建一个向量存储
vectorstore = FAISS.from_documents(docs, embedding_model)

# 创建一个检索器接口
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}  # 检索前 3 个最相关的块
)

print("Vector database created successfully!")  # 向量数据库创建成功！


# 语言模型配置
# 本实现使用GPT2作为基础模型，但实际应用中可替换为更高性能的模型如Llama-2或Mistral：

 # filepath: example.py
# 使用 Hugging Face 模型创建一个文本生成管道
if torch.cuda.is_available():
    device = 0
    print(f"Using GPU (device {device}) for inference.")
else:
    device = -1
    print("No GPU found, falling back to CPU.")

llm_pipeline = pipeline(
    "text-generation",
    model="GPT2",  # 你可以用其他模型替换它，例如 "mistralai/Mistral-7B-v0.1"
    device = device,
    max_new_tokens=200  # 控制响应长度
)

# 将管道包装在 LangChain 的接口中
llm = HuggingFacePipeline(pipeline=llm_pipeline)

print("Language model loaded successfully!")  # 语言模型加载成功！


# 提示模板设计
# RAG系统的输出质量很大程度上取决于提示模板的设计。以下是针对问答任务的专业提示模板：

# filepath: example.py
# 定义一个用于问答的提示模板
# prompt_template = """
# Answer the question based only on the following context:
#
# Context:
# {context}
#
# Question: {question}
#
# Helpful Answer:"""

prompt_template = """
Please answer the following question based on the context:

Context:
{context}

Question: {question}

Please reply in the following JSON format:
{
"answer": "Your answer",
"sources": ["source1", "source2"]
}"""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template
)

print("Prompt template created!")  # 提示模板已创建！

# RAG流程集成
# 将所有组件连接起来，构建完整的RAG处理流程：

# filepath: example.py
# 创建一个结合了检索器和语言模型的链
retrieval_qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # "stuff" 只是将所有检索到的文档放入提示中
    retriever=retriever,
    return_source_documents=True,  # 在响应中包含源文档
    chain_type_kwargs={"prompt": prompt}  # 使用我们的自定义提示
)

# retrieval_qa.combine_documents_chain.document_prompt = PromptTemplate(
#     input_variables=["page_content"], template="{page_content}")



print("RAG pipeline assembled and ready to use!")  # RAG 管道已组装好并可以使用！


# 系统测试与应用
# 通过针对亚洲主题的问题测试RAG系统的表现：

 # filepath: example.py
# 帮助清晰显示响应的函数
def ask_question(question):
    print(f"Question: {question}\n")

    # 从我们的 RAG 系统获取响应
    result = retrieval_qa({"query": question})

    print("Answer:")  # 回答：
    print(result["result"])

    print("\nSources:")  # 来源：
    for i, doc in enumerate(result["source_documents"]):
        print(f"Source {i+1}: {doc.metadata.get('source', 'Unknown')}")  # 来源 {i+1}: {doc.metadata.get('source', 'Unknown')}

    print("\n" + "-"*50 + "\n")

# 尝试一些问题
ask_question("How is green tea made?")
ask_question("Can you introduce red for me?")
ask_question("How is white tea defined?")
