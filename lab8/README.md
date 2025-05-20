## Requirements:
pip install langchain torch transformers faiss-cpu  # CPU版本，GPU版需安装faiss-gpu\
pip install sentence-transformers  # 嵌入模型依赖
## TODO:
在根文件夹中新建一个自定义文档集，其中加入相关的txt文本作为知识文本。在RAG.py文件中更改问题，以与知识基础文档集适配。