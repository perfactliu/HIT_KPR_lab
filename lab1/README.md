## Requirements:
 pip install gensim\
 pip install scikit-learn\
 pip install matplotlib\
 pip intall jieba\
 pip install torch\
 set TRANSFORMERS_OFFLINE=1 //linux为export TRANSFORMERS_OFFLINE=1\
 pip install transformers\
 pip install safetensors\
 pip install datasets\
 pip install accelerate -U\
 pip install evaluate\
 pip install seqeval
## Code:
|  Code   | Function  |
|  ----  | ----  |
| process.py | 将文本进行分词 |
| word2vec.py  | 训练词向量模型 |
| train_ner.py & train_cls.py | 训练文本文本表示模型
## TODO:
在根目录下新建data文件夹，并加入一段文本txt文档。记得更新代码文件路径。