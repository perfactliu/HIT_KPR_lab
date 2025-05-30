# 哈工大知识表示与推理实验介绍
$~~~~~~~~$知识表示与推理是研究如何将人类世界中的知识更加高效、有逻辑、全面的表示与存储在计算机中的学科。随着近几十年来机器学习的进步，使用人工智能方法来处理人类知识之间的依赖、推理等各种逻辑关系逐渐收到了广泛的关注。在现实生活中，有很多应用知识表示推理的例子。实际上，浏览器搜索引擎就是一个最重要的应用，近年来兴起的大模型agent之所以能回答各种领域方方面面的问题，也与知识推理的训练脱不开关系。\
$~~~~~~~~$本门课由哈工大SCIR实验室的fxc老师教授，从课程到实验再到考试都令人感到一言难尽，笔者在这里不展开，下面简单介绍一下实验。本门课共8次实验，基本上都是改编自近年比较经典的AI领域论文，很多实验代码都是直接copy自网络，并挖了一些空来填。此外，由于SCIR实验室的性质，实验会更偏向NLP、LLM等方向。下面给出8次实验的简介，详细信息请进入各个文件夹获取。
## 实验1：文本表示推理
包含Word2Vec词表示、LSTM搭建与训练、基于BERT的词表示与句子表示三个子实验。主要展示了文本机器学习近年来的进展。
## 实验2：视觉表示学习
MNIST手写数字识别实验。
## 实验3：图表示学习
知识图谱是知识表示的重要方法之一。本次实验包含图节点表示、图神经网络两部分。
## 实验4：命名实体识别
命名实体识别（NER）是NLP领域中一项基础且重要的任务。本实验使用了bert、ltp、大模型等方式实现。
## 实验5：关系抽取
与NER一样，关系抽取也是一项基础任务。本实验参考了三篇论文，包括基于卷积神经网络、远程监督、预训练模型的关系抽取。（吐了，纯纯的预制实验。。。）
## 实验6：知识图谱补全
包括TransE、TransH、TransR三种方法，实现(h,r,t)的预测。
## 实验7：基于知识图谱的应用（上）
基于BM25算法实现的文档检索器。
## 实验8：基于知识图谱的应用（下）
通过预训练模型与大语言模型实现的问答系统。