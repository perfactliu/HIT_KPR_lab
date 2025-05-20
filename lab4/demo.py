from ltp import LTP

ltp = LTP()
# 加载模型
sentence = "小飞同学今年夏天要去上海参加上海交通大学的夏令营。"
# 两种任务：先分词、再进行命名实体识别
result = ltp.pipeline([sentence], tasks=["cws", "ner"])
print(result.ner)
