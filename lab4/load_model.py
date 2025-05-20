from transformers import AutoTokenizer, AutoModelForMaskedLM

local_path = "./chinese-bert-wwm-ext"
tokenizer = AutoTokenizer.from_pretrained(local_path)
model = AutoModelForMaskedLM.from_pretrained(local_path)

# 测试一下
text = "今天天气真[MASK]。"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
print("模型加载成功！")
