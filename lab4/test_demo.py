import torch
from transformers import BertTokenizer
from model import BertNer
from config import NerConfig
import codecs
import numpy as np

args = NerConfig("duie")
# 加载模型和分词器
MODEL_PATH = "./model_hub/chinese-bert-wwm-ext/"
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertNer(args)
model.load_state_dict(torch.load("checkpoint/duie/pytorch_model_ner.bin"))
model.eval()


def predict_ner(text):
    # 分词和编码
    if len(text) > args.max_seq_len - 2:
        text = text[:args.max_seq_len - 2]
    tmp_input_ids = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    tmp_input_ids = tmp_input_ids["input_ids"].squeeze(0).tolist()
    attention_mask = [1] * len(tmp_input_ids)
    input_ids = tmp_input_ids + [0] * (args.max_seq_len - len(tmp_input_ids))
    # print(input_ids)
    attention_mask = attention_mask + [0] * (args.max_seq_len - len(attention_mask))
    # print(np.array(attention_mask).shape)
    input_ids = torch.tensor(np.array(input_ids)).unsqueeze(0)
    attention_mask = torch.tensor(np.array(attention_mask)).unsqueeze(0)
    with torch.no_grad():
        output = model(input_ids, attention_mask, None)
    logits = output.logits
    attention_mask = attention_mask.detach().cpu().numpy()
    length = sum(attention_mask[0])
    logit = logits[0][1:length]
    logit = [args.id2label[i] for i in logit]
    return logit


with codecs.open("duie_data/demo.json", 'r', encoding="utf-8", errors="replace") as fp:
    lines = fp.read().strip().split("\n")
for i, line in enumerate(lines):
    data = eval(line)
    text = data["text"]
    print(f"text[{i+1}]: {text}")
    predicted_entities = predict_ner(text)
    print(f"prediction[{i+1}]: {predicted_entities}")
