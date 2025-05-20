from transformers import AutoTokenizer, TrainingArguments, Trainer
from transformers import DataCollatorWithPadding
from datasets import load_dataset
import numpy as np
import evaluate
from datasets import DownloadMode
import torch
from BERT import DistilBertForSequenceClassification
from seqeval.metrics import accuracy_score


id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}

tokenizer = AutoTokenizer.from_pretrained("./cache/distilbert")
model = DistilBertForSequenceClassification.from_pretrained("./cache/distilbert", num_labels=2, id2label=id2label, label2id=label2id)

dataset = load_dataset("stanfordnlp___parquet", cache_dir="./cache", download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS)
# accuracy = evaluate.load("accuracy", cache_dir="./cache")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    results = {
        'accuracy': accuracy_score(predictions, labels)
    }
    return results


# 预处理数据：Tokenization
def preprocess_function(examples):
    return tokenizer(examples["sentence"], truncation=True, padding=True, max_length=128)


encoded_datasets = dataset.map(preprocess_function, batched=True)

# 划分训练集和验证集
train_dataset = encoded_datasets["train"]
eval_dataset = encoded_datasets["validation"]

# 数据整理器（自动填充批量数据）
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 训练参数
training_args = TrainingArguments(
    output_dir="./ckpt/CLS_ckpt",
    evaluation_strategy="epoch",  # 每个 epoch 进行验证
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
    save_total_limit=2,  # 仅保留最近两个 checkpoint
    load_best_model_at_end=True,  # 训练结束时加载最佳模型
    metric_for_best_model="accuracy",
)

# 定义 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# 训练模型
trainer.train()

# 评估模型
eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")
