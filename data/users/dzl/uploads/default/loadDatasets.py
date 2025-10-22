import json

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

json_file = './Gift_Cards.jsonl'
df = pd.read_json(json_file, lines=True)
print(df.head())
print(df.columns.values)
df.drop(['title', 'images', 'asin', 'parent_asin', 'user_id','timestamp','helpful_vote','verified_purchase'], axis=1, inplace=True)

print(df.head())
# 评分转标签：1-2分=负面(0)，3分=中性(1)，4-5分=正面(2)
labels = []
for label in df['rating'].values:
    if label <= 2:
        labels.append(0)
    elif label == 3:
        labels.append(1)
    else:
        labels.append(2)
print(len(labels))
print(labels[:10])
texts = list([text for text in df["text"]])

train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

tokenizer = BertTokenizer.from_pretrained('/home/dzl/baDouNLP/week/google-bert/bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('/home/dzl/baDouNLP/week/google-bert/bert-base-uncased',
                                                      num_labels=3)

# 假设 train_texts 是你的文本列表
lengths = [len(tokenizer.encode(text)) for text in train_texts]

print("最大长度:", max(lengths))
print("95%分位数:", np.percentile(lengths, 95))
print("平均长度:", np.mean(lengths))
print("中位数:", np.median(lengths))

train_encodings = tokenizer(train_texts, truncation=True, padding=True,max_length=128)
test_encodings= tokenizer(test_texts, truncation=True, padding=True,max_length=128)

# 将编码后的数据和标签转换为 Hugging Face `datasets` 库的 Dataset 对象
train_dataset = Dataset.from_dict({
    'input_ids': train_encodings['input_ids'],           # 文本的token ID
    'attention_mask': train_encodings['attention_mask'], # 注意力掩码
    'labels': train_labels                               # 对应的标签
})
test_dataset = Dataset.from_dict({
    'input_ids': test_encodings['input_ids'],
    'attention_mask': test_encodings['attention_mask'],
    'labels': test_labels
})


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    # 基础指标（保持不变）
    acc = accuracy_score(labels, predictions)
    precision_macro = precision_score(labels, predictions, average='macro')
    recall_macro = recall_score(labels, predictions, average='macro')
    f1_macro = f1_score(labels, predictions, average='macro')
    f1_weighted = f1_score(labels, predictions, average='weighted')

    # 获取每个类别的指标（返回数组，顺序对应类别 0, 1, 2, ..., num_classes-1）
    precisions = precision_score(labels, predictions, average=None, zero_division=0)
    recalls = recall_score(labels, predictions, average=None, zero_division=0)
    f1s = f1_score(labels, predictions, average=None, zero_division=0)

    # 获取类别数量（可选：用于命名）
    num_classes = len(precisions)

    # 构建 per-class 指标字典
    per_class_metrics = {}
    for i in range(num_classes):
        per_class_metrics[f"precision_class_{i}"] = float(precisions[i])
        per_class_metrics[f"recall_class_{i}"] = float(recalls[i])
        per_class_metrics[f"f1_class_{i}"] = float(f1s[i])

    # 合并所有指标
    return {
        'accuracy': acc,
        'f1_macro': f1_macro,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_weighted': f1_weighted,
        **per_class_metrics,  # 展开 per-class 指标
    }
from sklearn.utils.class_weight import compute_class_weight
# ----- 1. 计算类别权重（解决不平衡） -----
train_labels = np.array(train_dataset["labels"])
num_classes = len(np.unique(train_labels))
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(num_classes),
    y=train_labels
)
# class_weights = torch.tensor(class_weights, dtype=torch.float)
# 在计算 class_weights 后，强制提升 class_1
class_weights = torch.tensor([3.6, 30.0, 0.4])  # class_1 权重 ×2

# ----- 2. 自定义带权重的 Trainer -----
class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # class_weights 应为 1D tensor，长度 = num_labels
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute loss with optional class weights.
        Compatible with Transformers >=4.30
        """
        # 1. 前向传播
        outputs = model(**inputs)
        logits = outputs.get("logits")
        labels = inputs.get("labels")

        # 2. 选择损失函数
        if self.class_weights is not None:
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        else:
            loss_fct = nn.CrossEntropyLoss()

        # 3. 计算 loss
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))

        # 4. 返回格式必须与父类一致
        return (loss, outputs) if return_outputs else loss


# 配置训练参数
training_args = TrainingArguments(
    output_dir='./ClassificationResultsBalanced',              # 训练输出目录，用于保存模型和状态
    num_train_epochs=8,                  # 训练的总轮数
    per_device_train_batch_size=64,      # 训练时每个设备（GPU/CPU）的批次大小
    per_device_eval_batch_size=64,       # 评估时每个设备的批次大小
    warmup_steps=500,                    # 学习率预热的步数，有助于稳定训练
    learning_rate=2e-5,
    weight_decay=0.01,                   # 权重衰减，用于防止过拟合
    logging_dir='./logs',                # 日志存储目录
    logging_steps=20,                   # 每隔100步记录一次日志

    # eval_strategy="epoch",               # 每训练完一个 epoch 进行一次评估
    # save_strategy="best",               # 每训练完一个 epoch 保存一次模型
    # load_best_model_at_end=True,         # 训练结束后加载效果最好的模型

    save_strategy="epoch",  # 或 "steps"，但需配合 evaluation_strategy
    eval_strategy="epoch",  # 注意：新版推荐用 evaluation_strategy
    save_total_limit=2,  # 只保留 best 2 个模型，节省空间
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",  # 👈 关键！指定根据 f1_macro 选 best
    greater_is_better=True,  # f1_macro 越大越好
    # 👇👇👇 关键：启用 wandb
    report_to="wandb",  # 启用 wandb 报告
    run_name="my-classification-run",  # 可选：自定义运行名称（在 wandb 看板显示）
    logging_strategy="steps",  # 按步记录日志（默认也是 steps）
)

# 实例化 Trainer
# trainer = Trainer(
#     model=model,                         # 要训练的模型
#     args=training_args,                  # 训练参数
#     train_dataset=train_dataset,         # 训练数据集
#     eval_dataset=test_dataset,           # 评估数据集
#     compute_metrics=compute_metrics,     # 用于计算评估指标的函数
# )
# ----- 5. 实例化自定义 Trainer -----
trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
    class_weights=class_weights,  # ✅ 传入权重
)
# 开始训练模型
trainer.train()
# 在测试集上进行最终评估
trainer.evaluate()
trainer.save_model("best")
print("Done")