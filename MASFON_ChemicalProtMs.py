import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification,get_linear_schedule_with_warmup
import torch.cuda
from transformers import BertModel, AutoTokenizer
from torch.optim import AdamW
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from SaveCsvData import savecsvdata
from SavePicData import savepicdata
from  model import *
import  copy
import random
############################################# 定义 BERT 模型和 tokenizer ##############################################
# SciBERT
# model_path_1 = r'Scibert_scivocab_uncased'
# model1_tokenizer = AutoTokenizer.from_pretrained(model_path_1)

#biobert
model_path_1 = r'biobertv1.1'
model1_tokenizer = AutoTokenizer.from_pretrained(model_path_1)

#pubmedbert
model_path_2 = r'PubMedBERT'
model2_tokenizer = AutoTokenizer.from_pretrained(model_path_2)

#bluebert
# model_path_3 =  r'BlueBERT'
# model3_tokenizer = AutoTokenizer.from_pretrained(model_path_3)

#BioM
model_path_3 =  r'BioM-ELECTRA-Base-Discriminator'
model3_tokenizer = AutoTokenizer.from_pretrained(model_path_3)
print("model load"+model_path_1+"\n"+model_path_2+"\n"+model_path_3)
print("BERT tokenizers loaded")

############################################ 初始化及读取数据 #################################################################
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
# 调用
set_seed(42)

df_train = pd.read_csv('../data/ChemProtMS/train.tsv', sep='\t').iloc[::10]
df_dev = pd.read_csv('../data/ChemProtMS/dev.tsv', sep='\t')
df_test = pd.read_csv('../data/ChemProtMS/test.tsv', sep='\t')
print('data of train:'+ df_train.count().to_string())
print('data of dev:'+ df_dev.count().to_string())
print('data of test:'+ df_test.count().to_string())
####################################################### 定义模型参数 #########################################################
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 定义标签映射
label_map = {
    'CPR:3': 0,
    'CPR:9': 1,
    'CPR:5': 2,
    'CPR:6': 3,
    'CPR:4': 4,
    'false': 5
}


############################################# 定义数据集和数据加载器 ###################################################
class DDIDataset(Dataset):
    def __init__(self, dataframe, tokenizer_1, tokenizer_2, tokenizer_3, max_length):
        self.data = dataframe
        self.tokenizer_1 = tokenizer_1
        self.tokenizer_2 = tokenizer_2
        self.tokenizer_3 = tokenizer_3
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence = self.data.iloc[idx]['sentence']  # 按位置查找
        label_str = self.data.iloc[idx]['label']  # 按位置查找
        label = label_map[label_str]

        encoding_1 = self.tokenizer_1(sentence, max_length=self.max_length, padding='max_length', truncation=True,
                                      return_tensors='pt')
        encoding_2 = self.tokenizer_2(sentence, max_length=self.max_length, padding='max_length', truncation=True,
                                      return_tensors='pt')
        encoding_3 = self.tokenizer_3(sentence, max_length=self.max_length, padding='max_length', truncation=True,
                                      return_tensors='pt')

        return {
            'input_ids_1': encoding_1['input_ids'].flatten(),
            'attention_mask_1': encoding_1['attention_mask'].flatten(),
            'input_ids_2': encoding_2['input_ids'].flatten(),
            'attention_mask_2': encoding_2['attention_mask'].flatten(),
            'input_ids_3': encoding_3['input_ids'].flatten(),
            'attention_mask_3': encoding_3['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def create_data_loader(df, tokenizer_1, tokenizer_2, tokenizer_3, max_length, batch_size):
    dataset = DDIDataset(
        dataframe=df,
        tokenizer_1=tokenizer_1,
        tokenizer_2=tokenizer_2,
        tokenizer_3=tokenizer_3,
        max_length=max_length
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


##################################################### 训练设置 ####################################################
# 定义模型参数
max_length = 300
batch_size = 8
num_labels = len(label_map)
epochs = 25
best_f1 = 0
best_dict = None
train_losses_history = []
dev_losses_history = []
test_losses_history = []
train_f1_history = []
dev_f1_history = []
test_f1_history = []
train_accuracy_history = []
dev_accuracy_history = []
test_accuracy_history = []
train_precision_history = []
dev_precision_history = []
test_precision_history = []
train_recall_history = []
dev_recall_history = []
test_recall_history = []


# 创建数据加载器
train_data_loader = create_data_loader(df_train, model1_tokenizer, model2_tokenizer, model3_tokenizer, max_length,
                                       batch_size)
dev_data_loader = create_data_loader(df_dev, model1_tokenizer, model2_tokenizer, model3_tokenizer, max_length,
                                     batch_size)
test_data_loader = create_data_loader(df_test, model1_tokenizer, model2_tokenizer, model3_tokenizer, max_length,
                                      batch_size)

# 实例化模型
cfg = MultiModelFusionConfig(
    hidden_dim=128,
    mlp_hidden_dim=256,
    num_heads=8,
    dropout_rate=0.3,
    num_labels=num_labels,
    model_names=[model_path_1, model_path_2, model_path_3]
)
model = MASFON(cfg)
model.to(device)


# 定义优化器
optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

# 学习率调度器
total_steps = len(train_data_loader) * epochs  # epochs * batches
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps
)


##################################################### 训练函数 ####################################################
def train_epoch(model, data_loader, optimizer, device, scheduler=None):
    model.train()
    losses = []
    targets = []
    predictions = []
    all_weights = []

    progress_bar = tqdm(data_loader, desc="Training")
    for batch in progress_bar:
        input_ids_1 = batch['input_ids_1'].to(device)
        attention_mask_1 = batch['attention_mask_1'].to(device)
        input_ids_2 = batch['input_ids_2'].to(device)
        attention_mask_2 = batch['attention_mask_2'].to(device)
        input_ids_3 = batch['input_ids_3'].to(device)
        attention_mask_3 = batch['attention_mask_3'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        # 训练时不返回权重以节省内存
        loss, logits = model(
            input_ids_1=input_ids_1, attention_mask_1=attention_mask_1,
            input_ids_2=input_ids_2, attention_mask_2=attention_mask_2,
            input_ids_3=input_ids_3, attention_mask_3=attention_mask_3,
            labels=labels, return_weights=False
        )

        predictions.extend(torch.argmax(logits, dim=1).tolist())
        targets.extend(labels.tolist())
        losses.append(loss.item())

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if scheduler:
            scheduler.step()

        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

    return losses, targets, predictions


def eval_model(model, data_loader, device, return_weights=False):
    model.eval()
    losses = []
    targets = []
    predictions = []
    all_weights = []

    with torch.no_grad():
        progress_bar = tqdm(data_loader, desc="Evaluating")
        for batch in progress_bar:
            input_ids_1 = batch['input_ids_1'].to(device)
            attention_mask_1 = batch['attention_mask_1'].to(device)
            input_ids_2 = batch['input_ids_2'].to(device)
            attention_mask_2 = batch['attention_mask_2'].to(device)
            input_ids_3 = batch['input_ids_3'].to(device)
            attention_mask_3 = batch['attention_mask_3'].to(device)
            labels = batch['labels'].to(device)

            if return_weights:
                loss, logits, weights = model(
                    input_ids_1=input_ids_1, attention_mask_1=attention_mask_1,
                    input_ids_2=input_ids_2, attention_mask_2=attention_mask_2,
                    input_ids_3=input_ids_3, attention_mask_3=attention_mask_3,
                    labels=labels, return_weights=True
                )
                if weights is not None:
                    all_weights.extend(weights.cpu().numpy())
                else:
                    # 可以选择跳过或填充默认值（如全零）
                    print("Warning: weights is None, skipping...")
            else:
                loss, logits = model(
                    input_ids_1=input_ids_1, attention_mask_1=attention_mask_1,
                    input_ids_2=input_ids_2, attention_mask_2=attention_mask_2,
                    input_ids_3=input_ids_3, attention_mask_3=attention_mask_3,
                    labels=labels, return_weights=False
                )

            predictions.extend(torch.argmax(logits, dim=1).tolist())
            targets.extend(labels.tolist())
            if loss is not None:
                losses.append(loss.item())

            progress_bar.set_postfix({'loss': f'{loss.item() if loss is not None else 0:.4f}'})

    if return_weights:
        return losses, targets, predictions, all_weights
    return losses, targets, predictions


def calculate_metrics(targets, predictions, average='weighted'):
    """计算评估指标"""
    accuracy = accuracy_score(targets, predictions)
    precision = precision_score(targets, predictions, average=average, zero_division=0)
    recall = recall_score(targets, predictions, average=average, zero_division=0)
    f1 = f1_score(targets, predictions, average=average, zero_division=0)
    return accuracy, precision, recall, f1


##################################################### 训练循环 ####################################################
best_epoch=0
print("Starting training with model...")

for epoch in range(epochs):
    print(f"\n{'=' * 50}")
    print(f'Epoch {epoch + 1}/{epochs}')
    print(f"{'=' * 50}")

    # 训练
    train_losses, train_targets, train_predictions = train_epoch(
        model, train_data_loader, optimizer, device, scheduler
    )
    train_accuracy, train_precision, train_recall, train_f1 = calculate_metrics(train_targets, train_predictions)

    # 验证
    dev_losses, dev_targets, dev_predictions = eval_model(model, dev_data_loader, device)
    dev_accuracy, dev_precision, dev_recall, dev_f1 = calculate_metrics(dev_targets, dev_predictions)

    #测试
    # test_losses, test_targets, test_predictions = eval_model(model, test_data_loader, device)
    # test_accuracy, test_precision, test_recall, test_f1 = calculate_metrics(test_targets, test_predictions)

    # 记录历史
    train_losses_history.append(np.mean(train_losses))
    dev_losses_history.append(np.mean(dev_losses))
    # test_losses_history.append(np.mean(test_losses))
    train_f1_history.append(train_f1)
    dev_f1_history.append(dev_f1)
    # test_f1_history.append(test_f1)
    train_accuracy_history.append(train_accuracy)
    dev_accuracy_history.append(dev_accuracy)
    # test_accuracy_history.append(test_accuracy)
    train_precision_history.append(train_precision)
    dev_precision_history.append(dev_precision)
    # test_precision_history.append(test_precision)
    train_recall_history.append(train_recall)
    dev_recall_history.append(dev_recall)
    # test_recall_history.append(test_recall)
    # 打印结果
    print(f"\nTraining Results:")
    print(f"  Loss: {np.mean(train_losses):.4f}")
    print(f"  Accuracy: {train_accuracy:.4f}")
    print(f"  Precision: {train_precision:.4f}")
    print(f"  Recall: {train_recall:.4f}")
    print(f"  F1 Score: {train_f1:.4f}")

    print(f"\nDevelopment Results:")
    print(f"  Loss: {np.mean(dev_losses):.4f}")
    print(f"  Accuracy: {dev_accuracy:.4f}")
    print(f"  Precision: {dev_precision:.4f}")
    print(f"  Recall: {dev_recall:.4f}")
    print(f"  F1 Score: {dev_f1:.4f}")

    # print(f"\nTest Results:")
    # print(f"  Loss: {np.mean(test_losses):.4f}")
    # print(f"  Accuracy: {test_accuracy:.4f}")
    # print(f"  Precision: {test_precision:.4f}")
    # print(f"  Recall: {test_recall:.4f}")
    # print(f"  F1 Score: {test_f1:.4f}")

    # 保存最佳模型
    if dev_f1 > best_f1:
        best_f1 = dev_f1
        best_epoch = epoch
        best_dict = copy.deepcopy(model.state_dict())
        print(f"🔥 New best model saved with Dev F1: {best_f1:.4f}")

print(f"\nTraining completed! Best Dev F1: {best_f1:.4f}")

##################################################### 数据保存 ####################################################
# 加载最佳模型
model.load_state_dict(best_dict)
modelname = 'MASFON'
dataname = 'Chem'
output_pretrained_dir = f"./{modelname}_{dataname}"
model.save_pretrained(output_pretrained_dir)
# 保存三个tokenizer
model1_tokenizer.save_pretrained(f"{output_pretrained_dir}/tokenizer1")
model2_tokenizer.save_pretrained(f"{output_pretrained_dir}/tokenizer2")
model3_tokenizer.save_pretrained(f"{output_pretrained_dir}/tokenizer3")
output_filename = f"{dataname}_{modelname}_training_history.csv"

savecsvdata(modelname+dataname+output_filename, epochs,
            train_losses_history, train_f1_history, train_accuracy_history, train_precision_history, train_recall_history,
            dev_losses_history, dev_f1_history, dev_accuracy_history, dev_precision_history, dev_recall_history,
            test_losses_history, test_f1_history, test_accuracy_history, test_precision_history, test_recall_history)
savepicdata(dataname, modelname, epochs,
            train_losses_history, train_f1_history, train_accuracy_history, train_precision_history, train_recall_history,
            dev_losses_history, dev_f1_history, dev_accuracy_history, dev_precision_history, dev_recall_history,
            test_losses_history, test_f1_history, test_accuracy_history, test_precision_history, test_recall_history)


##################################################### 最终评估 ####################################################
print(f"\n{'=' * 50}")
print("Final Evaluation with Best Model")
print(f"{'=' * 50}")

# 最终测试集评估（包含权重分析）
test_losses, test_targets, test_predictions, test_weights = eval_model(
    model, test_data_loader, device, return_weights=True
)

# 绘制混淆矩阵
cm = confusion_matrix(test_targets, test_predictions)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_map.keys(),
            yticklabels=label_map.keys())
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title(modelname+' - Confusion Matrix')
plt.tight_layout()
plt.savefig('confusion_matrix_attention_fusion.png', dpi=300, bbox_inches='tight')

test_accuracy, test_precision, test_recall, test_f1 = calculate_metrics(test_targets, test_predictions)
print(f"最优模型出现在第{best_epoch+1}轮")
print(f"\nFinal Test Results:")
print(f"  Accuracy: {test_accuracy:.4f}")
print(f"  Precision: {test_precision:.4f}")
print(f"  Recall: {test_recall:.4f}")
print(f"  F1 Score: {test_f1:.4f}")

