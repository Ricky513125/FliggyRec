import torch
print(torch.cuda.is_available())  # 应返回True
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
from embedding_layer import RecommenderModel
from RecDatasetWithNegative import RecDatasetWithNegative
# import embedding_layer
import os
from tqdm import tqdm  # 导入进度条库

# 设备设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

users = pd.read_csv('data/user_profile.csv')
items = pd.read_csv('data/item_profile.csv')
interactions = pd.read_csv('data/interactions.csv')

ACTION_WEIGHTS = {
    'clk': 1.0,  # 点击
    'fav': 2.0,  # 收藏
    'cart': 3.0, # 加购
    'pay': 4.0   # 支付
}


# 预处理部分
# 转换多标签列
users['label_list'] = users['label'].apply(lambda x: [] if x == '-1' else [int(i) for i in x.split(';')])
items['label_list'] = items['label'].apply(lambda x: [] if x == '-1' else [int(i) for i in x.split(';')])

# 年龄分桶
users['age_bucket'] = pd.cut(users['age'], bins=[0, 18, 25, 35, 50, 100], labels=False)

# 构建标签词典
all_labels = set()
for labels in users['label_list']:
    all_labels.update(labels)
for labels in items['label_list']:
    all_labels.update(labels)
label_vocab_size = len(all_labels) + 1  # +1 for padding index

# 特征词典大小
user_feat_sizes = {
    'user_id': users['user_id'].max() + 1,
    'gender_id': users['gender_id'].max() + 1,
    'job_id': users['job_id'].max() + 1,
    'city_id': users['city_id'].max() + 1
}
item_feat_sizes = {
    'item_id': items['item_id'].max() + 1,
    'category_id': items['category_id'].max() + 1
}




# 初始化
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RecommenderModel(user_feat_sizes, item_feat_sizes, label_vocab_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 数据加载（带负采样）
dataset = RecDatasetWithNegative(users, items, interactions, neg_ratio=3)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# ----------------- 修正1：独立损失函数 -----------------
def weighted_bce_loss(outputs, targets, weights=None):
    """
    独立定义的加权损失函数
    Args:
        outputs: 模型预测值 (batch_size,)
        targets: 真实标签 (batch_size,)
        weights: 样本权重 (batch_size,)
    """
    epsilon = 1e-7  # 防止log(0)
    outputs = torch.clamp(outputs, epsilon, 1. - epsilon)

    if weights is not None:
        loss = weights * (targets * torch.log(outputs) + (1 - targets) * torch.log(1 - outputs))
    else:
        loss = targets * torch.log(outputs) + (1 - targets) * torch.log(1 - outputs)
    return -torch.mean(loss)


# ----------------- 修正2：独立训练函数 -----------------
def train_epoch(model, dataloader, optimizer, device):
    """
    独立定义的训练函数
    Args:
        model: 模型实例
        dataloader: 数据加载器
        optimizer: 优化器
        device: 计算设备
    """
    model.train()
    total_loss = 0

    # 添加进度条
    progress_bar = tqdm(enumerate(dataloader),
                        total=len(dataloader),
                        desc=f'Epoch {epoch + 1}',
                        ncols=100)

    for user_data, item_data, labels in dataloader:
        # ----------------- 修正3：正确的设备转移 -----------------
        # 处理用户数据
        user_batch = {
            'user_id': user_data['user_id'].to(device),
            'gender_id': user_data['gender_id'].to(device),
            'age_bucket': user_data['age_bucket'].to(device),
            'label_list': user_data['label_list']  # 列表类型不转移设备
        }

        # 处理物品数据
        item_batch = {
            'item_id': item_data['item_id'].to(device),
            'category_id': item_data['category_id'].to(device),
            'label_list': item_data['label_list']  # 列表类型不转移设备
        }

        labels = labels.to(device).float()  # 确保标签是浮点型

        # 梯度清零
        optimizer.zero_grad()

        # 前向传播
        outputs = model(user_batch, item_batch)

        # 计算损失
        loss = weighted_bce_loss(outputs.squeeze(), labels)

        # 反向传播
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # 更新进度条信息
        progress_bar.set_postfix(loss=f'{loss.item():.4f}')

    return total_loss / len(dataloader)

# 训练循环
for epoch in tqdm(range(10), desc='Total Training Progress', ncols=100):
    avg_loss = train_epoch(model, dataloader, optimizer, device)
    print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")

    # 保存checkpoint（包含权重配置）
    if (epoch + 1) % 2 == 0:
        save_path = f"checkpoints/checkpoint_epoch{epoch + 1}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'action_weights': ACTION_WEIGHTS,  # 保存权重配置
            'label_vocab': all_labels
        }, save_path)