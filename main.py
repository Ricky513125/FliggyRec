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


def dynamic_collate_fn(batch):
    """支持变长标签且自动处理设备转移的collate函数"""
    # 分离用户、物品、标签数据
    user_batch, item_batch, labels = zip(*batch)

    # 自动获取当前设备（与模型相同的设备）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 处理用户数据
    user_data = {
        'user_id': torch.cat([x['user_id'] for x in user_batch]).to(device),
        'gender_id': torch.cat([x['gender_id'] for x in user_batch]).to(device),
        'job_id': torch.cat([x['job_id'] for x in user_batch]).to(device),  # 新增
        'age_bucket': torch.cat([x['age_bucket'] for x in user_batch]).to(device),
        # 'label_list': [label.to(device) for x in user_batch for label in x['label_list']],  # 展平+设备转移
        'label_list': torch.cat([x['label_list'] for x in user_batch]).to(device),  # 直接concat
        'label_length': torch.cat([x['label_length'] for x in user_batch]).to(device)
    }

    # 处理物品数据
    item_data = {
        'item_id': torch.cat([x['item_id'] for x in item_batch]).to(device),
        'category_id': torch.cat([x['category_id'] for x in item_batch]).to(device),
        # 'label_list': [label.to(device) for x in item_batch for label in x['label_list']],
        'label_list': torch.cat([x['label_list'] for x in item_batch]).to(device),
        'label_length': torch.cat([x['label_length'] for x in item_batch]).to(device)
    }

    return user_data, item_data, torch.cat(labels).to(device).float()


# 设备设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

users = pd.read_csv('data/user_profile.csv', header=None, names=['user_id', 'age', 'gender_id', 'job_id', 'city_id', 'label'])
items = pd.read_csv('data/item_profile.csv', header=None, names=['item_id', 'category_id', 'city_id', 'label'])
interactions = pd.read_csv('data/interactions.csv')

print("用户数据列：", users.columns.tolist())  # 必须包含job_id
assert 'job_id' in users.columns, "原始数据缺少job_id列"

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
    'job_id': users['job_id'].max() + 2,
    'city_id': users['city_id'].max() + 2
}
# 检查特征词典
print("user_feat_sizes内容：", user_feat_sizes)  # 必须包含job_id
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
dataloader = DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=dynamic_collate_fn) # 关键：使用自定义collate函数)
# 在创建dataloader后添加检查
print(f"Dataset length: {len(dataset)}")  # 应该输出正样本数 × (1 + neg_ratio)
print(f"Batch数量: {len(dataloader)}")  # 应该输出ceil(数据集长度/batch_size)
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
        # 修改后的数据转移逻辑
        user_batch = {
            'user_id': user_data['user_id'].to(device),
            'gender_id': user_data['gender_id'].to(device),
            'job_id': user_data['job_id'].to(device),
            'age_bucket': user_data['age_bucket'].to(device),
            'label_list': [x.to(device) for x in user_data['label_list']]  # 提前转移标签列表
        }

        item_batch = {
            'item_id': item_data['item_id'].to(device),
            'category_id': item_data['category_id'].to(device),
            'label_list': [x.to(device) for x in item_data['label_list']]  # 提前转移标签列表
        }

        labels = labels.to(device).float()

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