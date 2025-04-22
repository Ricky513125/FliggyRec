import torch
import torch.nn as nn
from collections import defaultdict


class MultiLabelEmbedding(nn.Module):
    def __init__(self, num_labels, embed_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(num_labels, embed_dim)

    def forward(self, label_lists):
        # label_lists: [[], [1,2], [3]] 这样的批次数据
        all_embeddings = []
        for labels in label_lists:
            if len(labels) == 0:
                # 处理无标签情况
                emb = torch.zeros(self.embedding.embedding_dim)
            else:
                # 对多个标签Embedding取平均
                labels = torch.LongTensor(labels).to(device)
                emb = self.embedding(labels).mean(dim=0)
            all_embeddings.append(emb)
        return torch.stack(all_embeddings)


class RecommenderModel(nn.Module):
    def __init__(self, user_feat_sizes, item_feat_sizes, label_vocab_size):
        super().__init__()
        # 用户侧Embedding
        self.user_id_emb = nn.Embedding(user_feat_sizes['user_id'], 64)
        self.gender_emb = nn.Embedding(user_feat_sizes['gender_id'], 16)
        self.job_emb = nn.Embedding(user_feat_sizes['job_id'], 16)
        self.city_emb = nn.Embedding(user_feat_sizes['city_id'], 16)
        self.age_emb = nn.Embedding(10, 16)  # 假设年龄分10个桶

        # 物品侧Embedding
        self.item_id_emb = nn.Embedding(item_feat_sizes['item_id'], 64)
        self.category_emb = nn.Embedding(item_feat_sizes['category_id'], 32)

        # 多标签处理
        self.label_emb = MultiLabelEmbedding(label_vocab_size, 32)

        # 用户塔和物品塔
        self.user_tower = nn.Sequential(
            nn.Linear(64 + 16 * 3 + 16 + 32, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        self.item_tower = nn.Sequential(
            nn.Linear(64 + 32 + 16 + 32, 256),
            nn.Linear(256, 128)
        )

    def forward(self, user_data, item_data):
        # 用户特征处理
        u_id = self.user_id_emb(user_data['user_id'])
        u_gender = self.gender_emb(user_data['gender_id'])
        u_job = self.job_emb(user_data['job_id'])
        u_city = self.city_emb(user_data['city_id'])
        u_age = self.age_emb(user_data['age_bucket'])
        u_labels = self.label_emb(user_data['label_list'])

        # 物品特征处理
        i_id = self.item_id_emb(item_data['item_id'])
        i_category = self.category_emb(item_data['category_id'])
        i_city = self.city_emb(item_data['city_id'])  # 复用用户city embedding
        i_labels = self.label_emb(item_data['label_list'])

        # 特征拼接 水平方向拼接，行数不变batch_size，列数变化
        user_feat = torch.cat([u_id, u_gender, u_job, u_city, u_age, u_labels], dim=1)
        item_feat = torch.cat([i_id, i_category, i_city, i_labels], dim=1)

        # 通过双塔
        user_vec = self.user_tower(user_feat)
        item_vec = self.item_tower(item_feat)

        return torch.sigmoid((user_vec * item_vec).sum(dim=1))