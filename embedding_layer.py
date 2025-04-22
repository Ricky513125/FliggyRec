import torch
import torch.nn as nn
from collections import defaultdict
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from DynamicWeightedAverage import DynamicWeightedAverage


# ----------------- 修正1：独立损失函数 -----------------
def weighted_bce_loss(outputs, targets, weights=None):
    """
    独立定义的加权损失函数
    Args:
        outputs: 模型预测值 (batch_size,)
        targets: 真实标签 (batch_size,)
        weights: 样本权重 (batch_size,)
    """
    if weights is not None:
        loss = weights * (targets * torch.log(outputs) + (1 - targets) * torch.log(1 - outputs))
    else:
        loss = targets * torch.log(outputs) + (1 - targets) * torch.log(1 - outputs)
    return -torch.mean(loss)


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
        # 确保user_feat_sizes包含job_id
        assert 'job_id' in user_feat_sizes, "user_feat_sizes必须包含job_id"
        # 用户侧Embedding
        self.user_id_emb = nn.Embedding(user_feat_sizes['user_id'], 64)
        self.gender_emb = nn.Embedding(user_feat_sizes['gender_id'], 16)
        self.job_emb = nn.Embedding(user_feat_sizes['job_id']+1, 16)
        self.city_emb = nn.Embedding(user_feat_sizes['city_id']+1, 16)
        self.age_emb = nn.Embedding(10, 16)  # 假设年龄分10个桶

        # 物品侧Embedding
        self.item_id_emb = nn.Embedding(item_feat_sizes['item_id'], 64)
        self.category_emb = nn.Embedding(item_feat_sizes['category_id'], 32)

        # # 多标签处理
        # self.label_emb = MultiLabelEmbedding(label_vocab_size, 32)

        # 修改多标签处理部分
        self.label_embed = nn.Embedding(label_vocab_size, 32)  # 标签Embedding层
        self.dynamic_pool = DynamicWeightedAverage(32)  # 动态池化层

        # print('user_id_emb', self.user_id_emb.weight.shape)
        # print('gender_emb', self.gender_emb.weight.shape)
        # print('job_emb', self.job_emb.weight.shape)
        # print('city_emb', self.city_emb.weight.shape)
        # print('age_emb', self.age_emb.weight.shape)
        # print('item_id_emb', self.item_id_emb.weight.shape)
        # print('category_emb', self.category_emb.weight.shape)
        # print('label_embed', self.label_embed.weight.shape)
        # print('dynamic_pool', self.dynamic_pool.weight.shape)


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

        # job_id 验证
        # 在模型forward开头添加
        # print("job_id取值范围：", user_data['job_id'].min(), user_data['job_id'].max())
        # assert (user_data['job_id'] >= 0).all(), "存在负值job_id"
        # assert (user_data['job_id'] < self.job_emb.num_embeddings).all(), "job_id超出embedding范围"





        # print('current_age', user_data['age'])

        # print('current_label', user_data['label'])
        # print('current_dynamic_pool', self.dynamic_pool.weight.shape)

        # 用户特征处理
        print('current_user_id', user_data['user_id'])
        u_id = self.user_id_emb(user_data['user_id'])
        print('current_gender', user_data['gender_id'])
        u_gender = self.gender_emb(user_data['gender_id'])
        print('current_job', user_data['job_id'])
        u_job = self.job_emb(user_data['job_id'])
        print('current_city', user_data['city_id'])
        u_city = self.city_emb(user_data['city_id'])
        print('current_age', user_data['age_bucket'])
        u_age = self.age_emb(user_data['age_bucket'])
        # u_labels = self.label_emb(user_data['label_list'])
        user_labels = torch.cat([torch.LongTensor(x).to(device) for x in user_data['label_list']])
        # user_labels = user_data['label_list']  # 已经是拼接好的Tensor
        user_label_emb = self.label_embed(user_labels)
        u_labels_pooled = self.dynamic_pool(user_label_emb, user_data['label_length'])

        # 物品特征处理
        print('current_item', user_data['item_id'])
        i_id = self.item_id_emb(item_data['item_id'])
        print('current_category', item_data['category_id'])
        i_category = self.category_emb(item_data['category_id'])
        print('current_item_city', item_data['city_id'])
        i_city = self.city_emb(item_data['city_id'])  # 复用用户city embedding
        # i_labels = self.label_emb(item_data['label_list'])
        # 物品侧动态池化（同理）
        item_labels = torch.cat([torch.LongTensor(x).to(device) for x in item_data['label_list']])
        item_label_emb = self.label_embed(item_labels)
        i_labels_pooled = self.dynamic_pool(item_label_emb, item_data['label_length'])

        # # 分段处理（根据label_length）
        # user_pooled = []
        # start = 0
        # for length in user_data['label_length']:
        #     end = start + length
        #     segment = user_label_emb[start:end]
        #     weights = self.weight_net(segment)
        #     user_pooled.append(torch.sum(segment * weights, dim=0))
        #     start = end
        #
        # user_label_feat = torch.stack(user_pooled)


        # 合并其他特征
        # u_emb = torch.cat([
        #     self.user_id_emb(user_data['user_id']),
        #     self.gender_emb(user_data['gender_id']),
        #     u_labels_pooled  # 使用池化后的标签特征
        # ], dim=1)
        #
        # i_emb = torch.cat([
        #     self.item_id_emb(item_data['item_id']),
        #     self.category_emb(item_data['category_id']),
        #     i_labels_pooled
        # ], dim=1)

        # 特征拼接 水平方向拼接，行数不变batch_size，列数变化
        user_feat = torch.cat([u_id, u_gender, u_job, u_city, u_age, u_labels_pooled], dim=1)
        item_feat = torch.cat([i_id, i_category, i_city, i_labels_pooled], dim=1)

        # 通过双塔
        user_vec = self.user_tower(user_feat)
        item_vec = self.item_tower(item_feat)

        return torch.sigmoid((user_vec * item_vec).sum(dim=1))


# # ----------------- 修正2：独立训练函数 -----------------
# def train_epoch(model, dataloader, optimizer, device):
#     """
#     独立定义的训练函数
#     Args:
#         model: 模型实例
#         dataloader: 数据加载器
#         optimizer: 优化器
#         device: 计算设备
#     """
#     model.train()
#     total_loss = 0
#
#     for user_data, item_data, labels in dataloader:
#         # ----------------- 修正3：正确的设备转移 -----------------
#         # 处理用户数据
#         user_batch = {
#             'user_id': user_data['user_id'].to(device),
#             'gender_id': user_data['gender_id'].to(device),
#             'age_bucket': user_data['age_bucket'].to(device),
#             'label_list': user_data['label_list']  # 列表类型不转移设备
#         }
#
#         # 处理物品数据
#         item_batch = {
#             'item_id': item_data['item_id'].to(device),
#             'category_id': item_data['category_id'].to(device),
#             'label_list': item_data['label_list']  # 列表类型不转移设备
#         }
#
#         labels = labels.to(device).float()  # 确保标签是浮点型
#
#         # 梯度清零
#         optimizer.zero_grad()
#
#         # 前向传播
#         outputs = model(user_batch, item_batch)
#
#         # 计算损失
#         loss = weighted_bce_loss(outputs.squeeze(), labels)
#
#         # 反向传播
#         loss.backward()
#         optimizer.step()
#
#         total_loss += loss.item()
#
#     return total_loss / len(dataloader)