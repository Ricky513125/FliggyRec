from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
class RecDatasetWithNegative(Dataset):
    def __init__(self, users, items, interactions, neg_ratio=3):
        self.users = users.set_index('user_id')
        self.items = items.set_index('item_id')
        self.interactions = interactions
        self.all_item_ids = items['item_id'].unique()
        self.neg_ratio = neg_ratio
        self.job_emb_num = users['job_id'].max() + 2  # 原始最大值+1 +1（为-1）
        self.city_emb_num = users['city_id'].max() + 2

    def __len__(self):
        """返回数据集的总长度（正样本数 × (1 + 负样本比例)）"""
        # return len(self.interactions) * (1 + self.neg_ratio)
        return len(self.interactions)

    def __getitem__(self, idx):
        # 正样本
        # pos_idx = idx // (1 + self.neg_ratio)
        pos_row = self.interactions.iloc[idx]
        user_id = pos_row['user_id']

        # 处理 job_id
        job_id = self.users.loc[user_id, 'job_id']
        if job_id == -1:
            job_id = self.job_emb_num - 1  # 映射到最后一维
        # else:
        #     job_id += 1  # 原始ID整体偏移（避免与 -1 冲突）

        # 处理 city_id（同理）
        city_id = self.users.loc[user_id, 'city_id']
        if city_id == -1:
            city_id = self.city_emb_num - 1
        # else:
        #     city_id += 1

        # 负采样（确保不在用户历史中）
        user_items = set(self.interactions[self.interactions['user_id'] == user_id]['item_id'])
        neg_items = np.random.choice(
            [x for x in self.all_item_ids if x not in user_items],
            size=self.neg_ratio,
            replace=False
        )



        # 构建批次
        batch_user = []
        batch_item = []
        batch_label = []

        # 添加正样本
        batch_user.append(user_id)
        batch_item.append(pos_row['item_id'])
        batch_label.append(pos_row['label'])

        # 添加负样本
        for item_id in neg_items:
            batch_user.append(user_id)
            batch_item.append(item_id)
            batch_label.append(0.0)  # 负样本权重为0

        # 安全处理age_bucket
        self.users['age_bucket'] = self.users['age_bucket'].fillna(-1).astype(int)
        age_bucket = int(self.users.loc[user_id, 'age_bucket'])
        if not -2147483648 <= age_bucket <= 2147483647:  # int32范围
            age_bucket = 0  # 设置默认值

        # 组装数据
        # user_data = {
        #     'user_id': torch.LongTensor(batch_user),
        #     'gender_id': torch.LongTensor([self.users.loc[user_id, 'gender_id']] * (1 + self.neg_ratio)),
        #     'job_id': torch.LongTensor([self.users.loc[user_id, 'job_id']]),  # 新增job_id
        #     'city_id': torch.LongTensor([self.users.loc[user_id, 'city_id']] * self.neg_ratio),
        #     # 'age_bucket': torch.LongTensor([self.users.loc[user_id, 'age_bucket']] * (1 + self.neg_ratio)),
        #     'age_bucket': torch.LongTensor([age_bucket] * (1 + self.neg_ratio)),
        #     # 'label_list': [self.users.loc[user_id, 'label_list']] * (1 + self.neg_ratio),  # 保持为列表
        #     'label_list': torch.LongTensor(self.users.loc[user_id, 'label_list']),  # 关键修改：转为Tensor
        #     'label_length': torch.LongTensor([len(self.users.loc[user_id, 'label_list'])] * (1 + self.neg_ratio)) # 动态平均池化
        # }

        # item_data = {
        #     'item_id': torch.LongTensor(batch_item),
        #     'category_id': torch.LongTensor([self.items.loc[i, 'category_id'] for i in batch_item]),
        #     # 'label_list': [self.items.loc[i, 'label_list'] for i in batch_item],  # 保持为列表
        #     'city_id': torch.LongTensor([self.items.loc[i, 'city_id'] for i in batch_item]),
        #     'label_list': torch.LongTensor(self.items.loc[item_id, 'label_list']),  # 关键修改：转为Tensor
        #     'label_length': torch.LongTensor([len(self.items.loc[i, 'label_list']) for i in batch_item]) # 动态平均池化
        # }

        user_data = {
            'user_id': torch.LongTensor(batch_user),  # 形状 [1+neg_ratio]
            'gender_id': torch.LongTensor([self.users.loc[user_id, 'gender_id']] * (1 + self.neg_ratio)),
            'job_id': torch.LongTensor([self.users.loc[user_id, 'job_id']] * (1 + self.neg_ratio)),  # 统一扩展
            'city_id': torch.LongTensor([self.users.loc[user_id, 'city_id']] * (1 + self.neg_ratio)),  # 统一扩展
            'age_bucket': torch.LongTensor([age_bucket] * (1 + self.neg_ratio)),
            'label_list': torch.LongTensor(self.users.loc[user_id, 'label_list']),  # 原始标签序列
            'label_length': torch.LongTensor([len(self.users.loc[user_id, 'label_list'])] * (1 + self.neg_ratio))
        }

        item_data = {
            'item_id': torch.LongTensor(batch_item),  # 形状 [1+neg_ratio]
            'category_id': torch.LongTensor([self.items.loc[i, 'category_id'] for i in batch_item]),
            'city_id': torch.LongTensor([self.items.loc[i, 'city_id'] for i in batch_item]),
            'label_list': torch.stack([torch.LongTensor(self.items.loc[i, 'label_list']) for i in batch_item]),  # 堆叠成张量
            'label_length': torch.LongTensor([len(self.items.loc[i, 'label_list']) for i in batch_item])
        }

        print("负采样")
        print("user_data:", user_data)
        print("item_data:", item_data)
        return user_data, item_data, torch.FloatTensor(batch_label)