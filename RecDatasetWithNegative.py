from torch.utils.data import Dataset, DataLoader

class RecDatasetWithNegative(Dataset):
    def __init__(self, users, items, interactions, neg_ratio=3):
        self.users = users.set_index('user_id')
        self.items = items.set_index('item_id')
        self.interactions = interactions
        self.all_item_ids = items['item_id'].unique()
        self.neg_ratio = neg_ratio

    def __getitem__(self, idx):
        # 正样本
        pos_row = self.interactions.iloc[idx]
        user_id = pos_row['user_id']

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

        # 组装数据
        user_data = {
            'user_id': torch.LongTensor(batch_user),
            'gender_id': torch.LongTensor([self.users.loc[user_id, 'gender_id']] * (1 + self.neg_ratio)),
            'age_bucket': torch.LongTensor([self.users.loc[user_id, 'age_bucket']] * (1 + self.neg_ratio)),
            'label_list': [self.users.loc[user_id, 'label_list']] * (1 + self.neg_ratio)
        }

        item_data = {
            'item_id': torch.LongTensor(batch_item),
            'category_id': torch.LongTensor([self.items.loc[i, 'category_id'] for i in batch_item]),
            'label_list': [self.items.loc[i, 'label_list'] for i in batch_item]
        }

        return user_data, item_data, torch.FloatTensor(batch_label)