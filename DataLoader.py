from torch.utils.data import Dataset, DataLoader


class RecDataset(Dataset):
    def __init__(self, users, items, interactions):
        self.users = users
        self.items = items
        self.interactions = interactions  # 包含user_id, item_id, label的DataFrame

    def __getitem__(self, idx):
        user_id = self.interactions.iloc[idx]['user_id']
        item_id = self.interactions.iloc[idx]['item_id']
        label = self.interactions.iloc[idx]['label']

        user_data = {
            'user_id': user_id,
            'gender_id': self.users.loc[user_id, 'gender_id'],
            'age_bucket': self.users.loc[user_id, 'age_bucket'],
            'label_list': self.users.loc[user_id, 'label_list']
        }

        item_data = {
            'item_id': item_id,
            'category_id': self.items.loc[item_id, 'category_id'],
            'label_list': self.items.loc[item_id, 'label_list']
        }

        return user_data, item_data, label