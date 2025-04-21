import pandas as pd
from datetime import datetime

# 加载数据
history = pd.read_csv('data/user_item_behavior_history.csv', header=None, names=['user_id', 'item_id', 'action_id', 'timestamp'], parse_dates=['timestamp'])
users = pd.read_csv('data/user_profile.csv', header=None, names=['user_id', 'age', 'gender_id', 'job_id', 'city_id', 'label'])
items = pd.read_csv('data/item_profile.csv', header=None, names=['item_id', 'category_id', 'city_id', 'label'])
# label 为 -1 或者435;320
print("history")
print(history.head())
# print(history.columns)
print("users")
# print(users.columns)
print("items")
# print(items.columns)
print(items.head(50))
# 确保交互数据按时间排序
# interactions = interactions.sort_values(by=['user_id', 'timestamp'])