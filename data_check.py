import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
import numpy as np

# 加载数据
history = pd.read_csv('data/user_item_behavior_history.csv', header=None, names=['user_id', 'item_id', 'action_id', 'timestamp'], parse_dates=False)
# print("原始时间戳数据类型:", type(history['timestamp'].iloc[0]))
# exit
users = pd.read_csv('data/user_profile.csv', header=None, names=['user_id', 'age', 'gender_id', 'job_id', 'city_id', 'label'])
items = pd.read_csv('data/item_profile.csv', header=None, names=['item_id', 'category_id', 'city_id', 'label'])


# 1. 检查user_id一致性
def check_user_consistency(history, users):
    """检查history中的user是否都在user表中"""
    history_users = set(history['user_id'].unique())
    user_table_users = set(users['user_id'].unique())

    # 找出异常用户
    orphan_users = history_users - user_table_users
    return orphan_users


# 2. 检查item_id一致性
def check_item_consistency(history, items):
    """检查history中的item是否都在item表中"""
    history_items = set(history['item_id'].unique())
    item_table_items = set(items['item_id'].unique())

    # 找出异常物品
    orphan_items = history_items - item_table_items
    return orphan_items


# 执行检查
orphan_users = check_user_consistency(history, users)
orphan_items = check_item_consistency(history, items)

# 打印结果
print(f"异常用户数量: {len(orphan_users)}")
print(f"异常物品数量: {len(orphan_items)}")
print(f"前10个异常用户: {list(orphan_users)[:10]}")
print(f"前10个异常物品: {list(orphan_items)[:10]}")

clean_history = history[~history['user_id'].isin(orphan_users) & ~history['item_id'].isin(orphan_items)

        ].copy()

clean_history.to_csv('data/cleaned_history.csv', index=False)