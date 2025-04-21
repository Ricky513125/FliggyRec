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

# 2. 安全转换时间戳（处理数值型和字符串型混合情况）
def convert_timestamp(ts):
    try:
        # 先尝试转换为数值型
        ts = pd.to_numeric(ts, errors='raise')
        # 数值型时间戳转换（秒级）
        return pd.to_datetime(ts, unit='s', errors='raise')
    except:
        # 如果已经是日期字符串，直接解析
        return pd.to_datetime(ts, errors='coerce')

# label 为 -1 或者435;320
# print("history")
# print(history.head())
# # print(history.columns)
# print("users")
# # print(users.columns)
# print("items")
# # print(items.columns)
# print(items.head(50))
# # 确保交互数据按时间排序
# interactions = interactions.sort_values(by=['user_id', 'timestamp'])


# 划分常规训练/测试集（时间敏感型划分）
# history['timestamp'] = pd.to_datetime(history['timestamp'], unit='s')  # 关键参数unit='s'
history['timestamp'] = convert_timestamp(history['timestamp'])

# 3. 验证转换结果
print("时间戳示例（前5行）:")
print(history['timestamp'].head())
print("\n时间戳统计信息:")
print(history['timestamp'].describe())

# 4. 检查并处理无效时间戳
if history['timestamp'].isnull().any():
    print(f"\n警告: 发现 {history['timestamp'].isnull().sum()} 条无效时间戳记录")
    history = history.dropna(subset=['timestamp'])




history = history.sort_values('timestamp')

# 按时间划分（保留最后20%作为常规测试集）
split_time = history['timestamp'].quantile(0.8)
train_history = history[history['timestamp'] < split_time]
test_history = history[history['timestamp'] >= split_time]

# 3. 冷启动测试集构建（修正版）
def get_cold_entities(full_data, train_data, entity_col):
    """获取训练集中未出现过的实体ID"""
    train_entities = set(train_data[entity_col].unique())
    return set(full_data[entity_col].unique()) - train_entities

# 获取冷启动用户和物品
cold_users = get_cold_entities(users, train_history, 'user_id')
cold_items = get_cold_entities(items, train_history, 'item_id')

# 构建冷启动测试集（使用安全的布尔索引）
user_cold_mask = test_history['user_id'].isin(cold_users)
item_cold_mask = test_history['item_id'].isin(cold_items)

user_cold_test = test_history[user_cold_mask]
item_cold_test = test_history[item_cold_mask]
mixed_cold_test = test_history[user_cold_mask | item_cold_mask]

# 4. 常规测试集（排除冷启动样本）
test_history = test_history[~(user_cold_mask | item_cold_mask)]

def print_stats(df, name):
    print(f"{name}统计:")
    print(f"- 用户数: {df['user_id'].nunique()}")
    print(f"- 物品数: {df['item_id'].nunique()}")
    print(f"- 记录数: {len(df)}\n")

print_stats(train_history, "常规训练集")
print_stats(test_history, "常规测试集")
print_stats(user_cold_test, "用户冷启动测试集")
print_stats(item_cold_test, "物品冷启动测试集")
print_stats(mixed_cold_test, "混合冷启动测试集")

train_history.to_csv('data/train_regular.csv', index=False)
test_history.to_csv('data/test_regular.csv', index=False)
user_cold_test.to_csv('data/test_user_cold.csv', index=False)
item_cold_test.to_csv('data/test_item_cold.csv', index=False)
mixed_cold_test.to_csv('data/test_mixed_cold.csv', index=False)