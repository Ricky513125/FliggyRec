import pandas as pd
import numpy as np

# 1. 加载数据（明确指定列名和数据类型）
history = pd.read_csv('data/user_item_behavior_history.csv',
                     header=None,
                     names=['user_id', 'item_id', 'action_id', 'timestamp'],
                     dtype={'timestamp': int})  # 确保读取为整数

# 2. 转换时间戳（秒级Unix时间戳）
history['timestamp'] = pd.to_datetime(history['timestamp'], unit='s')

# 3. 验证转换结果
print("转换后时间样例（前5条）：")
print(history[['timestamp']].head())

# 4. 定义业务有效时间范围
valid_start = pd.Timestamp('2019-06-03 00:00:00')
valid_end = pd.Timestamp('2021-06-03 23:59:59')

# 5. 检查时间范围
time_mask = history['timestamp'].between(valid_start, valid_end)
print(f"\n有效时间记录比例: {time_mask.mean():.2%}")

# 6. 检查action_id有效性
valid_actions = ['clk', 'fav', 'cart', 'pay']
action_mask = history['action_id'].isin(valid_actions)
print(f"有效action比例: {action_mask.mean():.2%}")

# 7. 综合清洗
clean_history = history[time_mask & action_mask].copy()

# 8. 保存结果
clean_history.to_csv('data/cleaned_history2.csv', index=False)
print(f"\n清洗完成！原始记录: {len(history):,} → 有效记录: {len(clean_history):,}")




# import pandas as pd
# import numpy as np
#
# history = pd.read_csv('data/user_item_behavior_history.csv', header=None, names=['user_id', 'item_id', 'action_id', 'timestamp'], parse_dates=False)
# # 假设数据已加载
# # 毫秒级转换（假设确认是13位时间戳）
# history['timestamp'] = pd.to_datetime(history['timestamp'], unit='ms')
#
# # history['timestamp'] = pd.to_datetime(history['timestamp'])
#
#
# def convert_timestamp(ts):
#     try:
#         # 先尝试转换为数值型
#         ts = pd.to_numeric(ts, errors='raise')
#         # 数值型时间戳转换（秒级）
#         return pd.to_datetime(ts, unit='s', errors='raise')
#     except:
#         # 如果已经是日期字符串，直接解析
#         return pd.to_datetime(ts, errors='coerce')
#
# # print(history.isnull().sum())
# # 定义有效时间范围（根据业务需求调整）
# valid_start = pd.Timestamp('2019-06-03 00:00:00')
# valid_end = pd.Timestamp('2021-06-03 23:59:59')
# # print(type(history['timestamp']))
# for i in range(10):
#     print("history", history['timestamp'].iloc[i], history['timestamp'].iloc[i] > valid_start)
# # 识别异常记录
# out_of_range = ~history['timestamp'].between(valid_start, valid_end)
# anomaly_count = out_of_range.sum()
# print(f"发现异常时间记录: {anomaly_count}条 ({anomaly_count/len(history):.2%})")
#
# action_abnormal = ~history['action_d'].isin(['clk', 'fav', 'cart', 'pay'])
#
#
#
#
# # cleaned_history = history[history['timestamp'].between(valid_start, valid_end)]
# # cleaned_history.to_csv('data/cleaned_history2.csv', index=False)
# # print("Successfully clean the history dataset!")