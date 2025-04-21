import pandas as pd
import numpy as np

history = pd.read_csv('data/user_item_behavior_history.csv', header=None, names=['user_id', 'item_id', 'action_id', 'timestamp'], parse_dates=False)
# 假设数据已加载
history['timestamp'] = pd.to_datetime(history['timestamp'])

print(history.isnull().sum())
# 定义有效时间范围（根据业务需求调整）
valid_start = pd.Timestamp('2019-06-03 00:00:00')
valid_end = pd.Timestamp('2021-06-03 23:59:59')

# 识别异常记录
out_of_range = ~history['timestamp'].between(valid_start, valid_end)
anomaly_count = out_of_range.sum()
print(f"发现异常时间记录: {anomaly_count}条 ({anomaly_count/len(history):.2%})")
cleaned_history = history[history['timestamp'].between(valid_start, valid_end)]
clean_history.to_csv('data/cleaned_history.csv', index=False)
print("Successfully clean the history dataset!")