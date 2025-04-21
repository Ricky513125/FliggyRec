import pandas as pd
import numpy as np

history = pd.read_csv('data/user_item_behavior_history.csv', header=None, names=['user_id', 'item_id', 'action_id', 'timestamp'], parse_dates=False)
# 假设数据已加载
history['timestamp'] = pd.to_datetime(history['timestamp'])


def convert_timestamp(ts):
    try:
        # 先尝试转换为数值型
        ts = pd.to_numeric(ts, errors='raise')
        # 数值型时间戳转换（秒级）
        return pd.to_datetime(ts, unit='s', errors='raise')
    except:
        # 如果已经是日期字符串，直接解析
        return pd.to_datetime(ts, errors='coerce')

# print(history.isnull().sum())
# 定义有效时间范围（根据业务需求调整）
valid_start = pd.Timestamp('2019-06-03 00:00:00')
valid_end = pd.Timestamp('2021-06-03 23:59:59')
print(type(histor['timestamp']))
for i in range(10):
    print("history", history['timestamp'].iloc[i], history['timestamp'].iloc(i) > valid_start)
# 识别异常记录
out_of_range = ~history['timestamp'].between(valid_start, valid_end)
anomaly_count = out_of_range.sum()
print(f"发现异常时间记录: {anomaly_count}条 ({anomaly_count/len(history):.2%})")
exit
cleaned_history = history[history['timestamp'].between(valid_start, valid_end)]
cleaned_history.to_csv('data/cleaned_history2.csv', index=False)
print("Successfully clean the history dataset!")