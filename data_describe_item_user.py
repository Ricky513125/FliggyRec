import pandas as pd
import numpy as np

# get_Data
users = pd.read_csv("/data/user_profile.csv")
items = pd.read_csv("/data/item_profile.csv")


def get_column_stats(df, name):
    stats = {}
    for col in df.columns:
        if col == 'label':
            # 特殊处理多标签列
            split_values = df[col][df[col] != '-1'].str.split(';').explode()
            stats[col] = {
                'unique_count': split_values.nunique(),
                'sample_values': split_values.value_counts().head(3).index.tolist()
            }
        else:
            # 普通列处理
            stats[col] = {
                'unique_count': df[col].nunique(),
                'sample_values': df[col].value_counts().head(3).index.tolist()
            }
    return pd.DataFrame.from_dict(stats, orient='index',
                                 columns=['unique_count', 'sample_values']).rename_axis(name)

# 生成报告
user_stats = get_column_stats(users, 'users')
item_stats = get_column_stats(items, 'items')

pd.concat([user_stats, item_stats], axis=0)