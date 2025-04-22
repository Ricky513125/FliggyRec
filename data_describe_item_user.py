import pandas as pd
import numpy as np

# get_Data
users = pd.read_csv('data/user_profile.csv', header=None, names=['user_id', 'age', 'gender_id', 'job_id', 'city_id', 'label'])
items = pd.read_csv('data/item_profile.csv', header=None, names=['item_id', 'category_id', 'city_id', 'label'])
# interactions = pd.read_csv('data/interactions.csv')


def get_column_stats(df, name):
    stats = {}
    for col in df.columns:
        if col == 'label':
            # 特殊处理多标签列
            split_values = df[col][df[col] != '-1'].str.split(';').explode()
            stats[col] = {
                'unique_count': split_values.nunique(),
                'sample_values': split_values.value_counts().head(3).index.tolist(),
                'min_value': split_values.min(),
                'max_value': split_values.max()
            }
        else:
            # 普通列处理
            stats[col] = {
                'unique_count': df[col].nunique(),
                'sample_values': df[col].value_counts().head(3).index.tolist(),
                'min_value': df[col].min(),
                'max_value': df[col].max()
            }
    return pd.DataFrame.from_dict(stats, orient='index',
                                 columns=['unique_count', 'sample_values', 'min_value', 'max_value']).rename_axis(name)

# 生成报告
user_stats = get_column_stats(users, 'users')
item_stats = get_column_stats(items, 'items')

print(pd.concat([user_stats, item_stats], axis=0))
