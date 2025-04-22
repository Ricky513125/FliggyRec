import pandas as pd
import numpy as np

# get_Data
users = pd.read_csv("data/user_profile.csv")
items = pd.read_csv("data/item_profile.csv")

# 处理多标签列
def parse_labels(label_str):
    if label_str == '-1':
        # 没有标签

    return [int(x) for x in label_str.split(';')]

