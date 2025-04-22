import pandas as pd

# 行为权重配置（可根据业务调整）
ACTION_WEIGHTS = {
    'clk': 1.0,  # 点击
    'fav': 2.0,  # 收藏
    'cart': 3.0, # 加购
    'pay': 4.0   # 支付
}

# 读取原始行为数据
history = pd.read_csv('data/train_regular.csv')

# 生成带权重的交互记录
interactions = history.groupby(['user_id', 'item_id'])['action_id'].apply(
    lambda x: sum(ACTION_WEIGHTS[a] for a in x)
).reset_index(name='weight')

# 归一化权重到[0,1]范围（可选）
interactions['label'] = interactions['weight'] / max(ACTION_WEIGHTS.values())

interactions.to_csv('data/interactions.csv', index=False)
print("Succefully saved interactions")
print(interactions.head(10))