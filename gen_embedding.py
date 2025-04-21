import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Dense, Concatenate
from tensorflow.keras.models import Model

# 假设您的数据维度
NUM_USERS = len(users['user_id'].unique())
NUM_ITEMS = len(items['item_id'].unique())
EMBEDDING_DIM = 64

# 用户塔
user_input = Input(shape=(1,), name='user_input')
user_embedding = Embedding(NUM_USERS, EMBEDDING_DIM)(user_input)
user_embedding = tf.squeeze(user_embedding, axis=1)

# 物品塔
item_input = Input(shape=(1,), name='item_input')
item_embedding = Embedding(NUM_ITEMS, EMBEDDING_DIM)(item_input)
item_embedding = tf.squeeze(item_embedding, axis=1)

# 计算点积得分
dot_product = tf.reduce_sum(user_embedding * item_embedding, axis=1, keepdims=True)
model = Model(inputs=[user_input, item_input], outputs=dot_product)

# 编译模型
model.compile(optimizer='adam', loss='mse')  # 先用简单MSE训练

# 准备训练数据（示例）
train_user_ids = history['user_id'].values
train_item_ids = history['item_id'].values
labels = np.ones(len(history))  # 正样本标签为1

# 训练生成初步embedding
model.fit(
    x=[train_user_ids, train_item_ids],
    y=labels,
    batch_size=1024,
    epochs=10
)

# 提取embedding层
user_embedding_model = Model(
    inputs=model.get_layer('user_input').input,
    outputs=model.get_layer('user_embedding').output
)
item_embedding_model = Model(
    inputs=model.get_layer('item_input').input,
    outputs=model.get_layer('item_embedding').output
)