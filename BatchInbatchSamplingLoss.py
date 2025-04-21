import tensorflow as tf
from tensorflow.keras import layers, losses


class BatchInbatchSamplingLoss(layers.Layer):
    """基于批内负采样的召回任务损失层"""

    def __init__(self, temperature=None, name='batch_inbatch_loss', ** kwargs):
        super().__init__(name=name, ** kwargs)
        self.temperature = temperature
        self.loss_fn = losses.CategoricalCrossentropy(from_logits=True, reduction='none')

    def call(self, inputs, training=None):
        """
        输入参数:
        - query_embeddings: [batch_size, dim] 用户嵌入向量
        - candidate_embeddings: [batch_size, dim] 物品嵌入向量
        - sample_weight: [batch_size] 样本权重 (可选)
        - candidate_sampling_probability: [batch_size] 物品采样概率 (可选)
        """
        # 解包输入参数
        query_embeddings, candidate_embeddings = inputs[0], inputs[1]
        sample_weight = inputs[2] if len(inputs) > 2 else None
        candidate_sampling_prob = inputs[3] if len(inputs) > 3 else None

        # 计算批内得分矩阵 [batch_size, batch_size]
        scores = tf.matmul(query_embeddings, candidate_embeddings, transpose_b=True)

        # 应用温度缩放
        if self.temperature is not None:
            scores = scores / self.temperature

        # 采样概率修正 (Sampled Softmax Adjustment)
        if candidate_sampling_prob is not None:
            # scores = logits - log(sampling_prob)
            scores = scores - tf.math.log(candidate_sampling_prob[tf.newaxis, :])

        # 生成标签矩阵 [batch_size, batch_size]
        batch_size = tf.shape(scores)[0]
        labels = tf.eye(batch_size, batch_size)  # 对角线为1的矩阵

        # 计算损失
        loss_per_sample = self.loss_fn(y_true=labels, y_pred=scores)

        # 应用样本权重
        if sample_weight is not None:
            loss_per_sample = loss_per_sample * sample_weight

        return tf.reduce_mean(loss_per_sample)