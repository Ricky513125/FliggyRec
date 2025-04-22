import torch
import torch.nn as nn
class DynamicWeightedAverage(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        # 权重生成网络
        self.weight_net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
            nn.Softmax(dim=0)  # 保证权重和为1
        )

    def forward(self, embeddings, lengths):
        """
        Args:
            embeddings: (total_labels, embed_dim) 展平后的所有标签embedding
            lengths: (batch_size,) 每个样本的标签数量
        Returns:
            pooled: (batch_size, embed_dim)
        """
        # 生成权重 (total_labels, 1)
        weights = self.weight_net(embeddings)

        # 分段求和
        pooled = []
        start = 0
        for l in lengths:
            end = start + l
            segment = embeddings[start:end]
            segment_weights = weights[start:end]
            pooled.append(torch.sum(segment * segment_weights, dim=0))
            start = end

        return torch.stack(pooled)