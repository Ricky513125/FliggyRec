


# 初始化
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RecommenderModel(user_feat_sizes, item_feat_sizes, label_vocab_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 数据加载（带负采样）
dataset = RecDatasetWithNegative(users, items, interactions, neg_ratio=3)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# 训练循环
for epoch in range(10):
    avg_loss = train_epoch(model, dataloader, optimizer)
    print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")

    # 保存checkpoint（包含权重配置）
    if (epoch + 1) % 2 == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'action_weights': ACTION_WEIGHTS,  # 保存权重配置
            'label_vocab': all_labels
        }, f'checkpoint_epoch{epoch + 1}.pt')