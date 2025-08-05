import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# 1. Synthetic Contextual Bandit Data with Best-Arm Labels
def generate_supervised_cmab_data(n_samples=1000, n_features=10, n_arms=5, seed=42):
    torch.manual_seed(seed)
    X = torch.randn(n_samples, n_features)
    W = torch.randn(n_arms, n_features)
    logits = X @ W.T
    labels = torch.argmax(logits, dim=1)  # best-arm labels
    return X, labels

# 2. Tabular Transformer Encoder (Same as Before)
class TabularEmbedding(nn.Module):
    def __init__(self, input_dim, d_model):
        super().__init__()
        self.linear = nn.Linear(input_dim, d_model)
    def forward(self, x):
        return self.linear(x).unsqueeze(1)  # [B, 1, D]

class TransformerBackbone(nn.Module):
    def __init__(self, d_model=64):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
    def forward(self, x):
        return self.encoder(x)  # [B, 1, D]

# 3. Policy Network (Classifier over Arms)
class ArmClassifier(nn.Module):
    def __init__(self, d_model, n_arms):
        super().__init__()
        self.out = nn.Linear(d_model, n_arms)
    def forward(self, x):
        return self.out(x.squeeze(1))  # [B, D] -> [B, A]

# 4. Train in Supervised Fashion
def train_cmab_supervised(X, y, n_arms=5, d_model=64, epochs=10, batch_size=64):
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    embed = TabularEmbedding(X.shape[1], d_model)
    backbone = TransformerBackbone(d_model)
    classifier = ArmClassifier(d_model, n_arms)

    model = nn.Sequential(embed, backbone, classifier)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        total_loss, correct = 0.0, 0
        for xb, yb in loader:
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (logits.argmax(dim=1) == yb).sum().item()

        acc = correct / len(X)
        print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}, Accuracy = {acc:.4f}")

# Run
X, y = generate_supervised_cmab_data()
train_cmab_supervised(X, y)

