import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# --- 1. Generate synthetic contextual bandit data
def generate_cmab_data(n_samples=1000, n_features=10, n_arms=5, seed=42):
    torch.manual_seed(seed)
    X = torch.randn(n_samples, n_features)
    W = torch.randn(n_arms, n_features)
    logits = X @ W.T
    probs = torch.softmax(logits, dim=1)
    actions = torch.multinomial(probs, num_samples=1).squeeze()
    rewards = logits.gather(1, actions.unsqueeze(1)).squeeze() + 0.1 * torch.randn(n_samples)
    return X, actions, rewards


# --- 2. Feature-as-token Embedding
class PerFeatureEmbedding(nn.Module):
    def __init__(self, n_features, d_model):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Linear(1, d_model) for _ in range(n_features)])

    def forward(self, x):
        tokens = [emb(x[:, i:i+1]) for i, emb in enumerate(self.embeddings)]
        return torch.stack(tokens, dim=1)  # [B, F, D]


# --- 3. Transformer Backbone
class TransformerBackbone(nn.Module):
    def __init__(self, d_model=64, nhead=4, num_layers=1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, x):
        return self.encoder(x)


# --- 4. Heads
class PolicyHead(nn.Module):
    def __init__(self, d_model, n_arms):
        super().__init__()
        self.out = nn.Linear(d_model, n_arms)

    def forward(self, x):
        return self.out(x.mean(dim=1))  # mean pool over tokens


class CriticHead(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.out = nn.Linear(d_model, 1)

    def forward(self, x):
        return self.out(x.mean(dim=1)).squeeze()


# --- 5. Combined Actor-Critic Model
class ActorCriticCMAB(nn.Module):
    def __init__(self, n_features, n_arms, d_model=64):
        super().__init__()
        self.embedding = PerFeatureEmbedding(n_features, d_model)
        self.backbone = TransformerBackbone(d_model)
        self.policy_head = PolicyHead(d_model, n_arms)
        self.critic_head = CriticHead(d_model)

    def forward(self, x):
        x = self.embedding(x)
        h = self.backbone(x)
        return self.policy_head(h), self.critic_head(h)


# --- 6. Training Loop
def train(model, X, A, R, epochs=10, batch_size=64):
    loader = DataLoader(TensorDataset(X, A, R), batch_size=batch_size, shuffle=True)
    opt_policy = torch.optim.Adam(list(model.embedding.parameters()) +
                                  list(model.backbone.parameters()) +
                                  list(model.policy_head.parameters()), lr=1e-3)
    opt_critic = torch.optim.Adam(model.critic_head.parameters(), lr=1e-3)

    for epoch in range(epochs):
        total_loss = 0
        for x_batch, a_batch, r_batch in loader:
            logits, values = model(x_batch)
            critic_loss = F.mse_loss(values, r_batch)

            opt_critic.zero_grad()
            critic_loss.backward(retain_graph=True)
            opt_critic.step()

            with torch.no_grad():
                advantage = r_batch - values.detach()

            log_probs = F.log_softmax(logits, dim=-1)
            logp = log_probs.gather(1, a_batch.unsqueeze(1)).squeeze()
            policy_loss = -torch.mean(logp * advantage)

            opt_policy.zero_grad()
            policy_loss.backward()
            opt_policy.step()

            total_loss += policy_loss.item() + critic_loss.item()
        print(f"Epoch {epoch+1}: Total Loss = {total_loss:.4f}")


# --- 7. Inference
def select_action(model, x_context):
    model.eval()
    with torch.no_grad():
        logits, _ = model(x_context)
        probs = F.softmax(logits, dim=-1)
        return torch.argmax(probs, dim=-1), probs


# --- 8. Run full training + inference
if __name__ == "__main__":
    X, A, R = generate_cmab_data()
    model = ActorCriticCMAB(n_features=X.shape[1], n_arms=5)
    train(model, X, A, R)

    x_test = torch.randn(4, X.shape[1])
    chosen_arm, arm_probs = select_action(model, x_test)
    print("Selected arms:", chosen_arm)
    print("Arm probabilities:", arm_probs)

