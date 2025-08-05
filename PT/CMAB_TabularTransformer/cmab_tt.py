import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# 1. Synthetic CMAB data
def generate_cmab_data(n_samples=1000, n_features=10, n_arms=5, seed=42):
    torch.manual_seed(seed)
    X = torch.randn(n_samples, n_features)
    W = torch.randn(n_arms, n_features)
    logits = X @ W.T
    probs = torch.softmax(logits, dim=1)
    actions = torch.multinomial(probs, num_samples=1).squeeze()
    rewards = logits.gather(1, actions.unsqueeze(1)).squeeze() + 0.1 * torch.randn(n_samples)
    return X, actions, rewards


# 2. Tabular Transformer Components
class TabularEmbedding(nn.Module):
    def __init__(self, input_dim, d_model):
        super().__init__()
        self.linear = nn.Linear(input_dim, d_model)

    def forward(self, x):
        return self.linear(x).unsqueeze(1)


class TransformerBackbone(nn.Module):
    def __init__(self, d_model=64, nhead=4, num_layers=1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, x):
        return self.encoder(x)


class PolicyHead(nn.Module):
    def __init__(self, d_model, n_arms):
        super().__init__()
        self.out = nn.Linear(d_model, n_arms)

    def forward(self, x):
        return self.out(x.squeeze(1))


class CriticHead(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.out = nn.Linear(d_model, 1)

    def forward(self, x):
        return self.out(x.squeeze(1)).squeeze(-1)


# 3. Combined Actor-Critic CMAB model
class ActorCriticCMAB(nn.Module):
    def __init__(self, input_dim, n_arms, d_model=64):
        super().__init__()
        self.embed = TabularEmbedding(input_dim, d_model)
        self.backbone = TransformerBackbone(d_model)
        self.policy = PolicyHead(d_model, n_arms)
        self.critic = CriticHead(d_model)

    def forward(self, x):
        x = self.embed(x)
        feats = self.backbone(x)
        logits = self.policy(feats)
        values = self.critic(feats)
        return logits, values


# 4. Training loop
def train_model(model, X, A, R, epochs=10, batch_size=64, lr=1e-3):
    dataset = TensorDataset(X, A, R)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer_policy = torch.optim.Adam(
        list(model.embed.parameters()) +
        list(model.backbone.parameters()) +
        list(model.policy.parameters()), lr=lr
    )
    optimizer_critic = torch.optim.Adam(model.critic.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0
        for x_batch, a_batch, r_batch in loader:
            logits, values = model(x_batch)

            # Critic update
            critic_loss = F.mse_loss(values, r_batch)
            optimizer_critic.zero_grad()
            critic_loss.backward(retain_graph=True)
            optimizer_critic.step()

            # Policy update
            with torch.no_grad():
                advantage = r_batch - values.detach()

            log_probs = F.log_softmax(logits, dim=-1)
            logp = log_probs.gather(1, a_batch.unsqueeze(1)).squeeze()
            policy_loss = -(logp * advantage).mean()

            optimizer_policy.zero_grad()
            policy_loss.backward()
            optimizer_policy.step()

            total_loss += policy_loss.item() + critic_loss.item()

        print(f"Epoch {epoch+1:02d} | Total Loss: {total_loss:.4f}")


# 5. Run example
if __name__ == "__main__":
    X, A, R = generate_cmab_data()
    model = ActorCriticCMAB(input_dim=X.shape[1], n_arms=5)
    train_model(model, X, A, R)

    # inference 
    def select_action(model, context, exploit=False):
        model.eval()
        with torch.no_grad():
            logits, _ = model(context.unsqueeze(0))  # [1, d]
            probs = F.softmax(logits, dim=-1)
            if exploit:
                return torch.argmax(probs, dim=-1).item()
            else:
                # explore
                return torch.multinomial(probs, num_samples=1).item()
    contexts = torch.randn(10, X.shape[1])  # 10 new test contexts
    actions = [select_action(model, c, exploit=True) for c in contexts]
    print("Chosen actions:", actions)

