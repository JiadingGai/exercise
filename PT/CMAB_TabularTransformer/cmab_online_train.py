import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

# --- 1. Environment (synthetic CMAB simulator)
class SyntheticCMABEnv:
    def __init__(self, n_features=10, n_arms=5, seed=42):
        torch.manual_seed(seed)
        self.n_features = n_features
        self.n_arms = n_arms
        self.W = torch.randn(n_arms, n_features)  # True reward weights

    def sample_context(self):
        return torch.randn(self.n_features)

    def get_reward(self, context, action):
        expected_reward = torch.matmul(self.W[action], context)
        reward = expected_reward + 0.1 * torch.randn(())  # Add noise
        return reward


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
        return self.out(x.mean(dim=1))


class CriticHead(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.out = nn.Linear(d_model, 1)

    def forward(self, x):
        return self.out(x.mean(dim=1)).squeeze()


# --- 5. Combined Model
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


# --- 6. Online Training with Active Exploration
def train_online(model, env, n_steps=1000):
    opt_policy = torch.optim.Adam(list(model.embedding.parameters()) +
                                  list(model.backbone.parameters()) +
                                  list(model.policy_head.parameters()), lr=1e-3)
    opt_critic = torch.optim.Adam(model.critic_head.parameters(), lr=1e-3)

    for step in range(n_steps):
        context = env.sample_context().unsqueeze(0)  # [1, D]
        logits, value = model(context)

        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        action = dist.sample()

        reward = env.get_reward(context.squeeze(0), action.item())
        advantage = reward - value.item()

        # --- Update Critic ---
        critic_loss = F.mse_loss(value, torch.tensor([reward]))
        opt_critic.zero_grad()
        critic_loss.backward(retain_graph=True)
        opt_critic.step()

        # --- Update Policy ---
        logp = dist.log_prob(action)
        policy_loss = -logp * advantage
        opt_policy.zero_grad()
        policy_loss.backward()
        opt_policy.step()

        if (step + 1) % 100 == 0:
            print(f"Step {step+1}: Reward = {reward:.3f}, Advantage = {advantage:.3f}")


# --- 7. Run Everything
if __name__ == "__main__":
    env = SyntheticCMABEnv()
    model = ActorCriticCMAB(n_features=10, n_arms=5)
    train_online(model, env)

