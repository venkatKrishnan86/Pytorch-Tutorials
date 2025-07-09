import torch
import math
from torch import nn

# Ref: https://youtu.be/7cMzfkWFWhI?si=ec0KwD-dov2w4cEp

class Block(nn.Module):
    def __init__(self, channels = 512):
        super(Block, self).__init__()
        self.linear = nn.Linear(channels, channels)
        self.norm = nn.LayerNorm(channels)
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.activation(self.norm(self.linear(x)))

class MLP(nn.Module):
    def __init__(self, channels_data=2, layers=5, channels=512, channels_t=512):
        super(MLP, self).__init__()
        self.channels_data = channels_data
        self.layers = layers
        self.channels = channels
        self.channels_t = channels_t
        self.in_projection = nn.Linear(channels_data, channels)
        self.t_projection = nn.Linear(channels_t, channels)
        self.blocks = nn.Sequential(*[
            Block(channels) for _ in range(layers)
        ])
        self.out_projection = nn.Linear(channels, channels_data)

    def generate_t_embedding(self, t, max_t=10000):
        t = t * max_t
        half_dim = self.channels_t // 2
        emb = math.log(max_t) / (half_dim - 1)
        emb = torch.arange(half_dim, device=t.device).float().mul(-emb).exp()
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=1)
        if self.channels_t % 2 == 1:
            emb = nn.functional.pad(emb, (0, 1), mode='constant')
        return emb

    def forward(self, x, t):
        x = self.in_projection(x)
        t = self.generate_t_embedding(t)
        t = self.t_projection(t)
        x = x + t
        x = self.blocks(x)
        return self.out_projection(x)