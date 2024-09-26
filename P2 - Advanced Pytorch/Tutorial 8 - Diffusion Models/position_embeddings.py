# Reference: https://towardsdatascience.com/diffusion-model-from-scratch-in-pytorch-ddpm-9d9760528946
import math
import torch
from torch import nn

class SinusoidalEmbeddings(nn.Module):
    def __init__(self,  time_steps:int, embed_dim: int) -> None:
        super(SinusoidalEmbeddings, self).__init__()
        position = torch.arange(time_steps).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim))
        embeddings = torch.zeros(time_steps, embed_dim, requires_grad=False)
        embeddings[:, 0::2] = torch.sin(position * div)
        embeddings[:, 1::2] = torch.cos(position * div)
        self.embeddings = embeddings
    
    def forward(self, x, t):
        embeds = self.embeddings[t].to(x.device)
        return embeds[:, :, None, None]