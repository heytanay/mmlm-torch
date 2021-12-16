import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, dim, num_heads=3, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, patch):
        B, N, C = patch.shape
        qkv = self.qkv(patch).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        # Get attention by multiplying queries and keys
        attention = torch.mm(q, k.transpose(-2, -1)) * self.scale
        attention = torch.softmax(attention, dim=-1)
        attention = self.attn_drop(attention)

        # Get the output by mutliplying the weighed attention with the values
        output = torch.mm(attention, v).transpose(1, 2).reshape(B, N, C)
        output = self.proj(output)
        output = self.proj_drop(output)

        return output