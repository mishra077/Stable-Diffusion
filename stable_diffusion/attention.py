import torch
from torch import nn
from torch.nn import functinal as F
import math

class SelfAttention(nn.Module):

    def __init__(self, n_heads: int, d_embed: int, in_proj_bias=True, out_proj_bias=True):
        super().__init__()

        # Weights for Query, Key and Value
        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias = in_proj_bias)

        # W0
        self.out_proj = nn.Linear(d_embed, d_embed, bias = out_proj_bias)

        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x: torch.Tensor, casual_mask=False):
        # x: (Batch, Seq_len, Dim)

        input_shape = x.shape

        batch_size, seq_len, d_embed = input_shape

        intermediate_shape = (batch_size, seq_len, self.n_heads, self.d_head)

        q, k, v = self.in_proj(x).chunk(3, dim = -1)

        # Batch, seq_len, dim -> batch, seq_len, heads, dim / heads -> batch,, heads, seq_len, dim / heads
        q = q.view(intermediate_shape).transpose(1, 2)
        k = k.view(intermediate_shape).transpose(1, 2)
        v = v.view(intermediate_shape).transpose(1, 2)


        # (batch, heads, seq_len, dim / heads) @ (batch, heads, dim / heads, seq_len) => (batch, heads, seq_len, seq_len)
        weight = q @ k.transpose(-1, -2)

        if casual_mask:
            # Shape of the mask will be equivalent to the weight shape and upper triangle is of value 1
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            weight.masked_fill_(mask, -torch.inf)

        weight /= math.sqrt(self.d_head)

        weight = F.softmax(weight, dim = -1)

        # (batch, heads, seq_len, seq_len) @ (batch, heads, seq_len, dim / heads) -> (batch, heads, seq_len, dim / heads)
        output = weight @ v

        output = output.transpose(1, 2)

        output = output.reshape(input_shape)

        output = self.out_proj(output)

        return output

