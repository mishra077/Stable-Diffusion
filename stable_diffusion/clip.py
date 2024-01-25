import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention



class CLIPEmbedding(nn.Module):

    def __init__(self, vocab_sz: int, vocab_dim: int, max_seq_len: int):
        super.__init__()

        self.token_emebedding = nn.Embedding(vocab_sz, vocab_sz)
        self.positional_embedding = nn.Parameter(torch.zeros(max_seq_len, vocab_dim))

    def forward(self, tokens):

        # (Batch, Seq_len) -> (Batch, seq_len, vocab_dim)
        x = self.token_emebedding(tokens)

        # Add postional embeddings
        x += self.positional_embedding

        return x
    
class CLIPLayer(nn.Module):

    def __init__(self, n_heads: int, vocab_dim: int):
        super.__init__()

        self.layernorm_1 = nn.LayerNorm(vocab_dim)
        self.attention = SelfAttention(n_heads, vocab_dim)
        self.layernorm_2 = nn.LayerNorm(vocab_dim)
        self.linear_1 = nn.Linear(vocab_dim, 4 * vocab_dim)
        self.linear_2 = nn.Linear(4 * vocab_dim, vocab_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # (Batch, seq_len, vocab_dim)

        residue = x

        # SELF ATTENTION

        x = self.layernorm_1(x)
        x = self.attention(x, casual_mask = True)
        x += residue # residual connection

        # FEED FORWARD LAYER

        residue = x

        x = self.layernorm_2(x)
        x = self.linear_1(x)

        x = x * torch.sigmoid(1.702 * x) # QuickGELU activation function

        x = self.linear_2(x)

        x += residue

        return x


class CLIP(nn.Module):

    def __init__(self):

        self.embedding = CLIPEmbedding(49408, 768, 77)

        self.layers = nn.Module([
            CLIPLayer(12, 768) for _ in range(12)
        ])

        self.layernorm = nn.LayerNorm(768)

    def forward(self, tokenx: torch.LongTensor) -> torch.FloatTensor:

        tokens = tokens.type(torch.long)

        state = self.embedding(tokens)

        for layer in self.layers:
            state = layer(state)

        output = self.layernorm(state)

        return output


