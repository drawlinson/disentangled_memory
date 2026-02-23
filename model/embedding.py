from dataclasses import dataclass
import torch
import torch.nn as nn
import math

@dataclass
class TokenIdEmbeddingConfig:
    embedding_size: int = 0
    num_tokens: int = 0


class TokenIdEmbedding(nn.Module):
    def __init__(self, config):
        """
        num_tokens: number of discrete token IDs (N)
        embedding_dim: dimensionality of each embedding vector (D)
        """
        super().__init__()
        self.config = config
        self.emb = nn.Embedding(self.config.num_tokens, self.config.embedding_size)

    def forward(self, token_ids):
        """
        token_ids: LongTensor of shape [batch] or [batch, sequence]
        returns: FloatTensor [batch, embedding_dim] or [batch, sequence, embedding_dim]
        """
        return self.emb(token_ids)


class SinusoidalEmbedding():
    """
    2D sinusoidal absolute position embedding (PE) for (x,y). 

    Pre-calculate for maximum position ranges and then slice to obtain the complete
    PE for a given observation.

    patch = pe_xy[x0:x0+6, y0:y0+6]    # [6,6,dim]

    If negative coordinates are required, shift to positive integer range.

    # PE was created as [W, H, E]
    PE = PE.permute(1, 0, 2)   # -> [H, W, E]
    """


    @staticmethod
    def sinusoidal_2d(width, height, embedding_size, base=50.0):
        """
        x_range: int, number of x positions
        y_range: int, number of y positions
        dim: embedding dimension
        Returns: [X, Y, dim] embedding tensor
        """
        X = torch.arange(width)
        Y = torch.arange(height)
        
        # create grid
        xx, yy = torch.meshgrid(X, Y, indexing='ij')  # shape [X,Y]
        
        # flatten to vectors
        xx_flat = xx.reshape(-1)
        yy_flat = yy.reshape(-1)
        
        # embed
        pe_flat = SinusoidalEmbedding.sinusoidal(
            xx_flat, 
            yy_flat, 
            embedding_size, 
            base=base
        )  # [X*Y, dim]
        
        # reshape to grid
        pe2d = pe_flat.reshape(width, height, embedding_size)
        pe2d_row_major = pe2d.permute(1, 0, 2) # PE was created as [W, H, E] -> [H, W, E]
        return pe2d_row_major
    
    @staticmethod
    def sinusoidal(x, y, dim, base=10000.0):
        """
        x, y: tensors of shape (...,) giving absolute coordinates (broadcastable).
        dim: total embedding dim (must be divisible by 2 and dim/2 by 2 ie divisible by 4).
        base: Use 10k for long histories of say thousands, 50-100 for smaller e.g. range of 50 possible positions
        Returns: (..., dim)
        """
        assert dim % 2 == 0, "dim must be divisible by 2"
        dim_half = dim // 2                  # half for x, half for y
        assert dim_half % 2 == 0, "dim/2 must be divisible by 2 for sin/cos pairs"
        num_freqs = dim_half // 2            # number of sin/cos pairs per axis

        # frequencies: shape [num_freqs]
        inv_freq = torch.exp(
            -math.log(base) * torch.arange(num_freqs, dtype=torch.float32) / num_freqs
        )  # [num_freqs]

        # x embedding
        x = x.to(dtype=inv_freq.dtype)
        x_exp = x.unsqueeze(-1)              # (..., 1)
        x_proj = x_exp * inv_freq            # (..., num_freqs)
        x_emb = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)  # (..., 2*num_freqs)

        # y embedding
        y = y.to(dtype=inv_freq.dtype)
        y_exp = y.unsqueeze(-1)
        y_proj = y_exp * inv_freq
        y_emb = torch.cat([torch.sin(y_proj), torch.cos(y_proj)], dim=-1)  # (..., 2*num_freqs)

        return torch.cat([x_emb, y_emb], dim=-1)  # (..., dim)

    @staticmethod
    def get_embedding_at_coordinates(positional_embedding, coordinates):
        """
        positional_embedding : [PE_H, PE_W, E]
        coordinates          : [B, H, W, 2] absolute world coords (row, col)

        returns : [B, H, W, E]
        """

        rows = coordinates[..., 0].long()   # [B,H,W]
        cols = coordinates[..., 1].long()   # [B,H,W]

        B, H, W = rows.shape
        PE_H, PE_W, E = positional_embedding.shape

        # 1) Clamp coords (to avoid any out-of-range crash)
        rows = rows.clamp(0, PE_H - 1)
        cols = cols.clamp(0, PE_W - 1)

        # 2) Expand PE to batched form
        pe = positional_embedding.unsqueeze(0).expand(B, -1, -1, -1)  # [B,PE_H,PE_W,E]

        # 3) Build batch index grid
        batch_idx = torch.arange(B, device=pe.device).view(B, 1, 1)

        # 4) Advanced indexing (the key fix)
        sliced = pe[batch_idx, rows, cols]   # → [B,H,W,E]
        return sliced
