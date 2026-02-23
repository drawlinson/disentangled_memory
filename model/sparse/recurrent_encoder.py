import torch
import torch.nn as nn
import math

class RecurrentEncoder(nn.Module):
    def __init__(self, input_size: int, recurrent_size: int):
        super().__init__()

        self.input_size = input_size
        self.recurrent_size = recurrent_size

        W_input = torch.randn(input_size, recurrent_size) / math.sqrt(input_size)
        W_recurrent = torch.randn(recurrent_size, recurrent_size) / math.sqrt(recurrent_size)

        self.register_buffer("W_input", W_input)
        self.register_buffer("W_recurrent", W_recurrent)

    def forward(self, x):
        """
        x: [B, T, H, W, E]
        returns: [B, recurrent_size]
        """

        # Make the input a sequence of pixel embeddings, 
        # which will be encoded serially (recurrently) per batch sample
        B, T, H, W, E = x.shape
        R = T*H*W
        x_3d = x.reshape(B, R, E)

        r = torch.zeros(B, self.recurrent_size, device=x.device)
        k = int(self.recurrent_size * 0.4)

        for i in range(R):
            input = x_3d[:, i]

            z = r @ self.W_recurrent
            values, indices = torch.topk(z, k, dim=1)
            s = torch.zeros_like(z).scatter_(1, indices, values)

            # squashing
            r = torch.tanh(
                input @ self.W_input + s
            )

            # stateless normalization (no learning) (last dim only)
            r = r / (r.norm(dim=-1, keepdim=True) + 1e-6)

        return r
