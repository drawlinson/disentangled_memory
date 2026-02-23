from __future__ import annotations
from dataclasses import dataclass
from copy import deepcopy
import torch

@dataclass
class ContextWindowConfig:
    batch_size: int = 0
    history_size: int = 0
    width: int = 0
    height: int = 0
    embedding_size: int = 0

class ContextWindow:
    """
    Maintains a rolling observation history:
        history shape = [B, T, Y, X, E]

    - new obs have shape [B, Y, X, E]
    - reset_mask has shape [B] / bool
    """

    def __init__(self, config, device=None, dtype=torch.float32):
        self.config = config
        self.device = device
        self.dtype = dtype
        self.create_history()

    def create_history(self):
        # Initialize blank history
        self.history = torch.zeros(
            (
                self.config.batch_size, 
                self.config.history_size,
                self.config.width,
                self.config.height, 
                self.config.embedding_size
            ),
            dtype=self.dtype,
            device=self.device,
        )
        #print(f"History shape={self.history.shape}")

    @torch.no_grad()
    def clone(self) -> ContextWindow:
        temp = self.history
        self.history = None
        history_copy = deepcopy(self)
        self.history = temp
        history_copy.history = temp.detach().clone()
        return history_copy

    @torch.no_grad()
    def reset(self, mask = None):
        """
        mask: Bool tensor [B]
        Zero out entire history for envs that just terminated.
        """
        if mask is None:
            mask = torch.ones(self.config.batch_size, dtype=torch.bool)
        if mask.any():
            self.history[mask] = 0

    def update(self, obs, reset_mask=None):
        """
        new_obs: [B, Y, X, E]
        reset_mask: optional, Bool [B]
        """
        # 1. First reset terminated environments
        if reset_mask is not None:
            self.reset(reset_mask)

        # 2. Roll history to the left (drop oldest)
        # Equivalent to: history[:, 1:T] -> history[:, 0:T-1]
        # DO NOT KEEP GRADS FROM PREVIOUS FORWARD PASSES
        with torch.no_grad():
            self.history = self.history.detach().clone().roll(shifts=-1, dims=1)

        # 3. Insert new observations at the end
        self.history[:, -1] = obs

    def get_tensor(self):
        """
        Returns full history tensor: [B, T, Y, X, E]
        """
        return self.history