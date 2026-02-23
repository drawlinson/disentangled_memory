from dataclasses import dataclass

import torch

from agent.q_learning_agent import QLearningAgent, QLearningAgentConfig
from disentangled.sparse_distributed_memory import SparseDistributedMemory
import logging

logger = logging.getLogger(__name__)

@dataclass
class DisentangledAgentConfig(QLearningAgentConfig):

    memory_size:int = 100
    learning_rate_values:float = 0.1
    sparsity:int = 16


class DisentangledAgent(QLearningAgent):
    """
    Uses a SparseDistributedMemory to learn returns.
    """

    def __init__(self, config:DisentangledAgentConfig):
        super().__init__(config)

    def _create_model(self, observation):
        super()._create_model(observation)

        # Create memory objects
        logger.info("Creating disentangled memory...")
        input_size = self.get_model_input_size()
        self.memory = SparseDistributedMemory(
            input_size = input_size, 
            memory_size = self.config.memory_size,
            sparsity = self.config.sparsity,
            value_size=1, # ie return, Q-learning
            learning_rate=self.config.learning_rate_values,
        )

    def predict_returns(self, state_action):
        returns = self.memory.get_value(state_action).squeeze(1)  # [B,V] = [B,1] --> [B]
        return returns

    def learn_returns(self, state_action, returns):
        if not self.is_mode_training():
            return

        self.memory.set_value(state_action, returns)

    def _create_model_optimizer(self):
        return None  # Override as no optimizer needed

    def _add_log_values(self):
        super()._add_log_values()

        min_v, max_v = torch.aminmax(self.memory.mem_value)
        mean_v = torch.mean(self.memory.mem_value)
        self.log_scalar("V_min", min_v)
        self.log_scalar("V_mean", mean_v)
        self.log_scalar("V_max", max_v)
