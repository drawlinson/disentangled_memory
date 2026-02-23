from dataclasses import dataclass

import torch
import torch.nn.functional as F

from agent.q_learning_agent import QLearningAgent, QLearningAgentConfig
from model.dense import DenseModel, DenseModelConfig
import logging

logger = logging.getLogger(__name__)


@dataclass
class EntangledAgentConfig(QLearningAgentConfig):
    model_layers: int = 0
    model_bias: bool = True
    model_nonlinearity: str = "leaky-relu"
    model_input_layer_norm: bool = False
    model_input_dropout: float = 0.0
    model_input_weight_clip: float = 0.0
    model_hidden_size: int = 0
    model_hidden_dropout: float = 1.0


class EntangledAgent(QLearningAgent):
    """
    Simply a 2-layer fully-connected feed-forward ANN.
    """

    def __init__(self, config: EntangledAgentConfig):
        super().__init__(config)
        self._loss = None

    def _create_model(self, observation):
        super()._create_model(observation)

        # Create memory objects
        logger.info("Creating entangled memory (dense MLP model)...")
        input_size = self.get_model_input_size()
        output_size = 1  # predicts returns
        model_name = "dense"
        model_config = DenseModelConfig(
            name=model_name,
            layers=self.config.model_layers,
            bias=self.config.model_bias,
            nonlinearity=self.config.model_nonlinearity,
            input_layer_norm=self.config.model_input_layer_norm,
            input_dropout=self.config.model_input_dropout,
            input_weight_clip=self.config.model_input_weight_clip,
            input_size=input_size,
            hidden_size=self.config.model_hidden_size,
            hidden_dropout=self.config.model_hidden_dropout,
            output_size=output_size,
            output_nonlinearity=None,
        )
        self.model = DenseModel(model_config)
        self.trainable_modules[model_name] = self.model

    def predict_returns(self, state_action):
        with torch.no_grad():
            returns = self.model(state_action).squeeze(1)
            return returns

    def learn_returns(self, state_action, returns):
        if not self.is_mode_training():
            self._optimizer.clear_gradients()
            return

        state_action = state_action.detach()  # don't modify token embedding
        returns_predicted = self.model(state_action).squeeze(1)
        self._loss = F.mse_loss(returns_predicted, returns.squeeze(1))

        if self._loss is None:
            return

        parameters = self._get_model_parameters()
        self._optimizer.optimize(self._loss, parameters)

    def _add_log_values(self):
        super()._add_log_values()
        self.log_scalar("loss", self._loss)
