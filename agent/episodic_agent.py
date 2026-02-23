import torch
from dataclasses import dataclass
from model.embedding import TokenIdEmbedding, TokenIdEmbeddingConfig
from model.context_window import ContextWindow, ContextWindowConfig
from environment.token_env import TokenEnvironment
from agent.base_agent import BaseAgent, BaseAgentConfig
import logging

logger = logging.getLogger(__name__)


@dataclass
class EpisodicAgentConfig(BaseAgentConfig):

    token_embedding_size: int = 0
    history_size: int = 0


class EpisodicAgent(BaseAgent):
    """
    Adds support for partial observability via the context window concept.
    Maintains a moving window of previous observations.
    """

    def __init__(self, config:EpisodicAgentConfig):
        super().__init__(config)
        assert(isinstance(self._env, TokenEnvironment))

    def _create_model(self, observation):
        self._create_embeddings(observation)

    def _create_embeddings(self, observation):
        num_tokens = self._env.get_num_tokens()
        self._create_token_embedding(num_tokens)
        observation_shape = self.get_observation_shape(observation)
        self._create_context_window(
            observation_shape=observation_shape,
        )

    def _create_token_embedding(self, num_tokens):
        token_embedding_config = TokenIdEmbeddingConfig(
            embedding_size = self.config.token_embedding_size,
            num_tokens = num_tokens,
        )
        self.token_embedding = TokenIdEmbedding(token_embedding_config)

    def _create_context_window(self, observation_shape):
        config = ContextWindowConfig(
            batch_size = self.config.batch_size,  # B
            history_size = observation_shape[0],  # T
            height = observation_shape[1],  # H
            width = observation_shape[2],  # W
            embedding_size = observation_shape[3],  # E
        )
        self.context_window = ContextWindow(config)
        logger.info(f"Obs. History shape: {self.context_window.history.shape}")

    def get_observation_shape(self, observation) -> list[int]:
        """
        Calculates the shape of a moving window of observations from a single observation.
        Does not include batch dimension.
        
        :param observation: This is a single observation, not a minibatch of sample observations. There's no batch dim.
        :return: The shape of a single observation with history as a list of int.
        :rtype: list[int]
        """
        k = self._env.get_token_key()
        observation_array = observation[k]
        observation_array_shape_list = list(observation_array.shape)
        #print(f"observation_shape = {observation_array_shape_list}")
        num_dim = len(observation_array_shape_list)
        if num_dim == 0:
            observation_shape = [
                self.config.history_size,  # T
                1,  # H
                1,  # W
                self.config.token_embedding_size, # E
            ]
        elif num_dim == 1:
            observation_shape = [
                self.config.history_size,  # T
                1,  # H
                observation_array_shape_list[0],  # W
                self.config.token_embedding_size, # E
            ]
        elif num_dim == 2:
            observation_shape = [
                self.config.history_size,  # T
                observation_array_shape_list[0],  # H
                observation_array_shape_list[1],  # W
                self.config.token_embedding_size, # E
            ]
        else:
            raise ValueError("Observation can't be more than 2D.")
        return observation_shape
    
    def reset(self):
        """
        Reset all environments and store reset obs.
        Reset observation history for all environments too.
        """
        super().reset()
        self.context_window.reset()  # Will see self.obs on first step()

    def _step_get_actions(self, observation):
        # Update observation history inc. preprocessing, with grads for embedding training.
        self._update_history_with_observation(
            context_window = self.context_window,
            observation = observation, 
        )

    def _step_result(
        self,
        obs,
        rewards,
        info,
        reset_mask,
    ):
        super()._step_result(obs, rewards, info, reset_mask)

        # Create a copy of history with the new obs.
        # The self.context_window_next is created without grads and then discarded.
        with torch.no_grad():
            self.context_window_next = self.context_window.clone()  # no grad anyways
            self._update_history_with_observation(
                context_window = self.context_window_next,
                observation = self.obs,  # new obs.
            )        

    def _step_reset_envs(self, reset_mask):
        super()._step_reset_envs(reset_mask)
        self.context_window.reset(reset_mask)  # will get new obs next step

    def _update_history_with_observation(self, context_window, observation):
        preprocessed_observation = self._preprocess_observation(observation)

        #logger.debug(f"History (before):\n{self.context_window.history}")
        context_window.update(
            obs = preprocessed_observation, 
            reset_mask=None,
        )
        #logger.debug(f"History (after):\n{self.context_window.history}")

    def _preprocess_observation(self, observation):
        k = self._env.get_token_key()
        observation_array = observation[k]
        obs_tokens = torch.tensor(observation_array).to(self.device).detach()
        obs_token_embedding = self.token_embedding(obs_tokens)

        observation_array_shape_list = list(observation_array.shape)
        num_dim = len(observation_array_shape_list)
        if num_dim == 1:  # [B,E]
            obs_token_embedding_4d = obs_token_embedding[:, None, None, :]
        elif num_dim == 2:  # [B,W,E]
            obs_token_embedding_4d = obs_token_embedding[:, None, :, :]
        elif num_dim == 3:  # [B,H,W,E]
            obs_token_embedding_4d = obs_token_embedding[:, :, :, :]
        else:
            raise ValueError("Observation can't be more than 2D.")

        return obs_token_embedding_4d
