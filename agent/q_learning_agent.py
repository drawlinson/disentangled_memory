from dataclasses import dataclass

import numpy as np
import torch

from agent.episodic_agent import EpisodicAgent, EpisodicAgentConfig
from model.sparse.random_projection import RandomProjection
from model.sparse.recurrent_encoder import RecurrentEncoder
from util.device import get_device
from util.sparse import one_hot_bit_sequences, nonzero_bit_sequences
import logging

logger = logging.getLogger(__name__)

@dataclass
class QLearningAgentConfig(EpisodicAgentConfig):

    encoding_method:str = None
    encoded_observation_size:int = 50
    encoded_action_size:int = 20

    discount:float = 0.9
    epsilon_greedy:float = 0.3


class QLearningAgent(EpisodicAgent):
    """
    Not a DQN because not a Deep ANN, even if it is an approximation to the full Q-table.
    This class exists to centralize as much code as possible leaving only the details of
    the actual model to derived Agent classes.
    """

    ENCODING_METHOD_FLATTEN = "flatten"
    ENCODING_METHOD_RECURRENT = "recurrent"

    def __init__(self, config:QLearningAgentConfig):
        super().__init__(config)

    def _create_model(self, observation):
        super()._create_model(observation)

        self.context_window_shape = self.get_observation_shape(observation)

        # Create encoders
        self.recurrent_observation_encoder = RecurrentEncoder(
            input_size = self.config.token_embedding_size,
            recurrent_size = self.config.encoded_observation_size,
        )
        self.action_encoder = RandomProjection(
            input_dim = self.num_actions, 
            output_dim = self.config.encoded_action_size, 
            requires_grad=False, 
        )

        # Enumerate and pre-encode all possible actions
        self.actions = self.create_actions(
            num_actions = self.num_actions,
            one_hot = True,
        )  # [possible actions, A]
        self.actions_encoded = self.action_encoder.transform(self.actions)  # [possible actions, A]

        # ... next create the actual model (in derived class)

    def create_actions(self, num_actions:int, one_hot:bool=True, device=None):
        """
        Enumerate all *possible* action combinations. This is done to avoid unnecessary 
        generation of meaningless or useless action combinations.
        """
        if device is None:
            device = get_device()
        if one_hot:
            return one_hot_bit_sequences(N=num_actions, device=device)
        return nonzero_bit_sequences(N=num_actions, device=device)

    def _step_result(
        self,
        observation,
        rewards,
        info,
        reset_mask,
    ):
        """
        Handle the result of the step, including rewards, losses etc.        
        """
        super()._step_result(observation, rewards, info, reset_mask)  # Just some logging
        t = self.context_window_next.get_tensor()
        self.encoded_obs_next = self.encode_observations(t)

        encoded_actions = self.encode_actions(self.a)
        state_action = self.concat_vectors([self.encoded_obs, encoded_actions])

        optimal_returns, optimal_actions = self.get_returns(self.encoded_obs_next)
        optimal_returns = optimal_returns.unsqueeze(1)  # [B] -->  [B,1]
        r = torch.tensor(
            rewards,
            dtype = torch.float
        ).unsqueeze(1)  # Should be 2d, [B,V] so make it [B,1]

        returns = r + (self.config.discount * optimal_returns)
        self.memory.set_value(state_action, returns)

    def get_model_input_size(self) -> int:
        if self.config.encoding_method == QLearningAgent.ENCODING_METHOD_FLATTEN:
            context_window_size = np.prod(self.context_window_shape)
            return context_window_size + self.config.encoded_action_size
        elif self.config.encoding_method == QLearningAgent.ENCODING_METHOD_RECURRENT:
            return self.config.encoded_observation_size + self.config.encoded_action_size
        else:
            raise ValueError("Observation encoding method not recognized.")

    def encode_observations(self, observation):
        if self.config.encoding_method == QLearningAgent.ENCODING_METHOD_FLATTEN:
            batch_size = observation.shape[0]
            context_window_1d = observation.view([
                batch_size,
                -1,
            ])
            #print("Using flatten")
        elif self.config.encoding_method == QLearningAgent.ENCODING_METHOD_RECURRENT:
            context_window_1d = self.recurrent_observation_encoder(observation)
            #print(f"encode_observations(): output 1d shape:{context_window_1d.shape}")
        else:
            raise ValueError("Observation encoding method not recognized.")
        return context_window_1d

    def encode_actions(self, actions):
        encoded_actions = self.action_encoder.transform(actions)
        return encoded_actions

    def concat_vectors(self, tensors:list[torch.Tensor]):
        output = torch.cat(tensors, dim=1)
        return output

    def get_random_action_combinations(self):
        num_action_combinations = self.actions.shape[0]
        random_indices = torch.randint(high=num_action_combinations, size=(self.config.batch_size,))  # [B,]
        actions = self.actions[random_indices]  # B, A
        return actions

    def get_returns(self, encoded_observations):
        """
        Returns two tensors, being the returns[B] and optimal actions[B,A] 
        to achieve those returns.
        """
        encoded_observations = encoded_observations.detach()  # no grads
        batch_size = encoded_observations.shape[0]
        num_actions = self.actions.shape[0]
        action_size = self.actions.shape[1]

        optimal_returns = torch.empty(batch_size).fill_(-torch.inf)
        optimal_actions = torch.zeros((batch_size, action_size))
        self.action_indices = torch.zeros(batch_size)

        # Find the best known action for each sample
        for action_index in range(num_actions):  # TODO could vectorize
            ea = self.actions_encoded[action_index,:].unsqueeze(0).expand(batch_size, -1)
            state_action = self.concat_vectors([encoded_observations, ea])

            returns = self.predict_returns(state_action)
            indices = torch.where(returns >= optimal_returns)[0]
            optimal_actions[indices] = self.actions[action_index]
            optimal_returns[indices] = returns[indices]

        return optimal_returns, optimal_actions

    def _step_get_actions(self, observation):
        super()._step_get_actions(observation)  # update obs history
        t = self.context_window.get_tensor()
        self.encoded_obs = self.encode_observations(t)
        optimal_returns, optimal_actions = self.get_returns(self.encoded_obs)

        # epsilon greedy exploration        
        B = t.shape[0]
        r = torch.rand(B)
        random_sample_indices = r < self.config.epsilon_greedy  # [B] bool; eps-greedy is P(random action)
        random_actions = self.get_random_action_combinations()

        actions = optimal_actions.clone()
        actions[random_sample_indices] = random_actions[random_sample_indices]
        self.a = actions
        return self.a

    def _step_result(
        self,
        observation,
        rewards,
        info,
        reset_mask,
    ):
        """
        Handle the result of the step, including rewards, losses etc.        
        """
        super()._step_result(observation, rewards, info, reset_mask)
        t = self.context_window_next.get_tensor()
        self.encoded_obs_next = self.encode_observations(t)

        encoded_actions = self.encode_actions(self.a)
        state_action = self.concat_vectors([self.encoded_obs, encoded_actions])

        optimal_returns, optimal_actions = self.get_returns(self.encoded_obs_next)
        optimal_returns = optimal_returns.unsqueeze(1)  # [B] -->  [B,1]
        r = torch.tensor(
            rewards,
            dtype = torch.float
        ).unsqueeze(1)  # Should be 2d, [B,V] so make it [B,1]

        returns = r + (self.config.discount * optimal_returns)
        self.learn_returns(state_action, returns)

    def predict_returns(self, state_action):
        raise NotImplementedError

    def learn_returns(self, state_action, returns):
        raise NotImplementedError
