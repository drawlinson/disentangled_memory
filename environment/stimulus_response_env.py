from typing import Optional
import numpy as np
import pandas as pd
import gymnasium as gym
import logging
from environment.token_env import TokenEnvironment

logger = logging.getLogger(__name__)

class StimulusResponseEnv(gym.Env, TokenEnvironment):
    """
    Creates stimulus-response episodes from a pre-defined CSV dataset 
    of Scenarios. Each scenario represents an encounter with an animal,
    plant, object etc. 
    
    The environment presents an initial observation and the agent
    can discover more information via additional observations, which are 
    performed as "internal" (i.e. perceptual) actions in the environment. 

    There are also "external" actions which directly modify the state of the 
    environment, leading to reward outcomes. Specific logic is defined in the 
    provided dataset CSV file. The dataset file can be replaced with online
    interaction with the generating LLM, but this runs much more slowly.
    """

    def __init__(
            self, 
            dataset_csv_file:str,
            scenario_column:str,
            scenario_include:list[str]|None,
            scenario_exclude:list[str]|None,
            initial_observation_column:str,
            internal_action_columns: list[str],
            external_action_rewards:dict,
            value_true:str = "yes",
            value_false:str = "no",
            max_steps: int|None = 100,
            terminate_min_reward:float|None = None,
            terminate_max_reward:float|None = None,
    ):
        self.dataset_csv_file = dataset_csv_file
        self.scenario_column = scenario_column
        self.initial_observation_column = initial_observation_column
        self.internal_action_columns = internal_action_columns
        self.external_action_rewards = external_action_rewards
        self.value_true = value_true
        self.value_false = value_false
        self.max_steps = max_steps
        self.terminate_min_reward = terminate_min_reward
        self.terminate_max_reward = terminate_max_reward

        self.steps = 0
        self.episode_terminated = False
        self.episode_truncated = False

        self.scenarios = pd.read_csv(dataset_csv_file)
        self._init_observations()
        logger.info(f"Num. scenarios: {self.get_num_scenarios()}")

        observation_space_dict = {}
        self._add_observation_spaces(observation_space_dict)
        self.observation_space = gym.spaces.Dict(observation_space_dict)

        action_space_dict = {}
        self._add_action_space(action_space_dict)
        self.action_space = gym.spaces.Dict(action_space_dict)

        self.set_scenario_filter(
            scenario_include,
            scenario_exclude,
        )

    def get_num_scenarios(self) -> int:
        return len(self.scenarios)

    def get_num_scenarios_filtered(self) -> int:
        return len(self.scenario_indices)

    def set_scenario_filter(
        self, 
        scenario_include:list[str]|None,
        scenario_exclude:list[str]|None,
    ):
        """
        Specify either include or exclude list, not both.
        """
        self.scenario_include = scenario_include
        self.scenario_exclude = scenario_exclude

        if scenario_include is not None and scenario_exclude is not None:
            raise ValueError("Specify scenarios to include or exclude, but not both.")

        self.scenario_indices = []
        num_scenarios = len(self.scenarios)
        for i in range(num_scenarios):
            value =self.get_scenario_value(
                column = self.scenario_column, 
                row = i
            )
            if self.scenario_include is not None:
                if value not in self.scenario_include:
                    continue  # skip if not included explicitly

            if self.scenario_exclude is not None:
                if value in self.scenario_exclude:
                    continue  # skip if excluded explicitly

            self.scenario_indices.append(i)  # include
        logger.info(f"Num. scenarios [filtered]: {self.get_num_scenarios_filtered()}")

    def get_scenario_value(self, column:str, row:int = None):
        if row is None:
            row = self.scenario_index
        return self.scenarios.loc[row, column]
    
    def get_random_scenario(self) -> int:
        num_scenarios = self.get_num_scenarios_filtered()
        n = np.random.randint(0, num_scenarios)
        scenario_index = self.scenario_indices[n]
        return scenario_index

    @staticmethod
    def set_scenario_filter_for_envs(
        envs, 
        scenario_include:list[str]|None,
        scenario_exclude:list[str]|None
    ):
        """
        Allows the scenario filters to be updated to target different scenarios within the dataset.

        :param envs: The result of gym.make_vec i.e. a vectorized set of environments.
        """
        envs.call(
            "set_scenario_filter",
            scenario_include=scenario_include,
            scenario_exclude=scenario_exclude,
        )

    def _init_observations(self):
        """
        Creates data structures to map all observation values to a constant 
        list of unique strings.

        The agent can observe the response to any of the observation actions, 
        plus the initial observation. Therefore we need tokens for each of these.
        """
        all_tokens = set()

        # Find all unique initial observation values
        unique_values = self.scenarios[self.initial_observation_column].unique()
        for value in unique_values:
            column_value = self.create_column_value_encoding(
                  self.initial_observation_column,
                  value,
            )
            all_tokens.add(column_value)

        # Observation actions are questions the agent can ask. 
        # Their values are the possible answers which might be observed.
        for observation_action in self.internal_action_columns:
            unique_values = self.scenarios[observation_action].unique()
            for value in unique_values:
                column_value = self.create_column_value_encoding(
                    observation_action,
                    value,
                )
                all_tokens.add(column_value)

        # Action rewards are observable consequences from actions. 
        # Include all possible unique rewards in the set of tokens
        for action_key, external_action_rewards_dict in self.external_action_rewards.items():
            for conditional_reward in external_action_rewards_dict.values():
                column_value = self.create_column_value_encoding(
                    action_key, #self.observation_rewards,
                    str(float(conditional_reward)),
                )
                all_tokens.add(column_value)

            # A neutral reward of 0 can occur if nothing happens in response to the action
            column_value = self.create_column_value_encoding(
                action_key, #self.observation_rewards,
                str(0.0),
            )
            all_tokens.add(column_value)

        # Build a unique list; create a dict for reverse lookup
        # Ensure list is sorted so that the token-index ordering is constant for all env.
        self.observations = sorted(list(all_tokens))
        logger.info(f"All observable tokens: {all_tokens}")
        logger.info(f"Num. observable tokens: {self.get_num_unique_observations()}")

        self.observation_indices = {}
        for index, observation in enumerate(self.observations):
            self.observation_indices[observation] = index

    def create_column_value_encoding(self, column:str, value:str):
        return column + "=" + value
    
    def get_observation_index(self, observation_key) -> int|None:
        return self.observation_indices.get(observation_key)
        
    def get_num_unique_observations(self) -> int:
        return len(self.observations)

    def get_token_key(self) -> str:
        return "percept"

    def get_num_tokens(self) -> int:
        return self.get_num_unique_observations()
    
    def get_tokens(self) -> list[str]:
        return self.observations()
    
    def _add_observation_spaces(self, observation_space_dict:dict):
        num_values = self.get_num_unique_observations()
        observation_space_dict[self.get_token_key()] = gym.spaces.Box(
            low=np.array(0),
            high=np.array(num_values-1),  # inclusive bound 
            shape=(), 
            dtype=int,
        )  # token index

    def _add_action_space(self, action_space_dict:dict):
        num_actions = self.get_num_actions()
        action_space_dict["actions"] = gym.spaces.MultiBinary(num_actions)

    def get_num_actions(self) -> int:
        num_internal_action_columns = len(self.internal_action_columns)
        num_external_actions = len(self.external_action_rewards.keys())
        num_actions = num_internal_action_columns + num_external_actions
        return num_actions

    def is_internal_action(self, action:int) -> bool:
        num_internal_action_columns = len(self.internal_action_columns)
        return action < num_internal_action_columns

    def is_external_action(self, action:int) -> bool:
        return not self.is_internal_action(action)
    
    def get_action_index(self, actions):
        """
        Resolves an array of selected actions to a single selected one.
        """
        # Pick action indices which are nonzero        
        selected_indices = np.where(actions == 1)[0].tolist()

        # If no nonzero, then pick none
        num_selected_actions = len(selected_indices)
        if num_selected_actions == 0:
            num_actions = actions.shape[0]
            num_selected_actions = num_actions
            n = np.random.randint(0, num_selected_actions)
            action = n
        else:
            # Pick random one of these:
            n = np.random.randint(0, num_selected_actions)
            action = selected_indices[n]

        return action  # is ndarray

    def _get_info(self):
        """
        Compute auxiliary information for debugging.

        Returns:
            dict: Info about state of environment.
        """

        episode_complete = self.is_episode_complete()
        steps_at_completion = -1
        terminated_at_completion = -1

        if episode_complete:  # via any - terminated or truncated
            steps_at_completion = self.steps
            terminated_at_completion = int(self.episode_terminated)

        scenario_value = self.get_scenario_value(
            column = self.scenario_column,
        )

        last_action_info = self.last_action
        if self.last_action is None:
            last_action_info = -1  # Gym requires non-null

        return {
            "step": self.steps,
            "last-action": last_action_info,
            "last-reward": self.last_reward,
            "scenario": scenario_value,
            "obs-column": self.observation_column,
            "obs-value": self.observation_value,
            "obs-index": self.observation_index,
            "steps-at-completion": steps_at_completion,
            "terminated-at-completion": terminated_at_completion,
        }

    def _get_obs(self) -> dict:
        """Convert internal state to observation format.

        Returns:
            dict: Observation with agent and target positions
        """
        # Observation depends on previous action because some actions affect observation.
        
        if self.last_action is None:
            self.observation_column = self.initial_observation_column
            self.observation_column_value = self.get_scenario_value(column=self.observation_column)  # the answer
        elif self.is_internal_action(self.last_action): #self.last_action < num_perception_actions:
            self.observation_column = self.internal_action_columns[self.last_action]
            self.observation_column_value = self.get_scenario_value(column=self.observation_column)  # the answer
        else:  # external action; see initial obs + model has action as input
            self.observation_column = self.last_action_key
            self.observation_column_value = str(self.last_reward)  # Observe what happened, as effect

        self.observation_value = self.create_column_value_encoding(self.observation_column, self.observation_column_value)
        self.observation_index = self.get_observation_index(self.observation_value)
        return {
            self.get_token_key(): np.array(self.observation_index, dtype=int),   
        }
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        Start a new episode.

        Args:
            seed: Random seed for reproducible episodes
            options: Additional configuration (unused in this example)

        Returns:
            tuple: (observation, info) for the initial state
        """
        # IMPORTANT: Must call this first to seed the random number generator
        super().reset(seed=seed)
        self.steps = 0
        self.episode_terminated = False  # task achieved, goal reached, ended, etc.
        self.episode_truncated = False  # ran out of time
        self._reset_state()

        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def _reset_state(self):
        self.scenario_index = self.get_random_scenario()
        self.last_action = None
        self.last_action_key = None
        self.last_reward = 0.0

        self.observation_column = self.initial_observation_column
        self.observation_column_value = None
        self.observation_index = None

    def _update_state(self, action):
        actions_array = action.get("actions")
        action_index = self.get_action_index(actions_array)
        self.last_action = action_index
        num_internal_action_columns = len(self.internal_action_columns)
        action_reward_keys = list(self.external_action_rewards.keys())

        if self.is_internal_action(self.last_action):
            # Nothing happens unless you do the wrong thing
            # self.last_action will affect the next observation.
            pass  
        else:  # external

            # Find all the rewards applicable for this action in this scenario
            action_reward_index = self.last_action - num_internal_action_columns
            self.last_action_key = action_reward_keys[action_reward_index]
            external_action_rewards_dict = self.external_action_rewards.get(self.last_action_key)

            max_reward = 0.0
            min_reward = 0.0
            for condition, conditional_reward in external_action_rewards_dict.items():
                conditional_reward = float(conditional_reward)
                condition_value = self.get_scenario_value(column=condition)
                if condition_value == self.value_true:
                    max_reward = max(max_reward, conditional_reward)
                    min_reward = min(min_reward, conditional_reward)

            net_reward = max_reward + min_reward
            self.last_reward = net_reward

    def _get_reward(self) -> float:
        return self.last_reward

    def step(self, action):
        """
        Execute one timestep within the environment.

        Args:
            action: The action to take (0-3 for directions)

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # Map the discrete action (0-3) to a movement direction
        episode_complete = self.is_episode_complete()
        if not episode_complete:
            self._update_state(action)

        # Deferred termination. Before updating, check if we already ended the episode
        # last time. If so, only now can we issue terminated / truncated. This is to enable
        # streaming learning, where the last observation may contain vital reward signal and
        # the model needs to learn from it.
        #truncated = False
        #if episode_complete:  # ended last iter
        #    if self.episode_truncated:
        #        truncated = True

        # Only one of truncated and terminated should be true.
        # Check and latch if episode completion criteria reached
        if not self.episode_truncated:
            self.episode_terminated = self.episode_terminated or self._is_episode_terminated()

        # Check and latch if episode timeout reached
        if not self.episode_terminated:
            self.episode_truncated = self.episode_truncated or self._is_episode_truncated()

        # Simple reward structure: +1 for reaching target, 0 otherwise
        # Alternative: could give small negative rewards for each step to encourage efficiency
        reward = self._get_reward()

        observation = self._get_obs()  # will immediately display completion observation
        info = self._get_info()

        # if not episode_complete:
        self.steps += 1
        return observation, reward, self.episode_terminated, self.episode_truncated, info

    def is_episode_complete(self) -> bool:
        return self.episode_terminated or self.episode_truncated

    def _is_episode_truncated(self) -> bool:
        # e.g. max_steps=10, then steps >= 9 is truncated
        if self.steps >= (self.max_steps -1):
            return True
        return False
    
    def _is_episode_terminated(self) -> bool:
        """
        Specify logic to determine end of episode criteria.
        Otherwise, will end on timeout. 
        """
        if self.terminate_min_reward is not None:
            if self.last_reward <= self.terminate_min_reward:
                return True
        
        if self.terminate_max_reward is not None:
            if self.last_reward >= self.terminate_max_reward:
                return True

        return False        
