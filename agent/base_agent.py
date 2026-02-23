import gymnasium as gym
import torch
from torch import nn
from copy import deepcopy
from dataclasses import dataclass
from model.optimizer import ModelOptimizer, ModelOptimizerConfig
from util.device import get_device
from util.log import periodic, get_run_path
from util.log_writer import ScalarLogWriter, TensorboardLogWriter, WandbLogWriter
from util.config import dataclass_write_json_file

import logging

logger = logging.getLogger(__name__)


@dataclass
class BaseAgentConfig:

    # Experiment configuration
    environment_id:str = ""
    random_policy:bool = False
    print_parameters:bool = False
    batch_size:int = 0
    
    steps_training:int = 0
    steps_evaluate:int = 0
    steps_console_update: int = 100  # How often to update console with progress

    # Primary optimizer
    optimizer_type: str = "adam"
    optimizer_learning_rate: float = 0.01
    optimizer_momentum: float = 0.0
    optimizer_weight_decay: float = 0.0
    optimizer_clip_grad_norm: float = 0.0

    # Logging
    log_type: str = "tensorboard"
    log_path: str = "./runs"
    log_prefix: str = ""
    log_period: int = 100
    log_combined:bool = False  # If true, everything will be logged to one timeline


class BaseAgent(nn.Module):
    """
    A basic RL Agent which ties together model and environment.
    Derived from nn.Module because this is important for auto-detection of trainable parameters, 
    even when explicitly relayed to different optimizers.
    """

    mode_training = "training"
    mode_evaluate = "evaluate"
    mode_combined = "combined"

    def __init__(self, config:BaseAgentConfig):
        super().__init__()
        self.config = config
        self.device = get_device()
        self.run_path = get_run_path(
            prefix = self.config.log_prefix, 
            path = self.config.log_path,
        )
        dataclass_write_json_file(
            obj = config,
            file_path = self.run_path, 
            file_name = "config.json",
        )

        self.trainable_modules = nn.ModuleDict()
        self.mode = None
        self.writer = None
        self.global_steps = {
            BaseAgent.mode_training: 0,
            BaseAgent.mode_evaluate: 0,
            BaseAgent.mode_combined: 0,
        }

        # Create environments
        self.envs = gym.make_vec(
            id = self.config.environment_id,
            num_envs = self.config.batch_size,
            vectorization_mode="async",
            vector_kwargs={"shared_memory":False},  # Caused a bug if omitted
        )

        # Create an empty action space dict for later convenience
        self.action_space_dict = {} 
        action_sample = self.envs.action_space.sample()
        for key in action_sample:
            self.action_space_dict[key] = None

        # Create one env to examine its calculated properties.
        # Difficult to get things from async vector envs inside.
        self._env = gym.make(
            id = self.config.environment_id,
        ).unwrapped
        observation, info = self._env.reset()
        
        self.num_actions = BaseAgent._get_action_space_size(self._env.action_space)
        logger.info(f"Actions: {self.num_actions}")

        self._create_model(observation)        
        if self.config.print_parameters:
            self.print_parameters()

        self._optimizer = self._create_model_optimizer()
        self._create_log_writers()

    @staticmethod
    def _get_action_space_size(action_space:dict):
        total = 0
        for k, v in action_space.items():
            total += v.n
        return total

    def _actions_tensor_to_dict(self, actions_tensor):
        actions_numpy = actions_tensor.detach().numpy()
        action_space = deepcopy(self.action_space_dict)
        offset = 0
        for k in action_space.keys():
            action_size = self._env.action_space[k].n
            a1 = offset
            a2 = a1 + action_size
            actions_slice = actions_numpy[:,a1:a2]
            action_space[k] = actions_slice
            offset += action_size
        return action_space
    
    def _create_model_optimizer(self):
        optimizer_config = ModelOptimizerConfig(
            name = "optimizer",
            optimizer_type = self.config.optimizer_type,
            learning_rate = self.config.optimizer_learning_rate,
            momentum = self.config.optimizer_momentum,
            weight_decay = self.config.optimizer_weight_decay,
            clip_grad_norm = self.config.optimizer_clip_grad_norm,
        )
        parameters = self._get_model_parameters()
        optimizer = ModelOptimizer(optimizer_config, parameters)
        return optimizer

    def _get_model_parameters(self):
        parameters_list = []
        for key in self.trainable_modules.keys():
            parameters_list += list(self.trainable_modules[key].parameters())
        return parameters_list
    
    def print_parameters(self):
        for name, param in self.trainable_modules.named_parameters():
            print(f"Name: {name} | Shape: {param.shape}")

    def do_training(self):
        self.mode = BaseAgent.mode_training
        self.reset()
        self.do_steps(self.config.steps_training)

    @torch.no_grad()
    def do_evaluate(self):
        self.mode = BaseAgent.mode_evaluate
        self.reset()
        self.do_steps(self.config.steps_evaluate)

    def do_steps(self, num_steps:int):
        for step in range(num_steps):
            self.step(step, num_steps)

        global_step = self._get_global_step() +1
        self._flush_writer(global_step)

    def get_mode(self) -> str:
        return self.mode
    
    def is_mode_training(self):
        return self.mode == BaseAgent.mode_training

    def is_mode_evaluate(self):
        return self.mode == BaseAgent.mode_evaluate

    def reset(self):
        """
        Reset all environments and store reset obs.
        Reset observation history for all environments too.
        """
        self.obs, info = self.envs.reset()

    def _create_log_writers(self):
        self.writers = {}
        self.writers[BaseAgent.mode_training] = self._create_writer(BaseAgent.mode_training)
        self.writers[BaseAgent.mode_evaluate] = self._create_writer(BaseAgent.mode_evaluate)
        self.writers[BaseAgent.mode_combined] = self._create_writer(BaseAgent.mode_combined)

    def _create_writer(self, epoch_type:str) -> ScalarLogWriter:
        if self.config.log_type == TensorboardLogWriter.WRITER_TYPE:
            return TensorboardLogWriter(
                epoch_type = epoch_type,
                log_path = self.run_path,
            )
        elif self.config.log_type == WandbLogWriter.WRITER_TYPE:
            return WandbLogWriter(
                epoch_type = epoch_type,
                log_path = self.run_path,
            )
        raise NotImplementedError 

    def _get_writer(self, epoch_type:str) -> ScalarLogWriter:
        if self.config.log_combined:
            return self.writers[BaseAgent.mode_combined]
        return self.writers[epoch_type]

    def _flush_writer(self, global_step:int):
        writer = self._get_writer(self.mode)
        writer.write_scalars(time_index = global_step)
        writer.reset()  # clear logged values to allow another block to be accumulated

    def step(self, step:int, num_steps:int):
        if (step < 100) or (num_steps < 100) or periodic(step, self.config.steps_console_update):
            logger.info(f"Step: {step}/{num_steps} mode:{self.mode}")

        # Get actions from model
        actions = self._step_get_actions(self.obs)
        action_space = self._actions_tensor_to_dict(actions)

        # Optional hard override of whatever model / agent to generate random baseline
        # Model can still update, but actions guaranteed to be random.
        if self.config.random_policy:
            action_space = self._get_random_actions()

        # Update environments with actions
        self.obs, rewards, terminated, truncated, info = self.envs.step(action_space)

        # Update loss and train model given rewards and new observation
        reset_mask = torch.logical_or(torch.tensor(terminated), torch.tensor(truncated))
        self._step_result(
            self.obs,
            rewards, 
            info,
            reset_mask,
        )

        self._update_step()
        if reset_mask.any():
            self._step_reset_envs(reset_mask)

    def _get_random_actions(self):
        actions_sample = self.envs.action_space.sample()
        return actions_sample
    
    def _step_reset_envs(self, reset_mask):

        # Reset only the finished environments
        reset_obs, reset_info = self.envs.reset(
            options={"reset_mask": reset_mask.numpy()}
        )

        # Use boolean masking to update the main observation batch in place
        self.obs[reset_mask] = reset_obs

    def _update_step(self):

        # Track global step
        global_step = self._get_global_step() +1
        self._set_global_step(global_step)
        self._add_log_values()  # Log every step

        # Accumulate, average (for smoothing) and periodically write log values:
        if periodic(t=global_step, period=self.config.log_period):
            self._flush_writer(global_step)

    def _get_global_step(self) -> int:
        # Steps can be counted per mode, or combined
        # As they're mostly used in logging.
        key = self._get_global_step_key()
        global_step = self.global_steps[key]
        return global_step
    
    def _set_global_step(self, global_step:int):
        key = self._get_global_step_key()
        self.global_steps[key] = global_step
        return global_step

    def _get_global_step_key(self) -> str:
        if self.config.log_combined:
            return BaseAgent.mode_combined
        return self.mode

    def log_scalar(self, name:str, value:float):
        writer = self._get_writer(self.mode)
        writer.add_scalar(name, value)

    def _create_model(self, observation):
        """
        Override to instantiate your model.
                
        :param observation: A prototype observation from a single env reset().
        """
        raise NotImplementedError

    def _step_get_actions(self, observation):
        """
        Override this function to use your model to produce actions.
                
        :param self: Description
        :param observation: Description
        """
        return self.get_random_actions()

    def _step_result(
        self,
        obs,
        rewards,
        info,
        reset_mask,
    ):
        """
        Override this function to handle the result of a step(), 
        e.g. to train your model.
                
        :param self: Description
        :param obs: Description
        :param rewards: Description
        :param info: Description
        :param reset_mask: Description
        """
        self.log_scalar("Reward-mean", rewards.mean())

        # Log outcomes of episodes, if ended this step.
        def log_non_negative(key:str):
            array = info[key]
            non_negative = array[array >= 0]
            for x in non_negative:
                self.log_scalar(key, x)  # averaged later

        log_non_negative("steps-at-completion")
        log_non_negative("terminated-at-completion")

    def _add_log_values(self):
        """
        Accumulate logs about the step
        """
        pass
