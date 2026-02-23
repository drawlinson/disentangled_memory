import gymnasium as gym
from environment.stimulus_response_env import StimulusResponseEnv

class MouseGardenEnv(StimulusResponseEnv):
    ENV_NAME = "MouseGarden-v0"
    def __init__(
            self, 
    ):
        super().__init__(
            dataset_csv_file = "scenarios.csv",
            scenario_column = "Object",
            scenario_include = None,
            scenario_exclude = None,
            initial_observation_column= "Class",

            internal_action_columns = [
                'Does it look like a mouse?',
                'Is it bigger than a mouse?', 
                'Does it smell tasty?',
                'Does it have a long tail?',
                'Does it have four legs?',
                'Is it red?',
                'Is it green?',
                'Is it noisy?',
                'Is it watching you?',
                'Is it coming towards you?',
            ],  # + reset history [not implemented yet]
            external_action_rewards = {
                "Go to it": {
                    "Is it friendly?": 1,
                    "Does it eat mice?": -1,
                },
                "Eat it": {
                    "Is it edible?": 1,
                    "Is it poisonous?": -1,
                    "Does it eat mice?": -1,
                },
                "Hide": {
                    "Does it eat mice?": 1,
                },
                "Run away": {
                    "Does it chase mice?": -1,
                }, 
            },
            max_steps = 10,
            terminate_min_reward = -1.0,
            terminate_max_reward = 1.0,
        )

gym.register(
    id=MouseGardenEnv.ENV_NAME,
    entry_point=MouseGardenEnv,
)