from typing import Mapping, Any
import numpy as np
import textworld.gym


class RandomAgent(textworld.gym.Agent):
    """ Agent that randomly selects a command from the admissible ones. """

    def __init__(self, seed=1234):
        self.seed = seed
        self.rng = np.random.RandomState(self.seed)

    @property
    def infos_to_request(self) -> textworld.EnvInfos:
        return textworld.EnvInfos(admissible_commands=True)

    def act(self, obs: str, score: int, done: bool, infos: Mapping[str, Any]) -> str:
        return self.rng.choice(infos["admissible_commands"])

