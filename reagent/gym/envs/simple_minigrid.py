import gym
import gym_minigrid  # noqa
import numpy as np
from gym import spaces
from gym_minigrid.minigrid import DIR_TO_VEC


NUM_DIRECTIONS = len(DIR_TO_VEC)


class SimpleObsWrapper(gym.core.ObservationWrapper):
    """
    Encode the agent's position & direction in a one-hot vector
    """

    def __init__(self, env):
        super().__init__(env)

        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(self.env.width * self.env.height * NUM_DIRECTIONS,),
            dtype="float32",
        )

    def observation(self, obs):
        retval = np.zeros(
            (self.env.width * self.env.height * NUM_DIRECTIONS,), dtype=np.float32
        )
        retval[
            self.env.agent_pos[0] * self.env.height * NUM_DIRECTIONS
            + self.env.agent_pos[1] * NUM_DIRECTIONS
            + self.env.agent_dir
        ] = 1.0
        return retval
