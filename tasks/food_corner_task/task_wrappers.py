import gymnasium as gym
from typing import Tuple
import numpy as np
import torch
from torchvision import transforms


class TransformImageWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224), antialias=True),  # Add resize
                transforms.Normalize(mean=self.mean, std=self.std),
            ]
        )

        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Box(low=-10000, high=10000, shape=(2,)),
                "leftimage": gym.spaces.Box(low=0, high=1, shape=(3, 224, 224)),
                "rightimage": gym.spaces.Box(low=0, high=1, shape=(3, 224, 224)),
            }
        )

    def transform_image(self, image: np.ndarray) -> np.ndarray:
        image = image.transpose(2, 0, 1)
        image = torch.FloatTensor(image / 255.0)
        image = self.transform(image)
        image = image.numpy()
        return image

    def reset(self, seed=None, options=None) -> dict:
        observation, info = self.env.reset()
        observation["leftimage"] = self.transform_image(observation["leftimage"])
        observation["rightimage"] = self.transform_image(observation["rightimage"])
        return observation, info

    def step(self, action_waveform: np.ndarray) -> Tuple[dict, float, bool, bool, dict]:
        observation, reward, terminated, truncated, info = self.env.step(action_waveform)
        observation["leftimage"] = self.transform_image(observation["leftimage"])
        observation["rightimage"] = self.transform_image(observation["rightimage"])
        return observation, reward, terminated, truncated, info

def wrap_task_env(base_env: gym.Env, **kwargs):
    wrapped_env = TransformImageWrapper(base_env)
    return wrapped_env
