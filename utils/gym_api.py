from typing import Callable, List, Optional, Tuple, Union

import gym
import numpy as np
import torch
from PIL import Image
from torchvision import transforms


class SpoLacq3(gym.Env):
    def __init__(
        self,
        image_list: List[str],
        d_image_features: Optional[int] = None,
        d_embed: Optional[int] = None,
        action2text: Optional[Callable] = None,
        sounddic: Optional[List[str]] = None,
    ):
        if d_image_features:
            self.action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(d_image_features + d_embed,))
            self.action2text = action2text

        elif sounddic:
            self.action_space = gym.spaces.Discrete(len(sounddic))
            self.action2text = lambda i: sounddic[i]

        self.observation_space = gym.spaces.Dict(
            dict(
                state=gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64),
                leftimage=gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3, 224, 224), dtype=np.float64),
                rightimage=gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3, 224, 224), dtype=np.float64),
            )
        )

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.transform = transforms.Normalize(mean=self.mean, std=self.std)

        self.image_list = image_list
        self.num_images = len(image_list)
        self.make_food_storage(image_list)

        self.reset()

    def make_food_storage(self, image_list: list):
        self.food_storage = list()
        for path in image_list:
            image = Image.open(path)
            image = np.asarray(image)
            self.food_storage.append(image)

    def get_transformed_image(self, i: int):
        image = self.food_storage[i]
        image = image.transpose(2, 0, 1)
        image = torch.FloatTensor(image / 255.0)
        image = self.transform(image)
        image = image.numpy()
        return image

    def reset(self):
        self.internal_state = np.random.randint(0, 256, size=3)
        self.leftimage_idx = np.random.randint(self.num_images)
        self.rightimage_idx = np.random.randint(self.num_images)
        obs = dict(
            state=(self.internal_state / 255.0 - self.mean) / self.std,
            leftimage=self.get_transformed_image(self.leftimage_idx),
            rightimage=self.get_transformed_image(self.rightimage_idx),
        )
        return obs

    def step(self, action):
        reward = self.reward(action)
        self.internal_state = np.random.randint(0, 256, size=3)
        self.leftimage_idx = np.random.randint(self.num_images)
        self.rightimage_idx = np.random.randint(self.num_images)
        obs = dict(
            state=(self.internal_state / 255.0 - self.mean) / self.std,
            leftimage=self.get_transformed_image(self.leftimage_idx),
            rightimage=self.get_transformed_image(self.rightimage_idx),
        )
        return obs, reward, True, {}

    def get_correct_transcript(self):
        leftimage_rgb = np.mean(self.food_storage[self.leftimage_idx], axis=(0, 1))
        rightimage_rgb = np.mean(self.food_storage[self.rightimage_idx], axis=(0, 1))

        leftimage_distance = np.linalg.norm(leftimage_rgb - self.internal_state)
        rightimage_distance = np.linalg.norm(rightimage_rgb - self.internal_state)

        if leftimage_distance < rightimage_distance:
            correct_image_idx = self.leftimage_idx
        else:
            correct_image_idx = self.rightimage_idx

        correct_image_path = self.image_list[correct_image_idx]
        correct_food = correct_image_path.split("/")[3].replace("_", " ").upper()
        preposition = "AN" if correct_food[0] in ["A", "O", "E"] else "A"
        return f"I WANT {preposition} {correct_food}"

    def reward(self, action):
        transcript = self.action2text(action)
        correct_transcript = self.get_correct_transcript()

        if transcript == correct_transcript:
            return 1
        else:
            return 0
