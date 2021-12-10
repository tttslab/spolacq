from glob import glob
import pickle
import random
from typing import Callable, List, Tuple, Union

import gym
import librosa
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from stable_baselines3.common.base_class import BaseAlgorithm
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tqdm import tqdm

class Food():
    """
    Definition of food.
    It has name, image, and the mean RGB of an image.
    """
    
    def __init__(self, name: str, image: np.ndarray, RGB: np.ndarray):
        self.name = name
        self.image = image
        self.RGB = RGB


class DialogWorld:
    """
    Dialogue partner for the language learning agent,
    which is the outside environment.
    An agent receives randomly chosen two images of food from DialogWorld.
    The DialogWorld recognizes an agent's utterance and returns feedback.
    
    :param MAX_STEP: Number of dialogue turns per episode.
    :param hasNO: If an agent has the option of not eating either of two foods,
        set True.
    :param FOODS: Tuple of food names. The type of food names must match
        the type of the return value of ASR. For example, Wav2Vec2 returns
        a space-delimited sequence of uppercase letters, e.g., GREEN PEPPER.
    :param datadir: A directory of food images.
    :param asr: If args.use_real_time_asr==True, you can use any ASR of
        Callable[[np.ndarray], str]; otherwise, it is the identity function,
        i.e., lambda x: x.
    """
    
    def __init__(
        self,
        MAX_STEP: int,
        hasNO: bool,
        FOODS: tuple,
        datadir: str,
        asr: Callable[[Union[np.ndarray, str]], str],
    ):
        print("Initializaing DialogWorld")
        self.MAX_STEP = MAX_STEP
        self.hasNO = hasNO
        self.FOODS = FOODS
        self.make_food_storage(datadir)
        self.asr = asr
        print("Have prepared ASR")
        self.reset()
    
    def make_food_storage(self, datadir: str) -> None:
        self.food_storage = list() # list of food that the environment has
        for f in self.FOODS:
            if f == "NO": continue
            image_paths = glob(f"{datadir}/{f.replace(' ', '_').lower()}/test*/*.jpg")
            assert len(image_paths) == 30, "mismatch in test image dataset"
            for image_path in image_paths:
                image = Image.open(image_path)
                image = image.resize((224, 224))
                image = np.array(image)
                RGB = np.mean(image.astype(float)/255, axis=(0,1))
                self.food_storage.append(Food(f, image, RGB))
    
    def reset(self) -> None:
        self.num = self.MAX_STEP
        self.leftfood = random.choice(self.food_storage)
        self.rightfood = random.choice(self.food_storage)
        self.done = False
    
    def step(self, action: Union[np.ndarray, str]) -> Tuple[Tuple[bool, int], bool]:
        text = self.asr(action)
        if self.hasNO:
            correct_answer = ["NO", self.leftfood.name, self.rightfood.name]
        else:
            correct_answer = [self.leftfood.name, self.rightfood.name]
        if text in correct_answer:
            dlg_success = True
            foodID = self.FOODS.index(text)
        else:
            dlg_success = False
            foodID = -1
        self.num -= 1
        if self.num == 0:
            self.done = True
        self.leftfood = random.choice(self.food_storage)
        self.rightfood = random.choice(self.food_storage)
        return (dlg_success, foodID), self.done
    
    def observe(self) -> dict:
        return dict(num=self.num, leftfood=self.leftfood, rightfood=self.rightfood)


class SpoLacq1(gym.Env):
    """
    RL environment described in the paper
    
    M. Zhang, T. Tanaka, W. Hou, S. Gao, T. Shinozaki,
    "Sound-Image Grounding Based Focusing Mechanism for Efficient Automatic Spoken Language Acquisition,"
    in Proc. Interspeech, 2020.
    
    Internal state of spolacq agent is inside in this environment.
    An agent has a preferred color (RGB) as its internal state.
    An agent wants food that is close to that color.
    An agent receives randomly chosen two images of food from DialogWorld.
    An agent is rewarded when it speaks the name of the food it wants.
    
    :param FOODS: Tuple of food names. The type of food names must match
        the type of the return value of ASR. For example, Wav2Vec2 returns
        a space-delimited sequence of uppercase letters, e.g., GREEN PEPPER.
    :param datadir: A directory of food images.
    :param sounddic: If args.use_real_time_asr==True, it is an object of
        List[np.ndarray]; otherwise, it is an object of List[str].
    :param asr: If args.use_real_time_asr==True, you can use any ASR of
        Callable[[np.ndarray], str]; otherwise, it is the identity function,
        i.e., lambda x: x.
    """
    
    MAX_RGB = 1 # normalized image
    
    def __init__(
        self,
        FOODS: tuple,
        datadir: str,
        sounddic: Union[List[np.ndarray], List[str]],
        asr: Callable[[Union[np.ndarray, str]], str],
    ):
        super().__init__()
        self.dlgworld = DialogWorld(1, False, FOODS, datadir, asr)
        self.action_space = gym.spaces.Discrete(len(sounddic))
        self.observation_space = gym.spaces.Dict(
            dict(
                state=gym.spaces.Box(low=0, high=self.MAX_RGB, shape=(3,)),
                leftfoodRGB=gym.spaces.Box(low=0, high=self.MAX_RGB, shape=(3,)),
                rightfoodRGB=gym.spaces.Box(low=0, high=self.MAX_RGB, shape=(3,)),
                leftimage=gym.spaces.Box(low=0, high=255, shape=(224, 224, 3), dtype=np.uint8),
                rightimage=gym.spaces.Box(low=0, high=255, shape=(224, 224, 3), dtype=np.uint8),
                step=gym.spaces.Discrete(self.dlgworld.MAX_STEP+1),
                leftfoodID=gym.spaces.Discrete(len(FOODS)),
                rightfoodID=gym.spaces.Discrete(len(FOODS)),
            )
        )
        # Read sound files for sound dictionary
        self.sounddic = sounddic # convert categorical ID to wave utterance
        self.FOODS = FOODS
        self.succeeded_log = list()
        self.reset()
    
    def reset(self) -> dict:
        # Initialize the dialogue world
        self.dlgworld.reset()
        # Initialize the internal state of spolacq agent
        self.preferredR = random.random()
        self.preferredG = random.random()
        self.preferredB = random.random()
        return self.observe()
    
    def step(self, action: int) -> Tuple[dict, int, bool, dict]:
        """:param action: A number in self.action_space"""
        old_state = self.observe()
        utterance = self.sounddic[action]
        feedback, dlg_done = self.dlgworld.step(utterance)
        self.update_internal_state(feedback)
        new_state = self.observe()
        reward = self.reward(old_state, new_state, feedback)
        if reward > 0: self.succeeded_log.append(action)
        return new_state, reward, dlg_done, dict()
    
    def observe(self) -> dict:
        insideobs = np.array([self.preferredR, self.preferredG, self.preferredB])
        outsideobs = self.dlgworld.observe()
        return dict(
            state=insideobs,
            leftfoodRGB=outsideobs["leftfood"].RGB,
            rightfoodRGB=outsideobs["rightfood"].RGB,
            leftimage=outsideobs["leftfood"].image,
            rightimage=outsideobs["rightfood"].image,
            step=outsideobs["num"],
            leftfoodID=self.FOODS.index(outsideobs["leftfood"].name),
            rightfoodID=self.FOODS.index(outsideobs["rightfood"].name),
        )
    
    def update_internal_state(self, feedback: Tuple[bool, int]) -> None:
        # Initialize the internal state of spolacq agent
        self.preferredR = random.random()
        self.preferredG = random.random()
        self.preferredB = random.random()
    
    def reward(self, old_state: dict, new_state: dict, feedback: Tuple[bool, int]) -> int:
        if not feedback[0]: # failed dialogue
            return 0
        leftfood_distance = np.linalg.norm(old_state["state"] - old_state["leftfoodRGB"])
        rightfood_distance = np.linalg.norm(old_state["state"] - old_state["rightfoodRGB"])
        if leftfood_distance < rightfood_distance and feedback[1] == old_state["leftfoodID"]:
            return 1 # Agent want leftfood and agent's utterance is leftfood's name
        elif leftfood_distance >= rightfood_distance and feedback[1] == old_state["rightfoodID"]:
            return 1 # Agent want rightfood and agent's utterance is rightfood's name
        else:
            return 0
    
    def render(self, mode: str = "console", close=False) -> None:
        state = self.observe()
        leftfood_distance = np.linalg.norm(state["state"] - state["leftfoodRGB"])
        rightfood_distance = np.linalg.norm(state["state"] - state["rightfoodRGB"])
        if leftfood_distance < rightfood_distance:
            print(f"The left {self.FOODS[state['leftfoodID']]} is preferred.")
        else:
            print(f"The right {self.FOODS[state['rightfoodID']]} is preferred.")
        if mode == "human":
            plt.subplot(1, 2, 1)
            plt.imshow(state["leftimage"])
            plt.subplot(1, 2, 2)
            plt.imshow(state["rightimage"])
    
    def close(self) -> None:
        pass
    
    def seed(self, seed=None) -> None:
        random.seed(seed)


def asr_segments(segment_pkl: str, asr: Callable[[np.ndarray], str], num_segment: int
) -> Tuple[List[np.ndarray], List[str], List[str], List[np.ndarray]]:
    segment_wave = list()
    segment_text = list()
    segment_path = list()
    segment_spec = list()
    max_spec_len = 0
    with open(segment_pkl) as f:
        for path in tqdm(f, desc="ASR of audio segments", total=num_segment):
            path = path.strip()
            with open(path, "rb") as g:
                spectrogram = pickle.load(g)
                log_pad = 0.010
                spectrogram = np.log(np.abs(spectrogram) + log_pad).T
                max_spec_len = max(max_spec_len, len(spectrogram))
            
            path = path.replace("feat_pkls", "noisy_trimed_wavs")
            path = path.replace(".pkl", ".wav")
            utterance, _ = librosa.load(path, sr=16000)
            transcription = asr(utterance)
            
            segment_wave.append(utterance)
            segment_text.append(transcription)
            segment_path.append(path)
            segment_spec.append(spectrogram)
    
    # Padding
    for i, spec in enumerate(segment_spec):
        pad = spec[-1:].repeat(max_spec_len-len(spec), axis=0)
        spec = np.concatenate((spec, pad), axis=0)
        segment_spec[i] = spec
    return segment_wave, segment_text, segment_path, segment_spec


def test(num_episode: int, env: SpoLacq1, model: BaseAlgorithm) -> None:
    """Test the learnt agent."""
    
    for i in range(num_episode):
        print(f"episode {i}", "-"*40)
        state = env.reset()
        total_reward = 0
        while True:
            # render the state
            env.render()
            
            # Agent gets an environment state and returns a decided action
            action, _ = model.predict(state, deterministic=True)
            
            # Environment gets an action from the agent, proceeds the time step,
            # and returns the new state and reward etc.
            state, reward, done, info = env.step(action)
            total_reward += reward
            utterance = env.dlgworld.asr(env.sounddic[action])
            print(f"utterance: {utterance}, reward: {reward}")
            
            if done:
                print(f"total_reward: {total_reward}\n")
                break


def plot_reward(tb_path: str, save_path: str) -> None:
    """
    Plot the tensorboard log with matplotlib.
    
    :param tb_path: spolacq_tmplog/*/events.out.tfevents.*
    """
    
    event = EventAccumulator(tb_path)
    event.Reload()
    ep_rew_mean = event.Scalars("rollout/ep_rew_mean")
    
    x = np.zeros(len(ep_rew_mean))
    y = np.zeros(len(ep_rew_mean))
    
    for i, r in enumerate(ep_rew_mean):
        x[i] = r.step
        y[i] = r.value
    
    plt.plot(x, y)
    plt.xlabel("Episode")
    plt.ylabel("Reward (moving average of 100 steps)")
    plt.grid()
    plt.savefig(save_path)