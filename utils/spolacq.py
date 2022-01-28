from glob import glob
import pickle
import random
from typing import Callable, List, Tuple, Union

import gym
import h5py
import librosa
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pydub import AudioSegment
import resampy
from stable_baselines3.common.base_class import BaseAlgorithm
from tensorboard.backend.event_processing import event_accumulator, tag_types
import torch
from tqdm import tqdm

from add_sil_and_noise import add_sil_and_noise
from add_sin_noise import add_sin_noise

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
    Based on the internal state and the two images,
    the agent selects an audio segment from the sound dictionary and speaks it.
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
    
    def step(self, action: Union[np.ndarray, str]) -> Tuple[Tuple[bool, int, str], bool]:
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
        return (dlg_success, foodID, text), self.done
    
    def observe(self) -> dict:
        return dict(num=self.num, leftfood=self.leftfood, rightfood=self.rightfood)


class SpoLacq(gym.Env):
    """
    Base RL environment
    
    An agent has a preferred color (RGB) as its internal state.
    Internal state of spolacq agent is inside in this environment.
    An agent wants food that is close to that color.
    An agent receives randomly chosen two images of food from DialogWorld.
    An agent is rewarded when it speaks the name of the food it wants.
    
    :param FOODS: Tuple of food names. The type of food names must match
        the type of the return value of ASR. For example, Wav2Vec2 returns
        a space-delimited sequence of uppercase letters, e.g., GREEN PEPPER.
    :param datadir: A directory of food images.
    :param asr: If args.use_real_time_asr==True, you can use any ASR of
        Callable[[np.ndarray], str]; otherwise, it is the identity function,
        i.e., lambda x: x.
    """
    
    MAX_RGB = 1 # normalized image
    
    def __init__(
        self,
        FOODS: tuple,
        datadir: str,
        asr: Callable[[Union[np.ndarray, str]], str],
    ):
        super().__init__()
        self.dlgworld = DialogWorld(1, False, FOODS, datadir, asr)
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
        self.FOODS = FOODS
        self.reset()
    
    def reset(self) -> dict:
        # Initialize the dialogue world
        self.dlgworld.reset()
        # Initialize the internal state of spolacq agent
        self.preferredR = random.random()
        self.preferredG = random.random()
        self.preferredB = random.random()
        return self.observe()
    
    def step(self, action) -> Tuple[dict, int, bool, dict]:
        """:param action: A number in self.action_space"""
        raise NotImplementedError
    
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
    
    def update_internal_state(self, feedback: Tuple[bool, int, str]) -> None:
        # Initialize the internal state of spolacq agent
        self.preferredR = random.random()
        self.preferredG = random.random()
        self.preferredB = random.random()
    
    def reward(self, old_state: dict, new_state: dict, feedback: Tuple[bool, int, str]) -> int:
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


class SpoLacq1(SpoLacq):
    """
    RL environment described in the paper
    
    M. Zhang, T. Tanaka, W. Hou, S. Gao, T. Shinozaki,
    "Sound-Image Grounding Based Focusing Mechanism for Efficient Automatic Spoken Language Acquisition,"
    in Proc. Interspeech, 2020.
    
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
    
    def __init__(
        self,
        FOODS: tuple,
        datadir: str,
        sounddic: Union[List[np.ndarray], List[str]],
        asr: Callable[[Union[np.ndarray, str]], str],
    ):
        super().__init__(FOODS, datadir, asr)
        self.action_space = gym.spaces.Discrete(len(sounddic))
        # Read sound files for sound dictionary
        self.sounddic = sounddic # convert categorical ID to wave utterance
        self.succeeded_log = list()
    
    def step(self, action) -> Tuple[dict, int, bool, dict]:
        """:param action: A number in self.action_space"""
        old_state = self.observe()
        utterance = self.sounddic[action]
        feedback, dlg_done = self.dlgworld.step(utterance)
        self.update_internal_state(feedback)
        new_state = self.observe()
        reward = self.reward(old_state, new_state, feedback)
        if reward > 0: self.succeeded_log.append(action)
        return new_state, reward, dlg_done, dict()


class SpoLacq2(SpoLacq):
    """
    RL environment for a WaveGrad speech organ-based agent,
    which is described in the paper:
    
    T. Tanaka, R. Komatsu, T. Okamoto, and T. Shinozaki,
    "Pronunciation adaptive self speaking agent using WaveGrad,"
    in Proc. AAAI SAS, accepted, 2022.
    
    :param FOODS: Tuple of food names. The type of food names must match
        the type of the return value of ASR. For example, Wav2Vec2 returns
        a space-delimited sequence of uppercase letters, e.g., GREEN PEPPER.
    :param datadir: A directory of food images.
    :param asr: You can use any ASR of Callable[[np.ndarray], str].
    :param speech_organ: An instance of utils.main_wavegrad.WaveGradWrapper
        or utils.main_wavegrad.SoundDict.
    :param action_mask: Only those actions can be selected.
    :param add_sin_noise: Whether to add sine noise to the agent's utterance.
    :param sin_noise_db: dB of sine noise.
    :param sin_noise_freq: Frequency of sine noise.
    """
    
    def __init__(
        self,
        FOODS: tuple,
        datadir: str,
        asr: Callable[[np.ndarray], str],
        speech_organ,
        action_mask: List[int],
        add_sin_noise: bool,
        sin_noise_db = None,
        sin_noise_freq = None,
    ):
        super().__init__(FOODS, datadir, asr)
        self.action_space = gym.spaces.Discrete(len(speech_organ))
        self.action_mask = action_mask
        self.speech_organ = speech_organ
        self.num_image = len(self.dlgworld.food_storage)
        
        self.epoch = 0
        self.iteration = 0
        self.sum_reward = 0
        
        # log
        self.audio = dict()
        self.history = list()
        self.reward_list = list()
        
        # sin noise
        self.add_sin_noise = add_sin_noise
        self.sin_noise_db = sin_noise_db
        self.sin_noise_freq = sin_noise_freq
    
    def step(self, action) -> Tuple[dict, int, bool, dict]:
        """:param action: A number in self.action_space"""
        old_state = self.observe()
        action = self.action_mask.index(action)
        utterance = self.synthesize(action)
        feedback, dlg_done = self.dlgworld.step(utterance)
        self.update_internal_state(feedback)
        new_state = self.observe()
        reward = self.reward(old_state, new_state, feedback)
        self.update_info(action, reward, feedback[2])
        return new_state, reward, dlg_done, dict()
    
    def synthesize(self, action: int) -> np.ndarray:
        utterance, sr = self.speech_organ.predict(action)
        # utterance: torch.tensor, ndim == 1. -1 ~ 1.
        self.log_audio(utterance, self.epoch, self.iteration)
        utterance = self.recog_preprocess(utterance, sr=sr, noise_db=30, sil_len_ms=1000)
        return utterance
    
    def log_audio(self, audio: torch.Tensor, epoch: int, iteration: int) -> None:
        audio = audio.cpu().unsqueeze(0).numpy()
        savename = f"epoch{epoch:05}_iter{iteration:05}.wav"
        self.audio[savename] = audio
    
    def recog_preprocess(self, audio: torch.Tensor, sr: int, noise_db: int, sil_len_ms: int) -> np.ndarray:
        clean_elem = audio.cpu().numpy()
        clean_elem = AudioSegment(
            data=(clean_elem * (1 << 32 - 1)).astype("int32").tobytes(),
            sample_width=4,
            frame_rate=sr,
            channels=1,
        )
        noised_elem = add_sil_and_noise(clean_elem, noise_db=noise_db, sil_len_ms=sil_len_ms)
        if self.add_sin_noise:
            noised_elem = add_sin_noise(
                noised_elem,
                noise_db=self.sin_noise_db,
                freq=self.sin_noise_freq,
                sample_rate=sr,
                clean_dbfs=clean_elem.dBFS,
            )
        noised_elem = np.frombuffer(noised_elem.set_frame_rate(sr).raw_data, dtype="int32")
        noised_elem = (noised_elem / (1 << 32 - 1)).astype("float32")
        noised_elem = resampy.resample(noised_elem, sr, 16000)
        return noised_elem
    
    def update_info(self, action: int, reward: int, recognition_result: str) -> None:
        self.history += [((self.epoch, self.iteration), action, reward, recognition_result)]
        self.speech_organ.set_results(reward)
        self.sum_reward += reward
        self.iteration += 1
        if self.iteration == self.num_image:
            self.reward_list.append(self.sum_reward/self.num_image)
            self.sum_reward = 0
            self.iteration = 0
            self.epoch += 1
    
    def save_audio(self, audio_log_dir: str) -> None:
        with h5py.File(audio_log_dir, "w") as f:
            for savename, wave in self.audio.items():
                f[savename] = wave
    
    def save_history(self, path: str) -> None:
        with open(path, "w") as f:
            for time, action, reward, recognition_result in self.history:
                f.write(f"{time[0]},{time[1]}|{action}|{reward}|{recognition_result}\n")
    
    def save_reward(self, path: str) -> None:
        with open(path, "w") as f:
            f.write("\n".join([str(x) for x in self.reward_list]))


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


def plot_reward(tb_paths: List[str], label: str) -> None:
    """
    Plot the tensorboard log with matplotlib.
    
    :param tb_paths: list of spolacq_tmplog/*/events.out.tfevents.*
    :param label: legend of plot
    """
    
    x = list()
    y = list()
    
    for tb_path in tb_paths:
        event = event_accumulator.EventAccumulator(
            tb_path, size_guidance={tag_types.SCALARS: 1000000})
        event.Reload()
        ep_rew_mean = event.Scalars("rollout/ep_rew_mean")
        every100 = list()
        
        for r in ep_rew_mean:
            if r.step % 100 == 0:
                every100.append(r.value)
            if r.step % 1000 == 0:
                x.append(r.step)
                y.append(np.mean(every100))
                assert len(every100) == 10
                every100 = list()
    
    yticks = np.array(range(0, 11, 1))/10
    
    plt.plot(x, y, label=label)
    plt.xlabel("Episode")
    plt.ylabel("Reward (moving average of 1000 episodes)")
    plt.yticks(ticks=yticks, labels=[str(i) for i in yticks])
    plt.grid(visible=True, which="both")
    plt.legend()