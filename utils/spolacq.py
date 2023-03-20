from glob import glob
import os
import pickle
import random
from typing import Callable, List, Optional, Tuple, Union

import gym
import h5py
import librosa
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pydub import AudioSegment
import resampy
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from stable_baselines3.common.base_class import BaseAlgorithm
try:
    from tensorboard.backend.event_processing import event_accumulator, tag_types
except ImportError:
    pass
import torch
from torchvision import transforms
from tqdm import tqdm

from add_sil_and_noise import add_sil_and_noise
from add_sin_noise import add_sin_noise
from wav2vec2_api import ASR

class Food:
    """
    Definition of food.
    It has name, image, and the mean RGB of an image.
    """
    
    def __init__(self, name: str, image: np.ndarray, RGB: np.ndarray):
        self.name = name
        self.image = image
        self.RGB = RGB


class Question:
    """
    Definition of spoken question.
    :param spectrogram: spectrogram of spoken question.
    :param question_type: 0: Which do you want?, 1: Which do not you want?
    """
    
    def __init__(self, spectrogram: np.ndarray, question_type: int):
        self.spectrogram = spectrogram
        self.question_type = question_type


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
    :param question_paths: Paths of spoken questions from a dialogue partner,
        e.g., "Which do you want?".
    """
    
    def __init__(
        self,
        MAX_STEP: int,
        hasNO: bool,
        FOODS: tuple,
        datadir: str,
        asr: Callable[[Union[np.ndarray, str]], str],
        question_paths: Optional[List[Tuple[str, int]]] = None,
        spec_len: Optional[int] = None,
    ):
        print("Initializaing DialogWorld")
        self.MAX_STEP = MAX_STEP
        self.hasNO = hasNO
        self.FOODS = FOODS
        self.make_food_storage(datadir)
        if question_paths is not None:
            self.make_question_storage(question_paths, spec_len=spec_len)
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
    
    def make_question_storage(
        self,
        question_paths: List[Tuple[str, int]],
        win_len_sec: float = 0.025,
        hop_sec: float = 0.010,
        log_pad: float = 0.010,
        sr: int = 16000,
        spec_len: Optional[int] = None,
    ) -> None:
        question_storage = list()
        self.question_storage = list()
        max_spec_len = 0

        # wav -> spectrogram
        for path, question_type in question_paths:
            question, _ = librosa.load(path, sr=sr)
            question = librosa.stft(
                question,
                n_fft=int(win_len_sec*sr),
                hop_length=int(hop_sec*sr),
            )
            question = np.log(np.abs(question)+log_pad).T
            question_storage.append((question, question_type))
            max_spec_len = max(max_spec_len, len(question))

        if spec_len is None:
            spec_len = max_spec_len
        
        # Padding
        for spec, question_type in question_storage:
            if spec_len > len(spec):
                pad = spec[-1:].repeat(spec_len-len(spec), axis=0)
                spec = np.concatenate((spec, pad), axis=0)
            else:
                spec = spec[:spec_len]
            self.question_storage.append(Question(spec, question_type))

    def reset(self) -> None:
        self.num = self.MAX_STEP
        self.leftfood = random.choice(self.food_storage)
        self.rightfood = random.choice(self.food_storage)
        if hasattr(self, "question_storage"):
            self.question = random.choice(self.question_storage)
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
        if hasattr(self, "question_storage"):
            self.question = random.choice(self.question_storage)
        return (dlg_success, foodID, text), self.done
    
    def observe(self) -> dict:
        if hasattr(self, "question_storage"):
            return dict(num=self.num, leftfood=self.leftfood, rightfood=self.rightfood, question=self.question)
        else:
            return dict(num=self.num, leftfood=self.leftfood, rightfood=self.rightfood)
    
    def render(self, mode: str = "console", close=False) -> None:
        state = self.observe()
        if "question" not in state or state["question"].question_type == 0:
            print("Which do you want?")
        else:
            print("Which do not you want?")


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
    :param question_paths: Paths of spoken questions from a dialogue partner,
        e.g., "Which do you want?".
    """
    
    MAX_RGB = 1 # normalized image
    
    def __init__(
        self,
        FOODS: tuple,
        datadir: str,
        asr: Callable[[Union[np.ndarray, str]], str],
        question_paths: Optional[List[Tuple[str, int]]] = None,
        spec_len: Optional[int] = None,
    ):
        super().__init__()
        self.dlgworld = DialogWorld(1, False, FOODS, datadir, asr, question_paths, spec_len)
        space_dict = dict(
            state=gym.spaces.Box(low=0, high=self.MAX_RGB, shape=(3,)),
            leftfoodRGB=gym.spaces.Box(low=0, high=self.MAX_RGB, shape=(3,)),
            rightfoodRGB=gym.spaces.Box(low=0, high=self.MAX_RGB, shape=(3,)),
            leftimage=gym.spaces.Box(low=0, high=255, shape=(224, 224, 3), dtype=np.uint8),
            rightimage=gym.spaces.Box(low=0, high=255, shape=(224, 224, 3), dtype=np.uint8),
            step=gym.spaces.Discrete(self.dlgworld.MAX_STEP+1),
            leftfoodID=gym.spaces.Discrete(len(FOODS)),
            rightfoodID=gym.spaces.Discrete(len(FOODS)),
        )
        if hasattr(self.dlgworld, "question_storage"):
            space_dict.update(
                sound=gym.spaces.Box(
                    low=-float("inf"),
                    high=float("inf"),
                    shape=self.dlgworld.question_storage[0].spectrogram.shape,
                    dtype=float,
                ),
                question_type=gym.spaces.Discrete(2),
            )
        self.observation_space = gym.spaces.Dict(space_dict)
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
        obs = dict(
            state=insideobs,
            leftfoodRGB=outsideobs["leftfood"].RGB,
            rightfoodRGB=outsideobs["rightfood"].RGB,
            leftimage=outsideobs["leftfood"].image,
            rightimage=outsideobs["rightfood"].image,
            step=outsideobs["num"],
            leftfoodID=self.FOODS.index(outsideobs["leftfood"].name),
            rightfoodID=self.FOODS.index(outsideobs["rightfood"].name),
        )
        if "question" in outsideobs:
            obs.update(
                sound=outsideobs["question"].spectrogram,
                question_type=outsideobs["question"].question_type,
            )
        return obs
    
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

        if old_state.get("question_type", 0) == 0: # Which do you want?
            if leftfood_distance < rightfood_distance and feedback[1] == old_state["leftfoodID"]:
                return 1 # Agent want leftfood and agent's utterance is leftfood's name
            elif leftfood_distance >= rightfood_distance and feedback[1] == old_state["rightfoodID"]:
                return 1 # Agent want rightfood and agent's utterance is rightfood's name
            else:
                return 0
        else: # Which do not you want?
            if leftfood_distance < rightfood_distance and feedback[1] == old_state["rightfoodID"]:
                return 1 # Agent want leftfood and agent's utterance is rightfood's name
            elif leftfood_distance >= rightfood_distance and feedback[1] == old_state["leftfoodID"]:
                return 1 # Agent want rightfood and agent's utterance is leftfood's name
            else:
                return 0
    
    def render(self, mode: str = "console", close=False) -> None:
        self.dlgworld.render(mode=mode, close=close)
        state = self.observe()
        leftfood_distance = np.linalg.norm(state["state"] - state["leftfoodRGB"])
        rightfood_distance = np.linalg.norm(state["state"] - state["rightfoodRGB"])
        
        if state.get("question_type", 0) == 0:  # which do you want?
            if leftfood_distance < rightfood_distance:
                print(f"The left {self.FOODS[state['leftfoodID']]} is preferred.")
            else:
                print(f"The right {self.FOODS[state['rightfoodID']]} is preferred.")
        else:
            if leftfood_distance < rightfood_distance:
                print(f"The right {self.FOODS[state['rightfoodID']]} is NOT preferred.")
            else:
                print(f"The left {self.FOODS[state['leftfoodID']]} is NOT preferred.")
        
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
    RL environment described in the papers:
    
    M. Zhang, T. Tanaka, W. Hou, S. Gao, and T. Shinozaki,
    "Sound-Image Grounding Based Focusing Mechanism for Efficient Automatic Spoken Language Acquisition,"
    in Proc. Interspeech, 2020.
    
    K. Toyoda, Y. Kimura, M. Zhang, K. Hino, K. Mori, and T. Shinozaki,
    "Self-Supervised Spoken Question Understanding and Speaking With Automatic Vocabulary Learning,"
    in Proc. O-COCOSDA, 2021.
    
    :param FOODS: Tuple of food names. The type of food names must match
        the type of the return value of ASR. For example, Wav2Vec2 returns
        a space-delimited sequence of uppercase letters, e.g., GREEN PEPPER.
    :param datadir: A directory of food images.
    :param sounddic: If args.use_real_time_asr==True, it is an object of
        List[np.ndarray]; otherwise, it is an object of List[str].
    :param asr: If args.use_real_time_asr==True, you can use any ASR of
        Callable[[np.ndarray], str]; otherwise, it is the identity function,
        i.e., lambda x: x.
    :param question_paths: Paths of spoken questions from a dialogue partner,
        e.g., "Which do you want?".
    """
    
    def __init__(
        self,
        FOODS: tuple,
        datadir: str,
        sounddic: Union[List[np.ndarray], List[str]],
        asr: Callable[[Union[np.ndarray, str]], str],
        question_paths: Optional[List[Tuple[str, int]]] = None,
        spec_len: Optional[int] = None,
    ):
        super().__init__(FOODS, datadir, asr, question_paths, spec_len)
        self.action_space = gym.spaces.Discrete(len(sounddic))
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
            self.action_space = gym.spaces.Box(low=-np.inf, high=np.inf,
                                               shape=(d_image_features+d_embed,))
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
        image = torch.FloatTensor(image / 255.)
        image = self.transform(image)
        image = image.numpy()
        return image
    
    def reset(self):
        self.internal_state = np.random.randint(0, 256, size=3)
        self.leftimage_idx = np.random.randint(self.num_images)
        self.rightimage_idx = np.random.randint(self.num_images)
        obs = dict(
            state=(self.internal_state/255.-self.mean) / self.std,
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
            state=(self.internal_state/255.-self.mean) / self.std,
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


class RLPreprocessor:
    """
    Preprocessor for reinforcement learning.

    :param obj_name_list: List of food names
    :param segment_nnfeat: Path of segmented audio features
    :param image_nnfeat: Path of training image features
    """
    def __init__(
        self,
        obj_name_list: List[str],
        segment_nnfeat: str,
        image_nnfeat: Optional[str] = None,
    ):
        self.FOODS = tuple(f.upper().replace("_", " ") for f in obj_name_list)
        self.segment_features = np.load(segment_nnfeat).squeeze()
        if image_nnfeat: self.image_features = np.load(image_nnfeat).squeeze()
    
    def recognize(
        self,
        segment_list: str,
        asr_model_name: str,
        use_real_time_asr: bool = False,
        save: bool = True,
        wav2vec2_path = "./wav2vec2/",
    ) -> Callable[[Union[np.ndarray, str]], str]:
        """
        Recognize all audio segments in a file segment_list.
        """
        if os.path.isdir(wav2vec2_path):
            asr = ASR(wav2vec2_path)
        else:
            asr = ASR(asr_model_name)
            if save: asr.save(wav2vec2_path)
        
        self.segment_wave = list()
        self.segment_text = list()
        self.segment_path = list()
        with open(segment_list) as f:
            for path in tqdm(f, desc="ASR of audio segments", total=self.segment_features.shape[0]):
                path = path.strip()
                utterance, _ = librosa.load(path, sr=16000)
                transcription = asr(utterance)
                
                self.segment_wave.append(utterance)
                self.segment_text.append(transcription)
                self.segment_path.append(path)
        if use_real_time_asr:
            return asr
        else:
            return lambda x: x
    
    def focus(self, num_clusters: int, num_per_group: int, random_state: int = 2):
        kmeans = KMeans(n_clusters=num_clusters, random_state=random_state).fit(self.image_features)
        similarity = -cdist(kmeans.cluster_centers_, self.segment_features)
        focused_segment_ids = similarity.argsort(axis=1)[:, -num_per_group:].flatten()
        
        # Make sound dictionary, for spolacq agent, which
        # converts categorical ID to wave utterance.
        sounddic_wave = [self.segment_wave[i] for i in focused_segment_ids]
        sounddic_text = [self.segment_text[i] for i in focused_segment_ids]
        sounddic_path = [self.segment_path[i] for i in focused_segment_ids]
        self.check(sounddic_text)
        return sounddic_wave, sounddic_text, sounddic_path, kmeans.cluster_centers_
    
    def check(self, sounddic_text: List[str]) -> None:
        # Check if sounddic covers every food
        res_dict = {f: sounddic_text.count(f) for f in self.FOODS}
        print(res_dict, flush=True)
        assert 0 not in res_dict.values(), "The sound dictionary does not cover every food."


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


def plot_vwrr(qa_logs: List[str], label: str, obj_name_list: List[str]):
    """
    Plot Valid Word Recognition Rate (VWRR). VWRR is the rate that
    the recognition result of the agent's utterance is one of the foods.
    
    :param qa_logs: list of qa_log, i.e., ["*/nn/*/qa_log.txt", ...]
    :param label: legend of plot
    :param obj_name_list: list of FOODS
    """

    FOODS = tuple(f.upper().replace("_", " ") for f in obj_name_list)
    
    vw = list()
    for qa_log in qa_logs:
        with open(qa_log) as f:
            vw += [int(r.strip().split("|")[3] in FOODS) for r in f]
    
    x = list()
    y = list()

    for i in range(len(vw)//1000):
        moving_average = np.mean(vw[1000*i:1000*(i+1)])
        x.append(1000*(i+1))
        y.append(moving_average)
    
    yticks = np.array(range(0, 11, 1))/10
    
    plt.plot(x, y, label=label)
    plt.xlabel("Episode")
    plt.ylabel("VWRR (moving average of 1000 episodes)")
    plt.yticks(ticks=yticks, labels=[str(i) for i in yticks])
    plt.grid(visible=True, which="both")
    plt.legend()