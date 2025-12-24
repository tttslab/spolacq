import json
from pathlib import Path
import resampy
import numpy as np
import torchaudio
import torch
from torch import nn
import gymnasium as gym
from typing import Any, Dict, List, Optional, Tuple, Type, Union, Callable

from agents.base_agent_with_units import (
    BaseAgentWithUnits,
    BaseImage2Unit,
    BaseSpeech2Unit,
    BaseUnit2Speech,
    BaseRLUnit,
)
from .tools.I2U.get_s5hubert_unit import S5HubertForSyllableDiscovery
from .tools.I2U.models_i2u import ImageToUnit
from .tools.U2S.model_tacotron2 import Tacotron2
from .tools.U2S.text import text_to_sequence
from .tools.U2S.train2 import create_hparams
from .tools.hifi_gan.inference import Generator
from .utils.sb3_api import CustomDDPG, CustomTD3Policy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.td3.policies import TD3Policy
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import NormalActionNoise


class Image2Unit(ImageToUnit, BaseImage2Unit):
    def __init__(self, word_map=None):
        # Partially skip initializing the parent class or initialize with dummy values
        if word_map is None:
            self._initialized = False
        else:
            super().__init__(word_map=word_map)
            self._initialized = True

    def init_model(self, word_map):
        # Set only the necessary properties
        self.word_map = word_map
        # Call the necessary initialization procedures of the parent class
        super().__init__(word_map=word_map)
        self._initialized = True

    def set_beam_size(self, beam_size):
        self.beam_size = beam_size

    def infer(self, action: torch.Tensor) -> list[str]:
        unit_seq = super().infer(action=action, beam_size=self.beam_size)
        return unit_seq

    def load_checkpoint(self, image_to_unit_model_path: Path):
        self.load_state_dict(torch.load(str(image_to_unit_model_path)))


class Speech2Unit(BaseSpeech2Unit):
    def __init__(self, device):
        super().__init__()
        self.model = (
            S5HubertForSyllableDiscovery.from_pretrained("ryota-komatsu/s5-hubert", cache_dir="./cache/S2U")
            .to(device)
            .eval()
        )
        self.device = device

    def get_unit(self, path: Path) -> list[str]:
        wav, sr = torchaudio.load(path)
        wav = torchaudio.functional.resample(wav, sr, 16000)

        outputs = self.model(wav.to(self.device))
        units = outputs["units"].cpu().tolist()
        units = [str(u) for u in units]
        return units

    def to(self, device):
        self.device = device
        self.model.to(device)


class Unit2Mel(Tacotron2):
    def __init__(self, hparams=None):
        # Partially skip initializing the parent class or initialize with dummy values
        if hparams is None:
            self._initialized = False
        else:
            super().__init__(hparams=hparams)
            self._initialized = True

    def init_model(self, hparams):
        # Set only the necessary properties
        self.hparams = hparams
        # Call the necessary initialization procedures of the parent class
        super().__init__(hparams=hparams)
        self._initialized = True

    def load_checkpoint(self, checkpoint_path: Path):
        self.load_state_dict(torch.load(checkpoint_path)["state_dict"])


class Mel2Speech(Generator):
    def __init__(self, hparams=None):
        # Partially skip initializing the parent class or initialize with dummy values
        if hparams is None:
            self._initialized = False
        else:
            super().__init__(hparams=hparams)
            self._initialized = True

    def init_model(self, hparams):
        # Set only the necessary properties
        self.hparams = hparams
        # Call the necessary initialization procedures of the parent class
        super().__init__(h=hparams)
        self._initialized = True

    def load_checkpoint(self, checkpoint_path: Path, device):
        assert checkpoint_path.exists(), f"Checkpoint file {checkpoint_path} does not exist."
        checkpoint_dict = torch.load(checkpoint_path, map_location=device)
        self.load_state_dict(checkpoint_dict["generator"])


class Unit2Speech(BaseUnit2Speech):
    def __init__(self):
        super().__init__()
        self.u2m = Unit2Mel()
        self.m2s = Mel2Speech()

    def init_u2m(self, hparams):
        self.u2m.init_model(hparams)

    def init_m2s(self, hparams):
        self.m2s.init_model(hparams)

    def inference(self, sequence):
        mel_outputs, mel_outputs_postnet, _, alignments = self.u2m.inference(sequence)
        audio = self.m2s(mel_outputs_postnet)
        return audio

    def load_checkpoint(self, u2m_model_path: Path, m2s_model_path: Path, device: str):
        self.u2m.load_checkpoint(u2m_model_path)
        self.m2s.load_checkpoint(m2s_model_path, device)

    def init_model(self, u2m_hparams, m2s_hparams):
        self.u2m.init_model(u2m_hparams)
        self.m2s.init_model(m2s_hparams)

    def to(self, device):
        self.u2m.to(device)
        self.m2s.to(device)

    def eval(self):
        self.u2m.eval()


class RLUnit(CustomDDPG):
    def __init__(self, **kwargs):
        # Partially skip initializing the parent class or initialize with dummy values
        if not kwargs:  # In the case of an empty dictionary
            self._initialized = False
        else:
            self.init_model(**kwargs)

    def init_model(
        self,
        policy: Union[str, Type[TD3Policy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-4,
        buffer_size: int = 1000,  # 1e6
        learning_starts: int = 0,
        batch_size: int = 32,
        tau: float = 0.005,
        gamma: float = 0,
        train_freq: Union[int, Tuple[int, str]] = (4, "step"),
        gradient_steps: int = 1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[ReplayBuffer] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[torch.device, str] = "auto",
        _init_setup_model: bool = True,
        clip_sentence_embedding: float = 2.5,
    ):
        # Call the necessary initialization procedures of the parent class
        print("env action space", env.action_space)
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            action_noise=action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            optimize_memory_usage=optimize_memory_usage,
            tensorboard_log=tensorboard_log,
            create_eval_env=create_eval_env,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model,
            clip_sentence_embedding=clip_sentence_embedding,
        )
        self._initialized = True


class Action2SpeechWrapper(gym.Wrapper):
    def __init__(self, env, action2speach, d_image_features, d_embed):
        super().__init__(env)
        self.action2speach = action2speach
        self.action_space = gym.spaces.Box(low=-10000, high=10000, shape=(d_image_features + d_embed,))

    def step(self, action: np.ndarray) -> Tuple[dict, float, bool, bool, dict]:
        action_waveform = self.action2speach(action)
        observation, reward, terminated, truncated, info = self.env.step(action_waveform)
        # info["action_waveform"] = action_waveform
        return observation, reward, terminated, truncated, info


class AgentWithUnits(BaseAgentWithUnits):
    def __init__(self, agent_config):
        super().__init__(agent_config)

        self._i2u = Image2Unit()
        self._s2u = Speech2Unit(agent_config["device"])
        self._u2s = Unit2Speech()
        self._rl = RLUnit()

    @property
    def i2u(self) -> BaseImage2Unit:
        """Returns an instance of Image2Unit"""
        return self._i2u

    @property
    def s2u(self) -> BaseSpeech2Unit:
        """Returns an instance of Speech2Unit"""
        return self._s2u

    @property
    def u2s(self) -> BaseUnit2Speech:
        """Returns an instance of Unit2Speech"""
        return self._u2s

    def rl(self) -> BaseRLUnit:
        """Returns an instance of RLUnit"""
        return self._rl

    def load(
        self,
        i2u_init_info: dict,
        i2u_model_path: Path,
        s2u_init_info: dict,
        s2u_model_path: Path,
        u2m_init_info: dict,
        u2m_model_path: Path,
        m2s_init_info: dict,
        m2s_model_path: Path,
    ):
        word_map_path = i2u_init_info["word_map_path"]
        with open(word_map_path) as j:
            word_map = json.load(j)
        self.word_map = word_map
        self.i2u.init_model(word_map=word_map)
        self.i2u.load_checkpoint(i2u_model_path)

        u2m_config = u2m_init_info["config"]
        data_file_paths = u2m_init_info["data_file_paths"]
        m2s_config = m2s_init_info["config"]

        u2m_hparams = create_hparams(u2m_config, data_file_paths=data_file_paths)
        m2s_hparams = m2s_config
        self.u2s.init_model(u2m_hparams, m2s_hparams)
        self.u2s.load_checkpoint(u2m_model_path, m2s_model_path, m2s_init_info["device"])

    def init_rl(self, word_map: dict, device, special_words, beam_size):
        self.device = device
        self.rev_word_map = {v: k for k, v in word_map.items()}
        self.special_words = special_words
        self.i2u.set_beam_size(beam_size)

    @torch.inference_mode()
    def action2speech(self, action: np.ndarray) -> np.ndarray:
        self.u2s.eval()
        action = torch.from_numpy(action).unsqueeze(0).to(self.device)

        unit_seq = self.i2u.infer(action=action)
        words = [self.rev_word_map[idx] for idx in unit_seq if self.rev_word_map[idx] not in self.special_words]
        # print("Generated words:", words)

        sequence = np.array(text_to_sequence(" ".join(words), ["english_cleaners"]))[None, :]
        sequence = torch.autograd.Variable(torch.from_numpy(sequence)).to(self.device).long()

        # print("Sequence shape:", sequence.shape)

        try:
            audio = self.u2s.inference(sequence)
            # print("Generated audio shape:", audio.shape)
            audio = audio.squeeze().cpu().numpy().astype(np.float64)
            audio = resampy.resample(audio, 22050, self.agent_config["U2S"]["u2m"]["audio"]["sampling_rate"])

            # transcript = asr(audio)

        except RuntimeError as e:
            # transcript = ""
            audio = np.zeros(self.agent_config["U2S"]["u2m"]["audio"]["sampling_rate"])
            print(e)
        return audio

    def get_rl_env(self, env: gym.Env, rl_model_config):
        wrapped_env = Action2SpeechWrapper(
            env,
            self.action2speech,
            d_image_features=rl_model_config["d_image_features"],
            d_embed=rl_model_config["i2u_d_embed"],
        )
        return wrapped_env

    def to(self, device):
        self.device = device
        self.i2u.to(self.device)
        self.s2u.to(self.device)
        self.u2s.to(self.device)

    def get_rl_model(self, rl_model_config, env: gym.Env):
        action_noise = NormalActionNoise(
            mean=np.zeros(rl_model_config["d_image_features"] + rl_model_config["i2u_d_embed"]),
            sigma=np.concatenate(
                [
                    np.zeros(rl_model_config["d_image_features"]),
                    rl_model_config["action_noise_sigma"] * np.ones(rl_model_config["i2u_d_embed"]),
                ]
            ),
        )
        policy_kwargs = dict(
            net_arch=dict(
                pi=[[150, 75, 2], [150, 75, rl_model_config["i2u_d_embed"]]],
                qf=[
                    150 + rl_model_config["d_image_features"] + rl_model_config["i2u_d_embed"],
                    150 + rl_model_config["d_image_features"] + rl_model_config["i2u_d_embed"],
                    150 + rl_model_config["d_image_features"] + rl_model_config["i2u_d_embed"],
                    1,
                ],
            ),
        )
        self._rl.init_model(
            policy=CustomTD3Policy,
            env=env,
            learning_rate=rl_model_config["learning_rate"],
            buffer_size=rl_model_config["buffer_size"],
            learning_starts=rl_model_config["learning_starts"],
            batch_size=rl_model_config["batch_size"],
            tau=rl_model_config["tau"],
            gamma=rl_model_config["gamma"],
            train_freq=tuple(rl_model_config["train_freq"]),
            gradient_steps=rl_model_config["gradient_steps"],
            action_noise=action_noise,
            replay_buffer_class=None,
            replay_buffer_kwargs={"handle_timeout_termination": rl_model_config["handle_timeout_termination"]},
            optimize_memory_usage=rl_model_config["optimize_memory_usage"],
            tensorboard_log=rl_model_config["tensorboard_log_dir"],
            create_eval_env=rl_model_config["create_eval_env"],
            policy_kwargs=policy_kwargs,
            verbose=rl_model_config["verbose"],
            seed=rl_model_config["seed"],
            device=self.device,
            _init_setup_model=rl_model_config["_init_setup_model"],
            clip_sentence_embedding=rl_model_config["clip_sentence_embedding"],
        )
        return self._rl
