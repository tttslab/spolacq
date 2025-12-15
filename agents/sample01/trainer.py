from typing import Callable
from pathlib import Path
import json
import os
import re
import glob

# from .tools.I2U.train_i2u import main as train_i2u_main
import gymnasium as gym
import numpy as np
from stable_baselines3.common.callbacks import CallbackList, EvalCallback
from agents.base_agent_with_units import BaseAgentWithUnits
from agents.base_trainer import BaseTrainerWithUnits

from .tools.I2U.create_input_files2 import create_i2u_dataset
from .tools.I2U.train_i2u2 import train_i2u as train_i2u_

from .tools.U2S.preprocess2 import generate_u2s_dataset
from .tools.U2S.train2 import create_hparams, train_u2s as train_u2s_

from .tools.hifi_gan.env import AttrDict


# from food_task.py
class Action2SpeechWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, action2speach: Callable[[np.ndarray], np.ndarray]):
        super().__init__(env)
        self.action2speach = action2speach
        self.action_space = self.env.action_space

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        action_waveform = self.action2speach(action)
        observation, reward, terminated, truncated, info = self.env.step(action_waveform)
        # info["action_waveform"] = action_waveform
        return observation, reward, terminated, truncated, info


def get_latest_u2m_model_path(model_dir):
    model_dir = Path(model_dir)
    checkpoints = list(model_dir.glob("checkpoint_*"))
    if not checkpoints:
        return None

    # Extract the iteration values and find the file containing the maximum value.
    latest = max(checkpoints, key=lambda x: int(re.search(r"checkpoint_(\d+)", x.name).group(1)))
    return latest


class Trainer(BaseTrainerWithUnits):
    def __init__(self):
        super().__init__()

    def pretrain(
        self,
        agent: BaseAgentWithUnits,
        image_paths,
        audio_paths,
        base_config,
        pretrain_config,
        agent_dir: Path,
        agent_experiment_dir: Path,
    ):
        # Train the I2U model
        i2u_info = self.train_i2u(
            agent=agent,
            image_paths=image_paths,
            audio_paths=audio_paths,
            base_config=base_config,
            pretrain_config=pretrain_config,
            agent_dir=agent_dir,
            agent_experiment_dir=agent_experiment_dir,
        )

        # Train the U2S model
        u2s_info = self.train_u2s(
            agent=agent,
            image_paths=image_paths,
            audio_paths=audio_paths,
            base_config=base_config,
            pretrain_config=pretrain_config,
            agent_dir=agent_dir,
            agent_experiment_dir=agent_experiment_dir,
        )

        info = {
            "i2u": {
                "word_map_path": i2u_info["word_map_path"],
                "i2u_model_path": i2u_info["output_model_path"],
            },
            "u2s": u2s_info,
        }
        return info

    def generate_i2u_dataset(
        self,
        agent: BaseAgentWithUnits,
        image_paths,
        audio_paths,
        base_config,
        pretrain_config,
        agent_experiment_dir: Path,
    ):
        # Generate the I2U dataset
        i2u_data_dir = agent_experiment_dir / "data" / "I2U"
        data_info = create_i2u_dataset(
            agent,
            task_name=base_config["task_name"],
            image_paths=image_paths,
            audio_paths=audio_paths,
            output_data_dir=i2u_data_dir,
            captions_per_image=pretrain_config["I2U"]["dataset"]["captions_per_image"],
            min_word_freq=pretrain_config["I2U"]["dataset"]["min_word_freq"],
            max_len=pretrain_config["I2U"]["dataset"]["max_len"],
        )

        return data_info

    def train_i2u(
        self,
        agent: BaseAgentWithUnits,
        image_paths,
        audio_paths,
        base_config,
        pretrain_config,
        agent_dir: Path,
        agent_experiment_dir: Path,
    ):
        data_info = self.generate_i2u_dataset(
            agent,
            image_paths=image_paths,
            audio_paths=audio_paths,
            base_config=base_config,
            pretrain_config=pretrain_config,
            agent_experiment_dir=agent_experiment_dir,
        )

        i2u_pretrain_config = pretrain_config["I2U"]
        word_map_path = data_info["word_map_path"]
        with open(word_map_path) as j:
            word_map = json.load(j)

        agent.i2u.init_model(word_map=word_map)

        i2u_data_dir = data_info["i2u_data_dir"]

        output_model_dir = agent_experiment_dir / "models" / "I2U"
        output_model_dir.mkdir(parents=True, exist_ok=True)

        output_model_path = output_model_dir / "i2u_best_model.pth"

        if i2u_pretrain_config["train_i2u"]:
            train_i2u_(
                model=agent.i2u,
                i2u_pretrain_config=i2u_pretrain_config,
                input_data_folder=i2u_data_dir,
                data_filename=data_info["data_filename"],
                output_model_path=output_model_path,
                word_map=word_map,
            )
        return {
            "word_map_path": word_map_path,
            "output_model_path": output_model_path,
        }

    def generate_u2m_dataset(
        self,
        agent: BaseAgentWithUnits,
        audio_paths,
        base_config,
        u2m_pretrain_config,
        agent_experiment_dir: Path,
    ):
        u2s_pretrain_data_dir = agent_experiment_dir / "data" / "U2S"
        data_info = generate_u2s_dataset(
            u2m_pretrain_config,
            audio_paths=audio_paths,
            agent=agent,
            output_data_dir=u2s_pretrain_data_dir,
        )
        return data_info

    def train_u2s(
        self,
        agent: BaseAgentWithUnits,
        image_paths,
        audio_paths,
        base_config,
        pretrain_config,
        agent_dir: Path,
        agent_experiment_dir: Path,
    ):
        # Generate the U2S dataset
        u2m_pretrain_config = pretrain_config["U2S"]["u2m"]
        data_info = self.generate_u2m_dataset(
            agent=agent,
            audio_paths=audio_paths,
            base_config=base_config,
            u2m_pretrain_config=u2m_pretrain_config,
            agent_experiment_dir=agent_experiment_dir,
        )

        # Train the U2M model
        hparams = create_hparams(u2m_pretrain_config, data_file_paths=data_info)
        agent.u2s.init_u2m(hparams=hparams)

        u2m_model_dir = agent_experiment_dir / "models" / "U2S"
        train_u2m = u2m_pretrain_config["train_u2m"]

        if train_u2m:
            train_u2s_(
                model=agent.u2s.u2m,
                u2s_pretrain_config=u2m_pretrain_config,
                model_output_dir=u2m_model_dir,
                hparams=hparams,
            )

        m2s_pretrain_config = pretrain_config["U2S"]["m2s"]
        hparams = AttrDict(m2s_pretrain_config)
        agent.u2s.m2s.init_model(hparams=hparams)

        return {
            "u2m": {
                "u2m_model_dir": u2m_model_dir,
                "u2m_pretrain_config": u2m_pretrain_config,
                "data_info": data_info,
            },
            "m2s": {
                "m2s_pretrain_config": hparams,
            },
        }


    def train_rl(
        self,
        agent: BaseAgentWithUnits,
        env_train,
        env_test,
        base_config,
        rl_env_config,
        rl_agent_config,
        pretrain_info,
        agent_dir: Path,
        agent_experiment_dir: Path,
    ):
        i2u_init_info = {
            "word_map_path": pretrain_info["i2u"]["word_map_path"],
        }
        i2u_model_path = pretrain_info["i2u"]["i2u_model_path"]

        u2m_init_info = {
            "config": pretrain_info["u2s"]["u2m"]["u2m_pretrain_config"],
            "data_file_paths": pretrain_info["u2s"]["u2m"]["data_info"],
        }
        u2m_model_path = get_latest_u2m_model_path(pretrain_info["u2s"]["u2m"]["u2m_model_dir"])

        m2s_init_info = {
            "config": pretrain_info["u2s"]["m2s"]["m2s_pretrain_config"],
            "device": rl_agent_config["U2S"]["m2s"]["device"],
        }
        m2s_model_path = agent_dir / "models" / "U2S" / "m2s" / rl_agent_config["U2S"]["m2s"]["model_file_name"]

        # Load a pre-trained model
        agent.load(
            i2u_init_info=i2u_init_info,
            i2u_model_path=i2u_model_path,
            s2u_init_info=None,
            s2u_model_path=None,
            u2m_init_info=u2m_init_info,
            u2m_model_path=u2m_model_path,
            m2s_init_info=m2s_init_info,
            m2s_model_path=m2s_model_path,
        )

        # Initialize the RL model
        word_map_path = pretrain_info["i2u"]["word_map_path"]
        with open(word_map_path) as j:
            word_map = json.load(j)

        agent.init_rl(
            word_map=word_map,
            device=rl_agent_config["device"],
            special_words=rl_agent_config["U2S"]["m2s"]["special_words"],
            beam_size=rl_agent_config["I2U"]["beam_size"],
        )

        rl_model_config = rl_agent_config["RL"]
        rl_model_config["i2u_d_embed"] = rl_agent_config["I2U"]["d_embed"]
        rl_model_config["tensorboard_log_dir"] = agent_experiment_dir / rl_model_config["tensorboard_log"]

        # Wrap the environment
        env_train = agent.get_rl_env(env_train, rl_model_config)
        env_test = agent.get_rl_env(env_test, rl_model_config)

        agent.to(rl_agent_config["device"])

        # Get the RL model
        model = agent.get_rl_model(rl_model_config, env_train)

        eval_callback = EvalCallback(
            eval_env=env_test,
            eval_freq=rl_env_config["eval_freq"],  # Evaluate at the frequency specified in the config
            n_eval_episodes=rl_env_config[
                "n_eval_episodes"
            ],  # Evaluate based on the number of episodes specified in the config
            log_path=agent_experiment_dir / "models" / rl_env_config["eval_log_path"],  # Log Save Location
            best_model_save_path=agent_experiment_dir
            / "models"
            / rl_env_config["eval_log_path"]
            / "best_model",  # Best Model Save Location
            deterministic=rl_env_config["deterministic"],  # Deterministic actions are used during evaluation.
            render=rl_env_config["render"],  # Rendering is off
        )

        model.learn(
            total_timesteps=rl_env_config["total_timesteps"],
            callback=eval_callback,
            tb_log_name=rl_agent_config["name"],
            reset_num_timesteps=True,
            progress_bar=False,
        )
