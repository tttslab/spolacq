import json
import os
import sys
from glob import glob

import numpy as np
import resampy
import torch
import yaml
from stable_baselines3.common.noise import NormalActionNoise

from asr_api import ASR
from gym_api import SpoLacq3
from sb3_api import CustomDDPG, CustomTD3Policy

sys.path.append("../tools/I2U")
from models_i2u import ImageToUnit

sys.path.append("../tools/U2S")
from hparams import create_hparams
from text import text_to_sequence
from train import load_model

sys.path.append("../tools/hifi-gan")
from env import AttrDict
from inference import load_checkpoint
from models_hifi_gan import Generator


def make_action2text(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # I2U
    word_map_path = os.path.join(os.path.dirname(__file__), "..", config["I2U"]["word_map"])
    image_to_unit_model_path = os.path.join(os.path.dirname(__file__), "..", config["I2U"]["model_path"])

    with open(word_map_path) as j:
        word_map = json.load(j)
    rev_word_map = {v: k for k, v in word_map.items()}
    special_words = {"<unk>", "<start>", "<end>", "<pad>"}

    image_to_unit = ImageToUnit(word_map)
    image_to_unit.load_state_dict(torch.load(image_to_unit_model_path))
    image_to_unit.to(device).eval()

    # U2S
    checkpoint_path = os.path.join(os.path.dirname(__file__), "..", config["U2S"]["tacotron2"])
    hparams = create_hparams()
    tacotron2 = load_model(hparams)
    tacotron2.load_state_dict(torch.load(checkpoint_path)["state_dict"])
    tacotron2.to(device).eval()

    # hifi-gan
    hifi_gan_config_path = os.path.join(os.path.dirname(__file__), "..", config["HiFi_GAN"]["config"])
    hifi_gan_checkpoint_path = os.path.join(os.path.dirname(__file__), "..", config["HiFi_GAN"]["checkpoint"])

    with open(hifi_gan_config_path) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)
    generator = Generator(h).to(device)
    state_dict_g = load_checkpoint(hifi_gan_checkpoint_path, device)
    generator.load_state_dict(state_dict_g["generator"])

    # ASR
    asr_dir = os.path.join(os.path.dirname(__file__), "..", config["ASR"]["dir"])
    asr = ASR(asr_dir)

    @torch.inference_mode()
    def action2text(action):
        action = torch.from_numpy(action).unsqueeze(0).to(device)

        unit_seq = image_to_unit.infer(action=action, beam_size=config["I2U"]["beam_size"])
        words = [rev_word_map[idx] for idx in unit_seq if rev_word_map[idx] not in special_words]

        sequence = np.array(text_to_sequence(" ".join(words), ["english_cleaners"]))[None, :]
        sequence = torch.autograd.Variable(torch.from_numpy(sequence)).to(device).long()

        try:
            _, mel_outputs_postnet, _, _ = tacotron2.inference(sequence)

            audio = generator(mel_outputs_postnet)
            audio = audio.squeeze().cpu().numpy().astype(np.float64)
            audio = resampy.resample(audio, 22050, 16000)

            transcript = asr(audio)

        except RuntimeError as e:
            transcript = ""
            print(e, flush=True)

        return transcript

    return action2text


if __name__ == "__main__":
    with open("../conf/spolacq3.yaml") as y:
        config = yaml.safe_load(y)

    action_noise = NormalActionNoise(
        mean=np.zeros(768 + config["I2U"]["d_embed"]),
        sigma=np.concatenate(
            [
                np.zeros(768),
                config["RL"]["action_noise_sigma"] * np.ones(config["I2U"]["d_embed"]),
            ]
        ),
    )

    action2text = make_action2text(config)

    env = SpoLacq3(
        glob("../data/dataset/*/train_number[123]/*.jpg"),
        d_image_features=768,
        d_embed=config["I2U"]["d_embed"],
        action2text=action2text,
    )

    eval_env = SpoLacq3(
        glob("../data/dataset/*/test_number[12]/*.jpg"),
        d_image_features=768,
        d_embed=config["I2U"]["d_embed"],
        action2text=action2text,
    )

    model = CustomDDPG(
        CustomTD3Policy,
        env,
        learning_rate=config["RL"]["learning_rate"],
        buffer_size=config["RL"]["buffer_size"],
        learning_starts=config["RL"]["learning_starts"],
        batch_size=config["RL"]["batch_size"],
        action_noise=action_noise,
        replay_buffer_kwargs=dict(handle_timeout_termination=False),
        tensorboard_log=os.path.join(os.path.dirname(__file__), "..", config["RL"]["tensorboard_log"]),
        policy_kwargs=dict(
            net_arch=dict(
                pi=[[150, 75, 2], [150, 75, config["I2U"]["d_embed"]]],
                qf=[
                    150 + 768 + config["I2U"]["d_embed"],
                    150 + 768 + config["I2U"]["d_embed"],
                    150 + 768 + config["I2U"]["d_embed"],
                    1,
                ],
            ),
        ),
        clip_sentence_embedding=config["RL"]["clip_sentence_embedding"],
    )

    model.learn(
        total_timesteps=config["RL"]["total_timesteps"],
        eval_env=eval_env,
        eval_freq=config["RL"]["eval_freq"],
        n_eval_episodes=config["RL"]["n_eval_episodes"],
        eval_log_path=os.path.join(os.path.dirname(__file__), "..", config["RL"]["eval_log_path"]),
    )
