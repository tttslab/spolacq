import torch
import json
import os
from pathlib import Path
import glob
import pydub
import math
import torchaudio

from kokoro import KPipeline

import scipy.io.wavfile as wavfile
from tqdm import tqdm
import ffmpeg
import numpy as np
from dataclasses import dataclass


from abc import ABC, abstractmethod

from .create_image_dataset import get_image_paths


@dataclass
class Audio(object):
    audio: np.ndarray
    sr: int = 22050

    def save(self, filepath):
        wavfile.write(filepath, self.sr, self.audio)


class AudioModel(ABC):
    @abstractmethod
    def generate(self, text: str, config) -> Audio:
        pass



    
    
class AudioModelKokoro(AudioModel):
    def __init__(self, device):
        self.pipeline = KPipeline(lang_code="a")

    def generate(self, text: str, config):
        generator = self.pipeline(text, voice = config["voice"])
        gs, ps, audio = next(generator)
        audio_data = audio.numpy().flatten()
        audio_tensor = torch.from_numpy(audio_data).unsqueeze(0)
        resampler = torchaudio.transforms.Resample(orig_freq=24000, new_freq=22050)
        resampled = resampler(audio_tensor).squeeze(0)
        audio_norm = np.int16(resampled.numpy() * 32767)
        return Audio(audio_norm)


def get_speech_synthesizer(device, model_name: str = "kokoro") -> AudioModel:
    """Get a speech synthesizer"""
    if model_name == "kokoro":
        return AudioModelKokoro(device=device)


def make_mp3_wav(
    data_dir: Path, item_name: str, text: str, i: int, model: AudioModel, config=None, overwrite=False, is_print=False
):
    item_dir = data_dir / item_name
    item_dir.mkdir(parents=True, exist_ok=True)
    output_path = item_dir / f"{item_name}_{i}.wav"

    if output_path.exists() and not overwrite:
        if is_print:
            print(f"{output_path} already exists.")
    else:
        tmp_path = data_dir / "tmp.mp3"
        model.generate(text=text, config=config).save(str(tmp_path))
        ffmpeg.run(
            ffmpeg.output(
                ffmpeg.input(str(tmp_path)),
                str(output_path),
            ),
            quiet=True,
        )

    return output_path


def add_indications(name, idx, utterance_patterns_path):
    """
    Generate various natural utterances for both corners and food items
    Args:
        name: item name
        idx: index of the utterance pattern to use
    Returns:
        string: generated utterance
    """
    # Load speech patterns and corner lists from JSON files
    with open(utterance_patterns_path, "r") as f:
        patterns = json.load(f)

    # Get a list of corners from a JSON file
    scene = patterns["scene"]

    # Check whether the name is included in the corner list
    cleaned_name = name.replace("_", " ").lower()
    cleaned_scene = {k.replace("_", " ").lower(): v for k, v in scene.items()}
    if cleaned_name in cleaned_scene:
        # Patterns for corners
        pattern_list = patterns[cleaned_scene[cleaned_name]]
    else:
        # Pattern for items
        pattern_list = patterns["else"]

    # Return the specified pattern (using the first pattern as the default if the index is out of range)
    selected_pattern = pattern_list[idx % len(pattern_list)]
    return selected_pattern.format(name=name).replace("_", " ")


def add_noise(y_ori, sr_ori, SNR=30):
    mean = 0
    var = 1
    sigma = var**0.5
    y, sr = y_ori.copy(), sr_ori
    len_seg = sr // 50  # 20 ms
    len_y = len(y)
    M = math.ceil(len_y / len_seg)
    noise = np.random.normal(mean, sigma, len(y))
    for m in range(M - 1):
        start = m * len_seg
        end = min(len_y, (m + 1) * len_seg)
        # Avoid too short segments.
        if len_y - end > 0 and len_y - end < len_seg / 2:
            end = len_y
        y_seg = np.array(y[start:end], dtype="float64")
        n_seg = np.array(noise[start:end], dtype="float64")
        sum_s = np.sum(y_seg**2)
        sum_n = np.sum(n_seg**2)
        w = np.sqrt(sum_s / (sum_n * pow(10, SNR / 10)))  # SNR: 30db
        # print(sum_s, np.sqrt(sum_s/(sum_n * pow(10, -10 / 10))))
        n_seg = w * n_seg
        noise[start:end] = n_seg
        y[start:end] = y_seg
    noisy = noise + y
    snr = 10 * np.log10(np.sum(y**2) / np.sum(noise**2))
    return sr, noisy


def write_wav(f, sr, x):
    y = np.int16(x)
    channels = 2 if (x.ndim == 2 and x.shape[1] == 2) else 1
    song = pydub.AudioSegment(y.tobytes(), frame_rate=sr, sample_width=y.dtype.itemsize, channels=channels)
    song.export(f, format="wav")


def make_noise_audio(audio_data_dir: Path, food_name: str, i: int, j: int, SNR=30, overwrite=False, is_print=False):
    output_path = audio_data_dir / food_name / f"{food_name}_{i}_{j}.wav"
    if output_path.exists() and not overwrite:
        if is_print:
            print(f"{output_path} already exists.")
    else:
        input_path = audio_data_dir / food_name / f"{food_name}_{i}.wav"
        a = pydub.AudioSegment.from_wav(str(input_path))
        a = a.set_frame_rate(22050)
        y_ori = np.array(a.get_array_of_samples(), dtype="float64")
        sr_ori = a.frame_rate
        sr, noisy = add_noise(y_ori, sr_ori, SNR)
        write_wav(str(output_path), sr, noisy)
    return output_path


def make_audio(
    image_paths,
    audio_data_dir: Path,
    task_name: str,
    model,
    audio_config,
    mod=90,
    offset=0,
    noise_num=4,
    overwrite=False,
    is_print=False,
):
    utterance_patterns_path = Path("tasks") / task_name / "utterance_patterns.json"
    processed_items = set()
    audio_data_paths = []

    if is_print:
        print("Step 1: Generating base audio files...")

    for path in image_paths:
        item_name = Path(path).parts[-3]
        if item_name not in processed_items:
            for i in range(noise_num):
                text = add_indications(item_name, i, utterance_patterns_path)
                assert item_name.replace("_", " ") in text
                original_path = make_mp3_wav(
                    audio_data_dir,
                    item_name,
                    text,
                    i,
                    model,
                    config=audio_config,
                    overwrite=overwrite,
                    is_print=is_print,
                )
                if is_print:
                    print(f"Generated base audio: {original_path}")
            processed_items.add(item_name)

    if is_print:
        print("Step 2: Generating noise variations...")

    for j, path in enumerate(tqdm(image_paths)):
        item_name = Path(path).parts[-3]
        item_audio_paths = []
        for i in range(noise_num):
            path = make_noise_audio(
                audio_data_dir,
                item_name,
                i,
                j % mod + offset,
                audio_config["SNR"],
                overwrite=overwrite,
                is_print=is_print,
            )
            item_audio_paths.append(path)
        audio_data_paths.append(item_audio_paths)

        if is_print:
            print(f"Generated noise variations for {item_name}: {len(item_audio_paths)} files")

    if is_print:
        print(f"Audio generation completed. Total items processed: {len(processed_items)}")

    return audio_data_paths


def create_audio_dataset(
    task_name: str,
    image_data_dir: Path,
    audio_data_dir: Path,
    audio_config,
    device,
    train_images_per_folder: int = 30,
    test_images_per_folder: int = 10,
):
    """
    Create audio dataset for the specified task.
    Args:
        task_name (str): Name of the task
        image_data_dir (Path): Directory containing image data
        audio_data_dir (Path): Directory for audio output
        audio_config: Configuration for audio generation
        device: Device to use for model
        train_images_per_folder: Number of training images per folder
        test_images_per_folder: Number of test images per folder
    """

    train_image_paths, val_image_paths, test_image_paths = get_image_paths(image_data_dir)

    # Load the audio model
    model = get_speech_synthesizer(model_name=audio_config["model_name"], device=device)

    # Create audio dataset
    # 1/3
    train_audio_paths = make_audio(
        train_image_paths,
        audio_data_dir,
        task_name,
        model,
        audio_config,
        train_images_per_folder * 3,
        0,
        audio_config["noise_num"],
        audio_config["overwrite_audio_dataset"],
    )
    # 2/3
    val_audio_paths = make_audio(
        val_image_paths,
        audio_data_dir,
        task_name,
        model,
        audio_config,
        test_images_per_folder * 2,
        train_images_per_folder * 3,
        audio_config["noise_num"],
        audio_config["overwrite_audio_dataset"],
    )
    # 3/3
    test_audio_paths = make_audio(
        test_image_paths,
        audio_data_dir,
        task_name,
        model,
        audio_config,
        test_images_per_folder,
        (train_images_per_folder * 3 + test_images_per_folder * 2),
        audio_config["noise_num"],
        audio_config["overwrite_audio_dataset"],
    )

    return {
        "train": train_audio_paths,
        "val": val_audio_paths,
        "test": test_audio_paths,
    }
