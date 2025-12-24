import glob
import math
import os
import pickle
import sys
import json
from collections import Counter
from pathlib import Path

import ffmpeg
import librosa
import numpy as np
import pydub
import torch
import yaml
from gtts import gTTS
from tqdm import tqdm
import scipy.io.wavfile as wavfile

from utils import create_input_files
from get_s5hubert_unit import get_unit

from transformers import VitsModel, AutoTokenizer

sys.path.append("../S2U/")
from dataloaders.utils import compute_spectrogram
from run_utils import load_audio_model_and_state
from steps.unit_analysis import DenseAlignment, get_feats_codes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = VitsModel.from_pretrained("kakao-enterprise/vits-ljs").to(device)
tokenizer = AutoTokenizer.from_pretrained("kakao-enterprise/vits-ljs")


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
    if name in scene:
        # Patterns for corners
        pattern_list = patterns[patterns[name]]
    else:
        # Patterns for items
        pattern_list = patterns["else"]

    # Return the specified pattern (using the first pattern as the default if the index is out of range)
    selected_pattern = pattern_list[idx % len(pattern_list)]
    return selected_pattern.format(name=name).replace("_", " ")


def vits(text="", lang="en"):
    if lang != "en":
        raise ValueError("Currently only English (en) is supported")

    # Speech Generation
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model(**inputs).waveform.cpu()

    # Conversion to NumPy arrays and normalization
    audio_data = output.numpy().flatten()
    audio_norm = np.int16(audio_data * 32767)

    # Returns an object for save
    class VitsAudio:
        def __init__(self, audio):
            self.audio = audio

        def save(self, filepath):
            wavfile.write(filepath, 22050, self.audio)

    return VitsAudio(audio_norm)


def make_mp3_wav(data_dir, food_name, text, i):
    os.makedirs(f"{data_dir}/audio/{food_name}", exist_ok=True)
    if not os.path.isfile(f"{data_dir}/audio/{food_name}/{food_name}_{i}.wav"):
        # gTTS(text=text, lang="en").save(f"../../data/I2U/audio/tmp.mp3")
        vits(text=text, lang="en").save(f"{data_dir}/audio/tmp.mp3")
        ffmpeg.run(
            ffmpeg.output(
                ffmpeg.input(f"{data_dir}/audio/tmp.mp3"),
                f"{data_dir}/audio/{food_name}/{food_name}_{i}.wav",
            ),
            quiet=True,
        )


def make_noise_audio(data_dir, food_name, i, j):
    a = pydub.AudioSegment.from_wav(f"{data_dir}/audio/{food_name}/{food_name}_{i}.wav")
    a = a.set_frame_rate(22050)
    y_ori = np.array(a.get_array_of_samples(), dtype="float64")
    sr_ori = a.frame_rate
    snr = 30
    sr, noisy = add_noise(y_ori, sr_ori, snr)
    path = f"{data_dir}/audio/{food_name}/{food_name}_{i}_{j}.wav"
    write_wav(path, sr, noisy)
    return path


def write_wav(f, sr, x):
    y = np.int16(x)
    channels = 2 if (x.ndim == 2 and x.shape[1] == 2) else 1
    song = pydub.AudioSegment(y.tobytes(), frame_rate=sr, sample_width=y.dtype.itemsize, channels=channels)
    song.export(f, format="wav")


def add_noise(y_ori, sr_ori, SNR=30):
    mean = 0
    var = 1
    sigma = var**0.5
    y, sr = y_ori.copy(), sr_ori
    len_seg = sr // 3  # 20 ms
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


# def get_unit(path):
#     code_q3_ali = get_code_ali(audio_model, "quant3", path, device).get_sparse_ali()
#     encoded = []
#     enc_old = -1
#     for _, _, code in code_q3_ali.data:
#         assert enc_old != code
#         encoded.append(str(code))
#         enc_old = code
#     return encoded


# def get_code_ali(audio_model, layer, path, device):
#     y, sr = librosa.load(path, sr=None)
#     mels, nframes = compute_spectrogram(y, sr)
#     mels = mels[:, :nframes]
#     _, _, codes, spf = get_feats_codes(audio_model, layer, mels, device)
#     code_list = codes.detach().cpu().tolist()
#     return DenseAlignment(code_list, spf)


def make_captions(image_paths, mod: int, offset: int, output_data_dir_i2u: str, task_name: str):
    utterance_patterns_path = Path("tasks", task_name, "utterance_patterns.json")
    food_name_old = "init"
    image_captions = []
    for j, path in enumerate(tqdm(image_paths)):
        captions = []
        food_name = path.split("/")[4]
        for i in range(4):
            if food_name != food_name_old:
                text = add_indications(food_name, i, utterance_patterns_path.as_posix())
                make_mp3_wav(output_data_dir_i2u, food_name, text, i)
            captions.append(get_unit(make_noise_audio(output_data_dir_i2u, food_name, i, j % mod + offset)))
        food_name_old = food_name
        image_captions.append(captions)
    return image_captions


def create_input(
    task_name,
    s2u_model_dir,
    image_data_dir,
    output_data_dir,
    captions_per_image=4,
    min_word_freq=1,
    max_len=100,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load S2U
    exp_dir = s2u_model_dir  # "../../models/S2U"
    audio_model = load_audio_model_and_state(exp_dir=exp_dir)
    audio_model = audio_model.to(device)
    audio_model = audio_model.eval()

    # Load images
    train_image_paths = sorted(glob.glob(image_data_dir + "/*/train_number[123]/*.jpg"))
    val_image_paths = sorted(glob.glob(image_data_dir + "/*/test_number[12]/*.jpg"))
    test_image_paths = sorted(glob.glob(image_data_dir + "/*/test_number3/*.jpg"))

    train_image_captions = make_captions(train_image_paths, 90, 0, output_data_dir, task_name)
    val_image_captions = make_captions(val_image_paths, 20, 90)
    test_image_captions = make_captions(test_image_paths, 10, 110)

    word_freq = Counter()
    for captions in train_image_captions:
        for caption in captions:
            word_freq.update(caption)

    os.makedirs(output_data_dir, exist_ok=True)

    with open(os.path.join(output_data_dir, "train_image_paths.pickle"), "wb") as f:
        pickle.dump(train_image_paths, f)
    with open(os.path.join(output_data_dir, "train_image_captions.pickle"), "wb") as f:
        pickle.dump(train_image_captions, f)
    with open(os.path.join(output_data_dir, "val_image_paths.pickle"), "wb") as f:
        pickle.dump(val_image_paths, f)
    with open(os.path.join(output_data_dir, "val_image_captions.pickle"), "wb") as f:
        pickle.dump(val_image_captions, f)
    with open(os.path.join(output_data_dir, "test_image_paths.pickle"), "wb") as f:
        pickle.dump(test_image_paths, f)
    with open(os.path.join(output_data_dir, "test_image_captions.pickle"), "wb") as f:
        pickle.dump(test_image_captions, f)
    with open(os.path.join(output_data_dir, "word_freq.pickle"), "wb") as f:
        pickle.dump(word_freq, f)

    # Create input files (along with word map)
    create_input_files(
        dataset=task_name,
        captions_per_image=captions_per_image,
        min_word_freq=min_word_freq,
        output_folder=output_data_dir,
        max_len=max_len,
    )


if __name__ == "__main__":
    with open("../../conf/spolacq3.yaml") as y:
        config = yaml.safe_load(y)
    create_input(
        config["task_name"],
        "../../models/S2U",
        "../../data/dataset",
        "../../data/I2U",
        "food",
        4,
        1,
        "../../data/I2U",
        100,
    )
