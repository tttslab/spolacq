from collections import Counter
import glob
import math
import os
import pickle
import sys

import ffmpeg
from gtts import gTTS
import librosa
import numpy as np
import pydub
import torch
from tqdm import tqdm
import yaml

from utils import create_input_files

sys.path.append("../S2U/")
from dataloaders.utils import compute_spectrogram
from run_utils import load_audio_model_and_state
from steps.unit_analysis import get_feats_codes, DenseAlignment


obj2color = {
    'apple': 'red',
    'banana': 'yellow',
    'carrot': 'orange',
    'cherry': 'black',
    'cucumber': 'green',
    'egg': 'chicken',
    'eggplant': 'purple',
    'green_pepper': 'green',
    'hyacinth_bean': 'green',
    'kiwi_fruit': 'brown',
    'lemon': 'yellow',
    'onion': 'yellow',
    'orange': 'orange',
    'potato': 'brown',
    'sliced_bread': 'yellow',
    'small_cabbage': 'green',
    'strawberry': 'red',
    'sweet_potato': 'brown',
    'tomato': 'red',
    'white_radish': 'white',
}


def add_indications(name, idx):
    preposition = 'an' if name[0] in ['a', 'o', 'e'] else 'a'
    color = obj2color[name]
    preposition2 = 'an' if color[0] in ['a', 'o', 'e'] else 'a'
    ll = [name, f'{preposition} {name}',
          f'{preposition2} {color} {name}', f"i want {preposition} {name}"]
    return ll[idx].replace("_", " ")


def make_mp3_wav(food_name, text, i):
    os.makedirs(f"../../data/I2U/audio/{food_name}", exist_ok=True)
    if not os.path.isfile(f'../../data/I2U/audio/{food_name}/{food_name}_{i}.wav'):
        gTTS(text=text, lang='en').save(f'../../data/I2U/audio/tmp.mp3')
        ffmpeg.run(
            ffmpeg.output(ffmpeg.input("../../data/I2U/audio/tmp.mp3"),
                          f"../../data/I2U/audio/{food_name}/{food_name}_{i}.wav"),
            quiet=True,
        )


def make_noise_audio(food_name, i, j):
    a = pydub.AudioSegment.from_wav(f"../../data/I2U/audio/{food_name}/{food_name}_{i}.wav")
    a = a.set_frame_rate(22050)
    y_ori = np.array(a.get_array_of_samples(), dtype="float64")
    sr_ori = a.frame_rate
    snr = 30
    sr, noisy = add_noise(y_ori, sr_ori, snr)
    path = f"../../data/I2U/audio/{food_name}/{food_name}_{i}_{j}.wav"
    write_wav(path, sr, noisy)
    return path


def write_wav(f, sr, x):
    y = np.int16(x)
    channels = 2 if (x.ndim == 2 and x.shape[1] == 2) else 1
    song = pydub.AudioSegment(
        y.tobytes(), frame_rate=sr, sample_width=y.dtype.itemsize, channels=channels)
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
        y_seg = np.array(y[start:end], dtype='float64')
        n_seg = np.array(noise[start:end], dtype='float64')
        sum_s = np.sum(y_seg ** 2)
        sum_n = np.sum(n_seg ** 2)
        w = np.sqrt(sum_s / (sum_n * pow(10, SNR / 10)))  # SNR: 30db
        #print(sum_s, np.sqrt(sum_s/(sum_n * pow(10, -10 / 10))))
        n_seg = w * n_seg
        noise[start:end] = n_seg
        y[start:end] = y_seg
    noisy = noise + y
    snr = 10 * np.log10(np.sum(y ** 2) / np.sum(noise ** 2))
    return sr, noisy


def get_unit(path):
    code_q3_ali = get_code_ali(audio_model, 'quant3', path, device).get_sparse_ali()
    encoded = []
    enc_old = -1
    for _, _, code in code_q3_ali.data:
        assert enc_old != code
        encoded.append(str(code))
        enc_old = code
    return encoded


def get_code_ali(audio_model, layer, path, device):
    y, sr = librosa.load(path, sr=None)
    mels, nframes = compute_spectrogram(y, sr)
    mels = mels[:, :nframes]
    _, _, codes, spf = get_feats_codes(audio_model, layer, mels, device)
    code_list = codes.detach().cpu().tolist()
    return DenseAlignment(code_list, spf)


def make_captions(image_paths, mod: int, offset: int):
    food_name_old = "init"
    image_captions = []
    for j, path in enumerate(tqdm(image_paths)):
        captions = []
        food_name = path.split("/")[4]
        for i in range(4):
            if food_name != food_name_old:
                text = add_indications(food_name, i)
                make_mp3_wav(food_name, text, i)
            captions.append(get_unit(make_noise_audio(food_name, i, j%mod+offset)))
        food_name_old = food_name
        image_captions.append(captions)
    return image_captions


if __name__ == "__main__":
    with open("../../conf/spolacq3.yaml") as y:
        config = yaml.safe_load(y)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load S2U
    exp_dir = "../../models/S2U"
    audio_model = load_audio_model_and_state(exp_dir=exp_dir)
    audio_model = audio_model.to(device)
    audio_model = audio_model.eval()

    # Load images
    train_image_paths = sorted(glob.glob("../../data/dataset/*/train_number[123]/*.jpg"))
    val_image_paths = sorted(glob.glob("../../data/dataset/*/test_number[12]/*.jpg"))
    test_image_paths = sorted(glob.glob("../../data/dataset/*/test_number3/*.jpg"))

    train_image_captions = make_captions(train_image_paths, 90, 0)
    val_image_captions = make_captions(val_image_paths, 20, 90)
    test_image_captions = make_captions(test_image_paths, 10, 110)

    word_freq = Counter()
    for captions in train_image_captions:
        for caption in captions:
            word_freq.update(caption)
    
    data_folder = os.path.join(os.path.dirname(__file__), "../..", config["I2U"]["data_folder"])
    os.makedirs(data_folder, exist_ok=True)
    
    with open(os.path.join(data_folder, "train_image_paths.pickle"), "wb") as f:
        pickle.dump(train_image_paths, f)
    with open(os.path.join(data_folder, "train_image_captions.pickle"), "wb") as f:
        pickle.dump(train_image_captions, f)
    with open(os.path.join(data_folder, "val_image_paths.pickle"), "wb") as f:
        pickle.dump(val_image_paths, f)
    with open(os.path.join(data_folder, "val_image_captions.pickle"), "wb") as f:
        pickle.dump(val_image_captions, f)
    with open(os.path.join(data_folder, "test_image_paths.pickle"), "wb") as f:
        pickle.dump(test_image_paths, f)
    with open(os.path.join(data_folder, "test_image_captions.pickle"), "wb") as f:
        pickle.dump(test_image_captions, f)
    with open(os.path.join(data_folder, "word_freq.pickle"), "wb") as f:
        pickle.dump(word_freq, f)

    # Create input files (along with word map)
    create_input_files(dataset="food",
                       captions_per_image=4,
                       min_word_freq=1,
                       output_folder=data_folder,
                       max_len=100)
