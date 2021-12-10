import warnings
import argparse
import glob
import os
import pickle
import shutil
from pathlib import Path

import librosa
import numpy as np
import pydub
import resampy
from gtts import gTTS
from pydub import AudioSegment
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument("data_path", type=str)
args = parser.parse_args()

hop_sec = 0.010
win_len_sec = 0.025

def enc(a, sr):  # wave -> log spectrogram
    a = librosa.stft(a, n_fft=int(win_len_sec * sr),
                     hop_length=int(hop_sec * sr))
    return a


def dec(a, sr):  # log spectrogram -> wave
    a = librosa.istft(a, hop_length=int(hop_sec * sr),
                      win_length=int(win_len_sec * sr))
    return a


def mp3file_to_examples(mp3_file):
    a, sr = librosa.load(mp3_file)
    resr = 16000
    a = resampy.resample(a, sr, resr)
    feature = enc(a, resr)
    return feature


def extract_and_save_from_mp3file(io):
    in_path, out_path = io
    if Path(out_path).is_file():
        return
    try:
        feature = mp3file_to_examples(in_path)
        with open(out_path, "wb") as f:
            pickle.dump(feature, f)
    except Exception as e:
        print(e)
        with open('failedlist.txt', 'a') as f:
            f.write('{}\n'.format(in_path))


def gtts_wrapper(descript: str, file_name: str):
    if not os.path.exists(file_name):
        tts = gTTS(text=descript, lang='en')
        tts.save(file_name)
    wav_name = file_name.replace('mp3', 'wav')
    if not os.path.exists(wav_name):
        sound = AudioSegment.from_mp3(file_name)
        sound.export(wav_name, format="wav")
    pkl_name = file_name.replace('mp3', 'pkl')
    if not os.path.exists(pkl_name):
        extract_and_save_from_mp3file([file_name, pkl_name])


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
          f'{preposition2} {color} {name}', f"it's {preposition} {name}"]
    return ll[idx].replace("_", " ")


# setteing the path
data_path = args.data_path
dir_path = f'{data_path}'
parent_path = os.path.normpath(args.data_path+'/..')

obj_name_list = [i.split('/')[-1] for i in glob.glob(f'{dir_path}/*')]
obj_name_list.sort()

for obj_name in obj_name_list:
    os.makedirs(f'{data_path}/{obj_name}/description/clean', exist_ok=True)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for obj_name in tqdm(obj_name_list):
        for idx in range(4):
            descript = add_indications(obj_name, idx)
            short = descript.replace(' ', '')
            file_name = f'{data_path}/{obj_name}/description/clean/{short}.mp3'
            gtts_wrapper(descript, file_name)
    
    file_name = f'{parent_path}/no.mp3'
    gtts_wrapper('no', file_name)