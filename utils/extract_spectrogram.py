import sys
import argparse
import pickle
from pathlib import Path
import time

import numpy as np
import librosa
import resampy

def enc(a, sr):
    win_len_sec = 0.025
    hop_sec = 0.010
    ret = librosa.stft(a, n_fft=int(win_len_sec * sr),
                        hop_length=int(hop_sec * sr))
    #log_pad = 0.010
    #ret = np.log(np.abs(ret) + log_pad).T
    return ret

def extract_and_save_from_mp3file(io):
    in_path, out_path = io
    if Path(out_path).is_file(): return
    try:
        feature = mp3file_to_examples(in_path)
        with open(out_path, "wb") as f:
            pickle.dump(feature, f)
    except Exception as e:
        print("Failed: {}".format(in_path), file=sys.stderr)

def mp3file_to_examples(mp3_file):
    a, sr = librosa.load(mp3_file)
    resr = 16000
    a = resampy.resample(a, sr, resr)
    feature = enc(a, resr)
    return feature

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_list", type=str)
    parser.add_argument("save_list", type=str)
    args = parser.parse_args()

    audio_list = []
    with open(args.audio_list, "r") as f:
        for p in f:
            audio_list.append(p.strip())

    save_list = []
    with open(args.save_list, "r") as f:
        for p in f:
            save_list.append(p.strip())
        
    plist = list(zip(audio_list, save_list))

    from multiprocessing import Pool
    import multiprocessing as multi
    import sys

    pool = Pool(multi.cpu_count())
    for i, _ in enumerate(pool.imap_unordered(extract_and_save_from_mp3file, plist)):
        if i % 1000 == 0: print(i, flush=True)
    pool.close()