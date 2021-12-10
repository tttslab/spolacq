"""
    convert to mfcc
"""

import argparse
import librosa
from pathlib import Path
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("--wavlist", type=str, default="../tools/syllables/thetaOscillator/wavlist.list")
parser.add_argument("--mfcc", type=str, default="../exp/pkls/mfccs.pkl")
args = parser.parse_args()

wavlist_root = str(Path(args.wavlist).parent) + "/"

with open(args.wavlist) as f:
    lines = f.readlines()


mfccs = []
for line in lines:
    wav_path = ""
    if "/" == line[0]:
        # if absolute path
        wav_path = line.rstrip()
    else:
        # if relative path
        wav_path = wavlist_root + line.rstrip()
    x, sr = librosa.load(wav_path, sr=44100)
    mfccs_per_wav = librosa.feature.mfcc(x, sr=sr, hop_length=441, n_mfcc=20).T
    print("for wav file " + line.rstrip() + ", mfcc shape:")
    print(mfccs_per_wav.shape)
    mfccs.append(mfccs_per_wav)

with open(args.mfcc, "wb") as handle:
    pickle.dump(mfccs, handle)