import argparse
import h5py
import torchaudio

parser = argparse.ArgumentParser()
parser.add_argument("wg_train_list", type=str)
parser.add_argument("wav8kdir", type=str)
parser.add_argument("audio_log_h5", type=str)
args = parser.parse_args()

trainings = []
with open(args.wg_train_list, "r") as f:
    for r in f:
        action, wavname = r.strip().split(",")
        trainings.append((action, wavname))

with h5py.File(args.audio_log_h5, "a") as f:
    for _, wavname in trainings:
        if wavname in f: continue
        wav, sr = torchaudio.load(f"{args.wav8kdir}/{wavname}")
        assert sr == 8000
        f[wavname] = wav
