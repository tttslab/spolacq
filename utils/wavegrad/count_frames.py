import argparse

import h5py

parser = argparse.ArgumentParser()
parser.add_argument("wg_train_list", type=str)
parser.add_argument("audio_log_h5", type=str)
args = parser.parse_args()

trainings = []
with open(args.wg_train_list, "r") as f:
    for r in f:
        action, wavname = r.strip().split(",")
        trainings.append((action, wavname))

with h5py.File(args.audio_log_h5, "r") as f:
    for _, wavname in trainings:
        print(f[wavname][()].shape[1])
