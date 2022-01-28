import argparse

import torchaudio

parser = argparse.ArgumentParser()
parser.add_argument("qa_log_path", type=str)
parser.add_argument("used_audio_path", type=str)
parser.add_argument("wav8kdir", type=str)
args = parser.parse_args()

speech_dict = {}
with open(args.used_audio_path, "r") as f:
    for r in f:
        action, wavname = r.strip().split(" ")
        speech_dict[int(action)] = wavname

counter = {}
with open(args.qa_log_path, "r") as f:
    for r in f:
        time, action, reward, recognized = r.strip().split("|")
        if int(float(reward)) != 1: continue
        wavpath = speech_dict[int(action)]
        action_tuple = (action, wavpath)
        if action_tuple not in counter: counter[action_tuple] = 0
        counter[action_tuple] += 1

counter = list(counter.items())
counter = sorted(counter, key=lambda x: x[1], reverse=True)
ok = [x[0] for x in counter]

for action, wavpath in ok:
    wav, sr = torchaudio.load(f"{args.wav8kdir}/{wavpath}")
    assert sr == 8000
    if wav.size()[1] < 12000:
        print(f"{action} {wavpath}")
