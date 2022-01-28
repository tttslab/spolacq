import argparse

parser = argparse.ArgumentParser()
parser.add_argument("used_audio_path", type=str)
args = parser.parse_args()

speech_dict = {}
with open(args.used_audio_path, "r") as f:
    for i, r in enumerate(f):
        action, wavname = r.strip().split(" ")
        print(f"{i},{wavname}")
