import argparse

parser = argparse.ArgumentParser()
parser.add_argument("qa_log_path", type=str)
args = parser.parse_args()

res = []
with open(args.qa_log_path, "r") as f:
    for r in f:
        time, action, reward, recognized = r.strip().split("|")
        if int(float(reward)) != 1: continue
        epoch, iteration = time.split(",")
        epoch, iteration = int(epoch), int(iteration)
        wavpath = f"epoch{epoch:05}_iter{iteration:05}.wav"
        res.append((action, wavpath))

for action, wavpath in res:
    print(f"{action},{wavpath}")
