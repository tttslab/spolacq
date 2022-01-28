import argparse

parser = argparse.ArgumentParser()
parser.add_argument("filtered_used_audios", type=str)
parser.add_argument("qa_log_path", type=str)
args = parser.parse_args()

training_actions = []
with open(args.filtered_used_audios, "r") as f:
    for r in f:
        action, _ = r.strip().split(" ")
        training_actions.append(int(action))

transcripts = {}
with open(args.qa_log_path, "r") as f:
    for r in f:
        time, action, reward, recognized = r.strip().split("|")
        if int(float(reward)) != 1: continue
        if int(action) in transcripts:
            assert transcripts[int(action)] == recognized
            continue
        transcripts[int(action)] = recognized

for action in training_actions:
    print(transcripts[action])
