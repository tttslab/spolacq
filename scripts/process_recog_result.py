import os
import argparse, pickle

parser = argparse.ArgumentParser()
parser.add_argument('--data_name',   type=str)
args = parser.parse_args()

recog_result_path = "../exp/segmented_wavs_" + args.data_name + "/stt_results.txt"
exp_path = "../exp/pkls/recog_results_dict.pkl"

res_dict = {}

f = open(recog_result_path, "r")
lines = f.readlines()

res_dict['num_words'] = len(lines)
res_dict['up'] = 0
res_dict['down'] = 0
res_dict['left'] = 0
res_dict['right'] = 0
res_dict['forward'] = 0
res_dict['backward'] = 0

for line in lines:
    line_words = line.split(" ")
    if "up" in line_words[1]:
        res_dict['up'] += 1
    elif "down" in line_words[1]:
        res_dict['down'] += 1
    elif "left" in line_words[1]:
        res_dict['left'] += 1
    elif "right" in line_words[1]:
        res_dict['right'] += 1
    elif "forward" in line_words[1]:
        res_dict['forward'] += 1
    elif "backward" in line_words[1]:
        res_dict['backward'] += 1

print(res_dict)

with open(exp_path, 'wb') as handle:
    pickle.dump(res_dict, handle)