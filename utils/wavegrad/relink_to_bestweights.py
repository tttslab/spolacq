import argparse
import os
from pathlib import Path

import torch
import yaml

def relink(sourcename, to):
    if os.name == "nt":
        raise NotImplementedError()
    else:
        if os.path.islink(to):
            os.unlink(to)
        os.symlink(sourcename, to)


def update_args(yamlpath, opts):
    opts = vars(opts) #Namespace -> dict
    with open(yamlpath, "r") as f:
        conf = yaml.safe_load(f)
    assert set(opts.keys()).isdisjoint(set(conf.keys())) #opts & conf == {}
    opts.update(conf) #opts += conf
    return argparse.Namespace(**opts) #dict -> Namespace


parser = argparse.ArgumentParser()
parser.add_argument("wavegrad_dir", type=str)
parser.add_argument("rl_dir", type=str)
parser.add_argument("rlconf", type=str)
args = parser.parse_args()
args = update_args(args.rlconf, args)
args.wavegrad_paths = [f"{args.wavegrad_dir}/weights-{i}ep.pt" for i in range(
                       args.wgstart, args.wgend, args.wgstep)]
args.link_path      = f"{args.wavegrad_dir}/weights.pt"
args.ma_log         = f"{args.rl_dir}/ma_log.txt"

with open(args.ma_log, "r") as f:
    best_idx = int(f.readlines()[-1].strip().split("|")[0])
best_param = args.wavegrad_paths[best_idx]
relink(Path(best_param).name, args.link_path)