import argparse
import os
import torch

parser = argparse.ArgumentParser()
parser.add_argument("path", type=str)
args = parser.parse_args()


path = os.path.realpath(args.path)
cp = torch.load(path)
cp["step"] = 0
torch.save(cp, path)
