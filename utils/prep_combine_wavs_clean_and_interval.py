"""
    concatenate wavs into a single one
"""

import argparse
import random
from glob import glob
import os

from pydub import AudioSegment
from tqdm import tqdm
import yaml


def update_args(yamlpath, opts):
    opts = vars(opts) #Namespace -> dict
    with open(yamlpath, "r") as f:
        conf = yaml.safe_load(f)
    assert set(opts.keys()).isdisjoint(set(conf.keys())) #opts & conf == {}
    opts.update(conf) #opts += conf
    return argparse.Namespace(**opts) #dict -> Namespace


parser = argparse.ArgumentParser()
parser.add_argument("path", type=str)
parser.add_argument("combined_sounds", type=str)
parser.add_argument("conf", type=str)
args = parser.parse_args()
args = update_args(args.conf, args)

filenames = []

for obj_name in args.obj_name_list:
    objwav = glob(f"{args.path}/{obj_name}/description/clean/*.wav")
    q, r = divmod(720//len(args.obj_name_list), len(objwav))
    repeat_objwav = objwav*q + objwav[0:r]
    filenames += repeat_objwav
    assert len(objwav) == 4, "mismatch in len(objwav)"

parent_path = os.path.normpath(args.path+"/..")
if args.include_no:
    filenames += [f"{parent_path}/no.wav"] * 20

random.seed(202110291946)
random.shuffle(filenames)

interval_audio = glob(f"{args.path}/../interval_audio/*.wav")
assert len(interval_audio) == 3, "fail to load interval_audio"

combined_sounds = None
for filename in tqdm(filenames, desc="combining audio"):
    if combined_sounds is None:
        combined_sounds = AudioSegment.from_wav(filename)
    else:
        combined_sounds = combined_sounds + AudioSegment.from_wav(filename)
    
    interval = random.choice(interval_audio)
    combined_sounds += AudioSegment.from_wav(interval)

combined_sounds.export(args.combined_sounds, format="wav")
print("done combining!")