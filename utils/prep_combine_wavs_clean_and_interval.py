"""
    concatenate wavs into a single one
"""

import argparse
import random
from glob import glob
import os

from pydub import AudioSegment
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("path", type=str)
parser.add_argument("combined_sounds", type=str)
parser.add_argument("include_no", type=str, choices=["True", "False"])
args = parser.parse_args()

filenames = []
obj_name_list = ["cherry", "green_pepper", "lemon", "orange", "potato", "strawberry", "sweet_potato", "tomato"]

for obj_name in obj_name_list:
    objwav = glob(f"{args.path}/{obj_name}/description/clean/*.wav")
    repeat_objwav = objwav*22 + objwav[0:2]
    filenames += repeat_objwav
    assert len(objwav) == 4, "mismatch in len(objwav)"

parent_path = os.path.normpath(args.path+"/..")
if args.include_no == "True":
    filenames += [f"{parent_path}/no.wav"] * 90

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