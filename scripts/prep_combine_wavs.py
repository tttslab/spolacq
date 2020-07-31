'''
    concatenate wavs into a single one
'''

from os import walk
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--path',   type=str)
args = parser.parse_args()

path = args.path
_, _, filenames = next(walk(path), (None, None, []))

from pydub import AudioSegment

combined_sounds = None
for i, filename in enumerate(filenames):
    # if i == 1000:
    #     break
    if combined_sounds is None:
        combined_sounds = AudioSegment.from_wav(path + "/" + filename)
    else:
        combined_sounds = combined_sounds + AudioSegment.from_wav(path + "/" + filename)

combined_sounds.export("../combined_sounds/combined_sounds.wav", format="wav")

print('done combining!')
