import sys
import pickle as pickle
from pydub import AudioSegment
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--wordseg", type=str)
parser.add_argument("--seglist", type=str)
parser.add_argument("--wavlist", type=str)
parser.add_argument("--wavseg_dir", type=str)
args = parser.parse_args()

val = []

def main():

    wavs = []
    with open(args.wavlist, "r") as handle:
        for p in handle:
            wav = AudioSegment.from_wav(p.strip())
            wavs.append(wav)

    with open(args.seglist, "rb") as handle:
        seglist = pickle.load(handle)
    with open(args.wordseg, "rb") as handle:
        result_dict = pickle.load(handle)

    seglist_a = []
    for i,s in enumerate(seglist):
        for ss in s:
            seglist_a.append((i,ss[0],ss[1]))

    for key in result_dict.keys():
        for seg_index in result_dict[key]:
            # start and end time in milliseconds
            wavi = seglist_a[seg_index][0]
            wav = wavs[wavi]
            t_start = seglist_a[seg_index][1] * 10
            t_end   = seglist_a[seg_index][2] * 10
            
            for i in range(t_start//10, t_end//10):
                val.append((wavi, i))

            wav_segment = wav[t_start:t_end]
            wav_segment.export(args.wavseg_dir + "/" + str(key) + "_" + str(seg_index) + ".wav", format="wav")

    if len(set(val)) != len(val): #there are at least one duplication
        print("Error: There is an overlap between segmented wavs. The segmentation is FAILED.", file=sys.stderr)
        exit(1)

main()