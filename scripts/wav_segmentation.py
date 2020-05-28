import pickle as pickle
from pydub import AudioSegment
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--data_name',   type=str)
args = parser.parse_args()

seglist_path = '../tools/syllables/seglist/seglist.pkl'
seg_result_path = '../tools/eskmeans/results/'+ args.data_name + '.pkl'

export_dir_path = '../exp/segmented_wavs_' + args.data_name + '/'
if not os.path.exists(export_dir_path):
    os.mkdir(export_dir_path)

def main():

    wav_original = AudioSegment.from_wav("../combined_sounds/combined_sounds.wav")

    with open(seglist_path, 'rb') as handle:
        seglist = pickle.load(handle)
    with open(seg_result_path, 'rb') as handle:
        result_dict = pickle.load(handle)

    seglist = seglist[0]

    for key in result_dict.keys():
        for seg_index in result_dict[key]:
            # start and end time in milliseconds
            t_start = seglist[seg_index][0] * 10
            t_end   = seglist[seg_index][1] * 10

            wav_segment = wav_original[t_start:t_end]
            wav_segment.export(export_dir_path + str(key) + '_' + str(seg_index) + '.wav', format="wav")
    


main()