#!/bin/bash

work=$1

wavlist=$work/wavlist.txt
sylseg=$work/sylseg.csv #syllable segmentation result
wavseg_dir=$work/segmented_wavs

landmarks=$work/landmarks_syllable_seg.pkl
seglist=$work/seglist.pkl
wordseg=$work/eskseg.pkl
eskseg_result=$work/eskseg_result.txt
mfcc=$work/mfccs.pkl
embed=$work/embed_dur_dic.pkl

max_k=$2 #max k of ESKmeans
n=10 #number of mfccs to keep to make embedding feature of any length speech

python tools/syllables/get_landmarks.py --sylseg=$sylseg --landmarks=$landmarks || exit 1
echo "landmarks done"

python tools/syllables/get_seglist_from_landmarks.py --landmarks=$landmarks --seglist=$seglist || exit 1
echo "segmentation list done"

python scripts/prep_mfcc_extraction.py --wavlist=$wavlist --mfcc=$mfcc || exit 1
echo "mfcc extraction done"

python scripts/mfcc_downsampling.py --mfcc=$mfcc --seglist=$seglist --embed=$embed --n=$n || exit 1
echo "downsampling done"

python tools/eskmeans/eskmeans_wordseg.py --landmarks=$landmarks --embed=$embed --txt_path=$eskseg_result --pkl_path=$wordseg --max_k=$max_k || exit 1
echo "eskmeans done"

mkdir -p $wavseg_dir
python utils/wav_segmentation_multi.py --wordseg=$wordseg --seglist=$seglist --wavlist=$wavlist --wavseg_dir=$wavseg_dir || exit 1
echo "wav file segmentation done and results saved"