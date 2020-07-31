#!/bin/bash

if [[  ($# -le 1) || ($# -ge 3) ]]
then
    echo "Usage: bash run.bash DATA_NAME(arbitrary) MAX_K" 
    exit 1
fi 


# unsupervised syllable boundary detection by matlab
mkdir -p ../tools/syllables/thetaOscillator/wavs
mkdir -p ../tools/syllables/thetaOscillator/results
cp ../combined_sounds/combined_sounds.wav ../tools/syllables/thetaOscillator/wavs/

matlab -r "run('../tools/syllables/thetaOscillator/process_wavs.m');exit;"


mkdir -p ../exp
mkdir -p ../exp/pkls

# extract landmarks
mkdir -p ../tools/syllables/landmarks
mkdir -p ../tools/syllables/seglist

cd ../tools/syllables/
python get_landmarks.py
echo "landmarks done"
python get_seglist_from_landmarks.py
echo "segmentation list done"

# extract mfcc and downsampling to make fixed length vectors
cd -
cd ../scripts/
python prep_mfcc_extraction.py
echo "mfcc extraction done"
python mfcc_downsampling.py
echo "downsampling done"

# EMKmeans
cd -
mkdir -p ../tools/eskmeans/results
cd ../tools/eskmeans
python eskmeans_wordseg.py --data_name=$1 --max_k=$2
echo "eskmeans done and results saved"

# segment waveform according to results from ESKmeans
cd -
cd ../scripts/
python wav_segmentation.py --data_name=$1

# STT
cd -
cd ../tools/
python stt/stt_google.py --data_name=$1 --idx=0

# pass the results to DQN and run rl
# rl results are saved at exp/rl_results.csv
cd -
cd ../scripts/
mkdir -p ../exp/res_imgs
python process_recog_result.py --data_name=$1
python main.py

cd -
