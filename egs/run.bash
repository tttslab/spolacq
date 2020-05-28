#!/bin/bash

mkdir -p ../tools/syllables/thetaOscillator/wavs
mkdir -p ../tools/syllables/thetaOscillator/results
cp ../combined_sounds/combined_sounds.wav ../tools/syllables/thetaOscillator/wavs/

matlab -r "run('../tools/syllables/thetaOscillator/process_wavs.m');exit;"


mkdir -p ../exp
mkdir -p ../exp/pkls

mkdir -p ../tools/syllables/landmarks
mkdir -p ../tools/syllables/seglist

cd ../tools/syllables/
python get_landmarks.py
echo "landmarks done"
python get_seglist_from_landmarks.py
echo "segmentation list done"

cd -
cd ../scripts/
python prep_mfcc_extraction.py
echo "mfcc extraction done"
python mfcc_downsampling.py
echo "downsampling done"

cd -
mkdir -p ../tools/eskmeans/results
cd ../tools/eskmeans
python eskmeans_wordseg.py --data_name=$1 --max_k=$2
echo "eskmeans done and results saved"

cd -
cd ../scripts/
python wav_segmentation.py --data_name=$1

cd -
cd ../tools/
python stt/stt_google.py --data_name=$1 --idx=0

cd -
cd ../scripts/
mkdir -p ../exp/res_imgs
python process_recog_result.py --data_name=$1
python main.py

cd -
