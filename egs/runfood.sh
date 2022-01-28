#!/bin/sh

if [ $# -ne 1 ]
then
    echo "Usage: sh runfood.sh DATA_DIR_NAME(arbitrary)" 
    exit 1
fi 

workdir=$1
datadir=data/dataset
combined_sounds=data/combined_sounds.wav
segdir=$workdir/seg
rl1stdir=$workdir/unsup_rl/test
k=2 # max k of ES-Kmeans

sh setup.sh
cd ..


# Segmentation
sh steps/sylseg.sh $datadir $segdir $combined_sounds conf/rl.yaml
sh steps/wordseg.sh $segdir $k


# Correspondence learning
sh steps/make_spectrogram.sh $segdir $rl1stdir
sh steps/make_pathlist.sh $datadir $rl1stdir conf/rl.yaml
sh steps/train_unsup_nnKmean.sh $rl1stdir conf/pretrain.yaml
sh steps/extract_nnfeat.sh $rl1stdir


# Reinforcement learning
sh steps/spolacq.sh $datadir $rl1stdir conf/rl.yaml