#!/bin/sh

if [ $# -ne 1 ]
then
    echo "Usage: sh runwavegrad.sh DATA_DIR_NAME(arbitrary)"
    exit 1
fi

workdir=$1
datadir=data/dataset
combined_sounds=data/combined_sounds_8foods_interval_spolacq1.wav
segdir=$workdir/seg
rlbasedir=$workdir/unsup_rl/test
rl1stdir=$workdir/nn/trial_100x120_1st
rldir=$rl1stdir
k=2 # max k of ES-Kmeans

sh setup.sh
cd ..


# Segmentation
sh steps/sylseg.sh $datadir $segdir $combined_sounds conf/rl1st.yaml
sh steps/wordseg.sh $segdir $k


# Correspondence learning
sh steps/make_spectrogram.sh $segdir $rlbasedir
sh steps/make_pathlist.sh $datadir $rlbasedir conf/rl1st.yaml
sh steps/train_unsup_nnKmean.sh $rlbasedir conf/pretrain.yaml
sh steps/extract_nnfeat.sh $rlbasedir


# Reinforcement learning
sh steps/rl_1st.sh $rlbasedir $rl1stdir conf/rl1st.yaml

# ----- Change in environmental noise -----

# Sound dictionary based agent
rldirnew=$workdir/nn/trial_addsin_100x120_sounddic
sh steps/rl_resume_sounddic.sh $rlbasedir dummy $rl1stdir $rldirnew conf/rl_resume_sounddic.yaml

# WaveGrad speech organ based agent
for i in $(seq 1 10); do
    wgdirnew=$workdir/wavegrad/trial_addsin_100x120_24k_gen${i}
    rldirnew=$workdir/nn/trial_addsin_100x120_24k_gen${i}
    if [ ${i} = 1 ] ; then
        sh steps/wavegrad.sh $rldir $wgdirnew conf/train.yaml
        sh steps/rl_resume.sh $rlbasedir $wgdirnew $rldir $rldirnew conf/rl_resume_1.yaml
        preconf=conf/rl_resume_1.yaml
    else
        sh steps/wg_resume.sh $rldir $wgdir $wgdirnew $preconf conf/retrain.yaml
        sh steps/rl_resume.sh $rlbasedir $wgdirnew $rldir $rldirnew conf/rl_resume.yaml
        preconf=conf/rl_resume.yaml
    fi
    wgdir=$wgdirnew
    rldir=$rldirnew
done