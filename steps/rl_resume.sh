#!/bin/sh

rlloaddir=$1
wgsavedir=$2
savedir=$3
savedir2=$4
conf=$5

mkdir -p $savedir2
cp $savedir/*.zip $savedir/actionreward.txt $savedir2
cp $savedir/transcript_for_validation.txt $savedir/action_frames.txt $savedir2/
cp $savedir/used_audios_filtered.txt $savedir2/used_audios.txt 2>/dev/null || \
    cp $savedir/used_audios.txt $savedir2/used_audios.txt #if there is no the filtered file.
python utils/main_wavegrad.py $conf $rlloaddir $wgsavedir $savedir2 || exit 1
python utils/wavegrad/make_wavegrad_training_list.py $savedir2/qa_log.txt > $savedir2/wavegrad_training_list.txt