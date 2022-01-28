#!/bin/sh

rlloaddir=$1
wgsavedir=$2
savedir=$3
savedir2=$4
conf=$5

mkdir -p $savedir2
cp $savedir/*.zip $savedir/actionreward.txt $savedir2
cp $savedir/used_audios.txt $savedir2/used_audios.txt
python utils/main_wavegrad.py $conf $rlloaddir $wgsavedir $savedir2