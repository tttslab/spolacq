#!/bin/sh

rlloaddir=$1
savedir=$2
conf=$3

##1strl
mkdir $rlloaddir/trimed_wavs_8k $savedir -p
python utils/rl1st_preparing.py $conf $rlloaddir $savedir || exit 1
python utils/main_wavegrad.py $conf $rlloaddir dummy $savedir || exit 1
sh steps/wg1st_preparing.sh $rlloaddir $savedir
