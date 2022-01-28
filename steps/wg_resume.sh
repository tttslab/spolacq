#!/bin/sh

rlloaddir=$1
wgloaddir=$2
wgsavedir=$3
preconf=$4
conf=$5
mkdir $wgsavedir -p

cp -a $wgloaddir/*.pt $wgsavedir
python utils/wavegrad/relink_to_bestweights.py $wgsavedir $rlloaddir $preconf
python utils/wavegrad/reset_checkpoint_epoch.py $wgsavedir/weights.pt

sh steps/wavegrad.sh $rlloaddir $wgsavedir $conf