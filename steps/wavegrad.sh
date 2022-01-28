#!/bin/sh

rldir=$1
modeldir=$2
conf=$3
mkdir $modeldir -p

python tools/wavegrad/src/wavegrad/__main__.py $rldir $modeldir $conf || exit 1