workdir=$1
conf=$2

dataset_path=$workdir/audio_image_pair.txt

python utils/train_unsup_nnKmean.py $conf $dataset_path $workdir