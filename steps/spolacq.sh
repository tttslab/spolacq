datadir=$1
work=$2
conf=$3

mkdir $work/param

python utils/main.py $conf \
                     $datadir \
                     $work \
                     $work/unsup_backend.pth.tar \
                     $work/noisy_trimed_wavs.txt \
                     $work/data/train_img_fea_sim_8type_clean_limited_data.npy \
                     $work/data/new720db20k2_fea_sim_8type_clean_limited_data.npy