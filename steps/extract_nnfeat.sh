workdir=$1

train_list=$workdir/train_imgs.txt
test_list=$workdir/test_imgs.txt
segmented_audio_list=$workdir/seg_audios.txt

mkdir -p $workdir/data

python utils/extract_nnfeat.py audio $workdir/unsup_backend.pth.tar $segmented_audio_list $workdir/data/new720db20k2_fea_sim_8type_clean_limited_data.npy
python utils/extract_nnfeat.py image $workdir/unsup_backend.pth.tar $train_list $workdir/data/train_img_fea_sim_8type_clean_limited_data.npy
python utils/extract_nnfeat.py image $workdir/unsup_backend.pth.tar $test_list $workdir/data/test_img_fea_sim_8type_clean_limited_data.npy