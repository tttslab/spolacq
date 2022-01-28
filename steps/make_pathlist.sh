datadir=$1
workdir=$2
segdir=$workdir/feat_pkls
conf=$3

python utils/make_plist.py $datadir \
                           $workdir/audio_image_pair.txt \
                           $workdir/train_imgs.txt \
                           $workdir/test_imgs.txt \
                           $conf

# seg_audios.txt: [audio_path]
# Segmented audios used to make utterance.
find $segdir/*.pkl | sort > $workdir/seg_audios.txt