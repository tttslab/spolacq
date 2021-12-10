datadir=$1
workdir=$2
segdir=$workdir/feat_pkls


# audio_image_pair.txt: [(audio_path, image_path)]
# Audio-image pairs used to train backend (resnet50) that extract feature of audio and images.
python utils/make_plist.py $datadir | \
    sed -E "/\/(cherry|green_pepper|lemon|orange|potato|strawberry|sweet_potato|tomato)\//!d" \
    > $workdir/audio_image_pair.txt


# train_imgs.txt: [image_path]
# Images used in observation phase. This set should be same with image set of audio_image_pair.txt.
# test_imgs.txt: [image_path]
# images used in dialogue phase.
find $datadir/*/train_number*/*.jpg |
    sed -E "/\/(cherry|green_pepper|lemon|orange|potato|strawberry|sweet_potato|tomato)\//!d" > $workdir/train_imgs.txt
find $datadir/*/test_number*/*.jpg |
    sed -E "/\/(cherry|green_pepper|lemon|orange|potato|strawberry|sweet_potato|tomato)\//!d" > $workdir/test_imgs.txt


# seg_audios.txt: [audio_path]
# Segmented audios used to make utterance.
find $segdir/*.pkl | sort > $workdir/seg_audios.txt