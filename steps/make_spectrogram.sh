wavdir=$1/segmented_wavs
work=$2
spectrdir=$work/feat_pkls
trimedwav=$work/trimed_wavs # clean
targetwav=$work/noisy_trimed_wavs # S/N=30dB white noise

mkdir $work $trimedwav $spectrdir $targetwav -p

# Remove silence
python utils/rmsil.py $wavdir $trimedwav

# Extract spectrogram
find $trimedwav/*.wav > $work/segmented_wavs.txt
cat $work/segmented_wavs.txt | sed "s#${trimedwav}/\(.\+\)\.wav#${spectrdir}/\1.pkl#g" > $work/mfccs.txt
python utils/extract_spectrogram.py $work/segmented_wavs.txt $work/mfccs.txt

# Add S/N=30dB white noise
python utils/add_sil_and_noise.py $trimedwav $targetwav