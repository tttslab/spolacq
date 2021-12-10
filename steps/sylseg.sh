data=$1
work=$2
combined_sounds=$3
include_no=$4

# Make combined_sounds
python utils/prepare_data.py $data
python utils/prep_combine_wavs_clean_and_interval.py $data $combined_sounds $include_no

# Syllable segmentation
mkdir -p $work
echo $PWD/$combined_sounds > $work/wavlist.txt
abswork=$PWD/$work

cd tools/syllables/thetaOscillator
matlab -batch "process_wavs(\"${abswork}/wavlist.txt\", \"${abswork}/sylseg.mat\", \"${abswork}/sylseg.csv\")"
cd -