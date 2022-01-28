#!/bin/sh
rlloaddir=$1
savedir=$2

python utils/wavegrad/filtering_audios.py $savedir/qa_log.txt $savedir/used_audios.txt $rlloaddir/trimed_wavs_8k > $savedir/used_audios_filtered.txt
python utils/wavegrad/make_wavegrad_training_list_for1strl.py $savedir/used_audios_filtered.txt > $savedir/wavegrad_training_list.txt
python utils/wavegrad/add_training_audios_to_log.py $savedir/wavegrad_training_list.txt $rlloaddir/trimed_wavs_8k $savedir/audio_log.h5
python utils/wavegrad/count_frames.py $savedir/wavegrad_training_list.txt $savedir/audio_log.h5 > $savedir/action_frames.txt

#make transcript file for validation
python utils/wavegrad/make_transcript.py $savedir/used_audios_filtered.txt $savedir/qa_log.txt > $savedir/transcript_for_validation.txt
