import argparse
from pathlib import Path

import numpy as np
from pydub import AudioSegment
import yaml

from spolacq import RLPreprocessor


def resample(in_path, out_path, sr_to):
    s = AudioSegment.from_file(in_path)
    s = s.set_frame_rate(sr_to)
    s.export(out_path, format="wav")


def update_args(yamlpath, opts):
    opts = vars(opts) #Namespace -> dict
    with open(yamlpath, "r") as f:
        conf = yaml.safe_load(f)
    assert set(opts.keys()).isdisjoint(set(conf.keys())) #opts & conf == {}
    opts.update(conf) #opts += conf
    return argparse.Namespace(**opts) #dict -> Namespace


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("conf", type=str)
    parser.add_argument("rlloaddir", type=str)
    parser.add_argument("savedir", type=str)
    args = parser.parse_args()
    args = update_args(args.conf, args)

    # Data paths
    wavdir = f"{args.rlloaddir}/trimed_wavs"
    wav8kdir = f"{args.rlloaddir}/trimed_wavs_8k"
    vocabdata = f"{args.rlloaddir}/data/vocab_list_k12000_8type_clean_limited_data_new720db20k2clu_asr.txt"
    image_nnfeat = f"{args.rlloaddir}/data/train_img_fea_sim_8type_clean_limited_data.npy"
    segment_nnfeat = f"{args.rlloaddir}/data/new720db20k2_fea_sim_8type_clean_limited_data.npy"
    segment_list = f"{args.rlloaddir}/noisy_trimed_wavs.txt"
    cluster_centers_path = f"{args.rlloaddir}/cluster_centers.npy"
    speech_dict_path = f"{args.savedir}/used_audios.txt"

    # Preprocess for RL
    preprocessor = RLPreprocessor(
        args.obj_name_list,
        segment_nnfeat,
        image_nnfeat,
    )
    preprocessor.recognize(
        segment_list,
        args.asr_model_name,
    )
    sounddic_wave, sounddic_text, sounddic_path, cluster_centers = preprocessor.focus(
        args.num_clusters, args.num_per_group)
    np.save(cluster_centers_path, cluster_centers)

    # wavname, foodID
    with open(vocabdata, "w") as f:
        for path, transcription in zip(sounddic_path, sounddic_text):
            if transcription in preprocessor.FOODS:
                label = preprocessor.FOODS.index(transcription)
            else:
                label = -1
            f.write(f"{path.split('/')[-1]} {label}\n")

    # resample to 8k
    for wavpath in Path(wavdir).glob("*.wav"):
        wavname = str(wavpath.name)
        resample(f"{wavdir}/{wavname}", f"{wav8kdir}/{wavname}", 8000)

    # wavname, transcript_tag -> action_id, wavname
    speech_dict = []
    with open(vocabdata, "r") as f:
        for i, r in enumerate(f):
            wavname, _ = r.strip().split(" ")
            speech_dict.append((i, wavname))

    with open(speech_dict_path, "w") as f:
        for action, wavname in speech_dict:
            f.write(f"{action} {wavname}\n")