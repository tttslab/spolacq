import argparse
from pathlib import Path

import numpy as np
from pydub import AudioSegment
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
import yaml

from spolacq import asr_segments
from wav2vec2_api import ASR

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

parser = argparse.ArgumentParser()
parser.add_argument("conf", type=str)
parser.add_argument("rlloaddir", type=str)
parser.add_argument("savedir", type=str)
args = parser.parse_args()
args = update_args(args.conf, args)

wavdir = f"{args.rlloaddir}/trimed_wavs"
wav8kdir = f"{args.rlloaddir}/trimed_wavs_8k"
vocabdata = f"{args.rlloaddir}/data/vocab_list_k12000_8type_clean_limited_data_new720db20k2clu_asr.txt"
image_nnfeat = f"{args.rlloaddir}/data/train_img_fea_sim_8type_clean_limited_data.npy"
segment_nnfeat = f"{args.rlloaddir}/data/new720db20k2_fea_sim_8type_clean_limited_data.npy"
segment_pkl = f"{args.rlloaddir}/seg_audios.txt"
cluster_centers = f"{args.rlloaddir}/cluster_centers.npy"
speech_dict_path = f"{args.savedir}/used_audios.txt"


# Food type
FOODS = tuple(f.upper().replace("_", " ") for f in args.obj_name_list)

# Focusing mechanism
image_features = np.load(image_nnfeat).squeeze()
segment_features = np.load(segment_nnfeat).squeeze()
kmeans = KMeans(n_clusters=args.num_clusters, random_state=2).fit(image_features)
similarity = -cdist(kmeans.cluster_centers_, segment_features)
focused_segment_ids = similarity.argsort(axis=1)[:, -args.num_per_group:].flatten()
np.save(cluster_centers, kmeans.cluster_centers_)

# ASR of segmented wavs
asr = ASR(args.asr_model_name)
_, segment_text, segment_path, _ = asr_segments(segment_pkl, asr, segment_features.shape[0])

# Make sound dictionary, for spolacq agent, which
# converts categorical ID to wave utterance.
sounddic_text = [segment_text[i] for i in focused_segment_ids]
sounddic_path = [segment_path[i] for i in focused_segment_ids]


with open(vocabdata, "w") as f:
    for path, transcription in zip(sounddic_path, sounddic_text):
        if transcription in FOODS:
            label = FOODS.index(transcription)
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
