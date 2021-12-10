import argparse
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
import yaml

from sb3_api import CustomDQNPolicy, CustomDQN
from spolacq import SpoLacq1, asr_segments, test
from wav2vec2_api import ASR


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
    parser.add_argument("datadir", type=str)
    parser.add_argument("workdir", type=str)
    parser.add_argument("load_state_dict_path", type=str)
    parser.add_argument("save_asr_result_path", type=str)
    parser.add_argument("segment_pkl", type=str)
    parser.add_argument("image_nnfeat", type=str)
    parser.add_argument("segment_nnfeat", type=str)
    parser.add_argument("succeeded_log", type=str)
    args = parser.parse_args()
    args = update_args(args.conf, args)
    
    # Food type
    FOODS = (
        "CHERRY",
        "GREEN PEPPER",
        "LEMON",
        "ORANGE",
        "POTATO",
        "STRAWBERRY",
        "SWEET POTATO",
        "TOMATO",
    )
    
    # Focusing mechanism
    image_features = np.load(args.image_nnfeat).squeeze()
    segment_features = np.load(args.segment_nnfeat).squeeze()
    kmeans = KMeans(n_clusters=args.num_clusters, random_state=2).fit(image_features)
    similarity = -cdist(kmeans.cluster_centers_, segment_features)
    focused_segment_ids = similarity.argsort(axis=1)[:, -args.num_per_group:].flatten()
    
    # ASR of segmented wavs
    wav2vec2_path = "./wav2vec2/"
    if os.path.isdir(wav2vec2_path):
        asr = ASR(wav2vec2_path)
    else:
        asr = ASR(args.asr_model_name)
        if args.save: asr.save(wav2vec2_path)
    segment_wave, segment_text, segment_path, segment_spec = asr_segments(
        args.segment_pkl, asr, segment_features.shape[0])
    
    # Make sound dictionary, for spolacq agent, which
    # converts categorical ID to wave utterance.
    sounddic_wave = [segment_wave[i] for i in focused_segment_ids]
    sounddic_text = [segment_text[i] for i in focused_segment_ids]
    sounddic_path = [segment_path[i] for i in focused_segment_ids]
    sounddic_spec = [segment_spec[i] for i in focused_segment_ids]
    
    # Check if sounddic covers every food
    res_dict = {}
    for f in FOODS: res_dict[f] = sounddic_text.count(f)
    print(res_dict, flush=True)
    assert 0 not in res_dict.values(), "The sound dictionary does not cover every food."
    
    # RL environment creation
    if args.use_real_time_asr:
        env = SpoLacq1(FOODS, args.datadir, sounddic_wave, asr)
    else:
        env = SpoLacq1(FOODS, args.datadir, sounddic_text, lambda x: x)
        del asr
    
    # RL learning model creation
    policy_kwargs = dict(
        features_extractor_kwargs=dict(
            cluster_centers=kmeans.cluster_centers_,
            load_state_dict_path=args.load_state_dict_path,
            num_per_group=args.num_per_group,
            purify_rate=args.purify_rate,
        )
    )
    replay_buffer_kwargs = dict(handle_timeout_termination=False)
    
    model = CustomDQN(
        CustomDQNPolicy,
        env,
        policy_kwargs=policy_kwargs,
        replay_buffer_kwargs=replay_buffer_kwargs,
        tensorboard_log="./spolacq_tmplog/",
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
    )
    
    # RL model training
    model.learn(total_timesteps=args.total_timesteps, tb_log_name="spolacq1")
    
    # Save
    if args.save:
        model.save(args.workdir+"/dqn")
        model.save_replay_buffer(args.workdir+"/replay_buffer")
        
        with open(args.workdir+"/sounddic.pkl", "wb") as f:
            pickle.dump(sounddic_wave, f)
        
        with open(args.save_asr_result_path, "w") as f:
            for path, transcription in zip(sounddic_path, sounddic_text):
                if transcription in FOODS:
                    label = FOODS.index(transcription)
                else:
                    label = -1
                f.write(f"{path.split('/')[-1]} {label}\n")
        
        with open(args.succeeded_log, "w") as f:
            for log in env.succeeded_log:
                f.write(f"{sounddic_path[log].split('/')[-1]},{sounddic_text[log].replace(' ', '_').lower()},{log}\n")
    
    # Test the learnt agent
    test(10, env, model)