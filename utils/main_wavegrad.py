import argparse
import json
import os
import statistics
import sys
from typing import List

import numpy as np
import torch
import torchaudio
import yaml

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from sb3_api import CustomDQN, CustomDQNPolicy
from spolacq import SpoLacq2
from tools.wavegrad.src.wavegrad.inference import predict_using_model, make_model
from wav2vec2_api import ASR


def update_args(yamlpath, opts):
    opts = vars(opts) #Namespace -> dict
    with open(yamlpath, "r") as f:
        conf = yaml.safe_load(f)
    assert set(opts.keys()).isdisjoint(set(conf.keys())) #opts & conf == {}
    opts.update(conf) #opts += conf
    return argparse.Namespace(**opts) #dict -> Namespace


class SoundDict:
    def __init__(self, audio_dir: str, audio_list: str):
        self.sounddic = dict()
        with open(audio_list, "r") as f:
            for r in f:
                action, wavname = r.strip().split(" ")
                self.sounddic[int(action)] = f"{audio_dir}/{wavname}"
    
    def __len__(self):
        return len(self.sounddic)
    
    def predict(self, action: int):
        # wav: type = torch.tensor, shape = [1, wavlen], -1~1
        wav, sr = torchaudio.load(self.sounddic[action])
        assert sr == 8000 and -1 <= wav.min() and wav.max() <= 1 and wav.ndim == 2 and wav.shape[0] == 1
        return wav.squeeze(0), sr

    def set_results(self, acc):
        pass


class WaveGradWrapper:
    def __init__(
        self,
        num_actions: int,
        ma_weight: float,
        average_reward_adjustion: float,
        action_frames: str,
        wavegrad_paths: List[str],
        reward_log_path: str,
    ):
        self.num_actions = num_actions
        self.ma_weight = ma_weight
        
        # Prepare WaveGrad models
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = [make_model(p, device=self.device) for p in wavegrad_paths]
        
        with open(action_frames, "r") as f:
            self.num_frames = [int(r.strip()) for r in f]

        with open(reward_log_path, "r") as f:
            rewards_1st = [float(r.strip()) for r in f]
        
        # average last 50 epoch
        self._reset_stats(statistics.mean(rewards_1st[-50:])+average_reward_adjustion)
        
        # Initialize moving average reward for each WaveGrad model
        self.moving_averages = list()
        self.store_moving_averages(self.idx, self.acc)
    
    def _reset_stats(self, init_acc):
        self.acc = [init_acc] * len(self.models)
        self.idx = 0
    
    def store_moving_averages(self, selected_index, averages):
        log = f"{selected_index}|" + ",".join([str(x) for x in averages]) + "\n"
        self.moving_averages.append(log)
    
    def save_moving_averages(self, path: str):
        with open(path, "w") as f:
            f.writelines(self.moving_averages)
    
    def __len__(self):
        return self.num_actions
    
    def predict(self, action: int):
        num_frame = self.num_frames[action]
        action = torch.IntTensor([action]) # input to torch.nn.Embedding
        wav, sr = predict_using_model(action, self.models[self.idx], device=self.device)
        wav = wav.squeeze()[:num_frame]
        return wav, sr

    def set_results(self, acc):
        self.acc[self.idx] = self.acc[self.idx]*(1. - self.ma_weight) + acc*self.ma_weight
        self.idx = self.acc.index(max(self.acc)) # argmax
        self.store_moving_averages(self.idx, self.acc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("conf", type=str)
    parser.add_argument("rlloaddir", type=str)
    parser.add_argument("wgloaddir", type=str)
    parser.add_argument("savedir", type=str)
    args = parser.parse_args()
    args = update_args(args.conf, args)
    
    # Paths for speech organ
    if args.use_wavegrad:
        wavegrad_paths = [f"{args.wgloaddir}/weights-{i}ep.pt" for i in range(
            args.wgstart, args.wgend, args.wgstep)]
    else: # use sound dictionary
        audio_dir = f"{args.rlloaddir}/trimed_wavs_8k"
    audio_list = f"{args.savedir}/used_audios.txt"
    action_frames = f"{args.savedir}/action_frames.txt"
    
    # Paths for logging
    reward_log_path = f"{args.savedir}/actionreward.txt"
    audio_log_dir = f"{args.savedir}/audio_log.h5"
    qa_log_path = f"{args.savedir}/qa_log.txt"
    ma_log_path = f"{args.savedir}/ma_log.txt"
    dqn_path = f"{args.savedir}/dqn"
    
    # Paths for DQN
    cluster_centers_path = f"{args.rlloaddir}/cluster_centers.npy"
    load_state_dict_path = f"{args.rlloaddir}/unsup_backend.pth.tar"
    
    # Prepare ASR
    wav2vec2_path = "./wav2vec2/"
    if os.path.isdir(wav2vec2_path):
        # Try to load model from local
        with open(f"{wav2vec2_path}config.json") as f:
            conf = json.load(f)
        if conf["_name_or_path"] == args.asr_model_name:
            asr = ASR(wav2vec2_path)
        else:
            asr = ASR(args.asr_model_name)
    else:
        asr = ASR(args.asr_model_name)
    
    # Prepare speech organ
    if args.use_wavegrad:
        speech_organ = WaveGradWrapper(
            args.num_clusters*args.num_per_group,
            args.ma_weight,
            args.average_reward_adjustion,
            action_frames,
            wavegrad_paths,
            reward_log_path,
        )
    else: # use sound dictionary
        speech_organ = SoundDict(audio_dir, audio_list)
    
    with open(audio_list, "r") as f:
        action_mask = [int(r.strip().split(" ")[0]) for r in f]
    
    # RL environment creation
    env = SpoLacq2(
        tuple(f.upper().replace("_", " ") for f in args.obj_name_list),
        "data/dataset",
        asr,
        speech_organ,
        action_mask,
        getattr(args, "add_sin_noise", False),
        getattr(args, "sin_noise_db", None),
        getattr(args, "sin_noise_freq", None),
    )
    
    # RL model creation
    if args.load:
        model = CustomDQN.load(dqn_path, env=env)
        model.set_action_mask(action_mask)
    else:
        policy_kwargs = dict(
            features_extractor_kwargs=dict(
                cluster_centers=np.load(cluster_centers_path),
                load_state_dict_path=load_state_dict_path,
                action_mask=action_mask,
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
            batch_size=args.batch_size,
            train_freq=args.train_freq,
            exploration_initial_eps=0.0,
            exploration_final_eps=0.0,
        )
    
    # RL model training
    model.learn(
        total_timesteps=env.num_image*args.num_epoch,
        tb_log_name="spolacq2",
        reset_num_timesteps=args.reset_num_timesteps,
    )
    
    # Save logs
    if args.save:
        env.save_audio(audio_log_dir)
        env.save_history(qa_log_path)
        env.save_reward(reward_log_path)
        model.save(dqn_path)
        if args.use_wavegrad:
            speech_organ.save_moving_averages(ma_log_path)