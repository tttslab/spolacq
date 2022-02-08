import argparse
import pickle

import yaml

from sb3_api import CustomDQNPolicy, CustomDQN, CustomFeaturesExtractor
from spolacq import SpoLacq1, RLPreprocessor, test


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
    parser.add_argument("segment_pkl", type=str)
    parser.add_argument("segment_list", type=str)
    parser.add_argument("image_nnfeat", type=str)
    parser.add_argument("segment_nnfeat", type=str)
    args = parser.parse_args()
    args = update_args(args.conf, args)
    
    # Preprocess for RL
    preprocessor = RLPreprocessor(
        args.obj_name_list,
        args.segment_nnfeat,
        args.image_nnfeat,
    )
    asr = preprocessor.recognize(
        args.segment_list,
        args.asr_model_name,
        args.use_real_time_asr,
    )
    sounddic_wave, sounddic_text, sounddic_path, cluster_centers = preprocessor.focus(
        args.num_clusters, args.num_per_group)
    
    # Spoken questions (simplest case). You may add noise to wav files.
    question_paths = [
        ("data/which_do_you_want.wav", 0),
        ("data/which_do_not_you_want.wav", 1),
    ]

    # RL environment creation
    env = SpoLacq1(
        preprocessor.FOODS,
        args.datadir,
        sounddic_wave if args.use_real_time_asr else sounddic_text,
        asr,
        question_paths if args.use_spoken_question else None
    )
    
    # RL learning model creation
    policy_kwargs = dict(
        features_extractor_class=CustomFeaturesExtractor,
        features_extractor_kwargs=dict(
            cluster_centers=cluster_centers,
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
        seed=0,
    )
    
    # RL model training
    model.learn(total_timesteps=args.total_timesteps, tb_log_name="spolacq1")
    
    # Save
    if args.save:
        model.save(args.workdir+"/dqn")
        model.save_replay_buffer(args.workdir+"/replay_buffer")
        
        with open(args.workdir+"/sounddic.pkl", "wb") as f:
            pickle.dump(sounddic_wave, f)
    
    # Test the learnt agent
    test(10, env, model)