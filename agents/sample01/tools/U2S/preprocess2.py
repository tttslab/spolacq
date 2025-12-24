import os
import sys

import librosa
import torch
from tqdm import tqdm
import yaml
from pathlib import Path

# from ..S2U.dataloaders.utils import compute_spectrogram
# from ..S2U.run_utils import load_audio_model_and_state
# from ..S2U.steps.unit_analysis import get_feats_codes, DenseAlignment

from ..I2U.get_s5hubert_unit import get_unit

from ....base_agent_with_units import BaseAgentWithUnits


# def get_unit(path):
#     code_q3_ali = get_code_ali(audio_model, "quant3", path, device).get_sparse_ali()
#     encoded = []
#     enc_old = -1
#     for _, _, code in code_q3_ali.data:
#         assert enc_old != code
#         encoded.append(str(code))
#         enc_old = code
#     return encoded


# def get_code_ali(audio_model, layer, path, device):
#     y, sr = librosa.load(path, sr=None)
#     mels, nframes = compute_spectrogram(y, sr)
#     mels = mels[:, :nframes]
#     _, _, codes, spf = get_feats_codes(audio_model, layer, mels, device)
#     code_list = codes.detach().cpu().tolist()
#     return DenseAlignment(code_list, spf)


def generate_u2s_dataset(
    u2s_pretrain_config,
    audio_paths: list[list[Path]],
    agent: BaseAgentWithUnits,
    output_data_dir: Path,
    overwrite: bool = False,
):
    train_audio_paths = audio_paths["train"]
    val_audio_paths = audio_paths["val"]
    test_audio_paths = audio_paths["test"]

    # Setting the File Path
    filelists_train_path = output_data_dir / u2s_pretrain_config["filelist_name_train"]
    filelists_val_path = output_data_dir / u2s_pretrain_config["filelist_name_val"]
    filelists_test_path = output_data_dir / u2s_pretrain_config["filelist_name_test"]

    if filelists_train_path.exists() and filelists_val_path.exists() and filelists_test_path.exists() and not overwrite:
        print("All files already exist. Skipping...")
        train_data_num = len([line for line in open(filelists_train_path, "r")])
        val_data_num = len([line for line in open(filelists_val_path, "r")])
        test_data_num = len([line for line in open(filelists_test_path, "r")])
        data_info = {
            "train_path": filelists_train_path,
            "val_path": filelists_val_path,
            "test_path": filelists_test_path,
            "train_count": train_data_num,
            "val_count": val_data_num,
            "test_count": test_data_num,
        }
    else:
        # Data Generation
        train_data = [
            f"{path}|{' '.join(agent.s2u.get_unit(path))}\n" for paths in tqdm(train_audio_paths) for path in paths
        ]
        val_data = [
            f"{path}|{' '.join(agent.s2u.get_unit(path))}\n" for paths in tqdm(val_audio_paths) for path in paths
        ]
        test_data = [
            f"{path}|{' '.join(agent.s2u.get_unit(path))}\n" for paths in tqdm(test_audio_paths) for path in paths
        ]

        # Creating the Output Directory
        output_data_dir.mkdir(parents=True, exist_ok=True)

        # Writing to a file
        with open(str(filelists_train_path), "w") as f:
            f.writelines(train_data)

        with open(str(filelists_val_path), "w") as f:
            f.writelines(val_data)

        with open(str(filelists_test_path), "w") as f:
            f.writelines(test_data)

        # Setting the return value
        data_info = {
            "train_path": filelists_train_path,
            "val_path": filelists_val_path,
            "test_path": filelists_test_path,
            "train_count": len(train_data),
            "val_count": len(val_data),
            "test_count": len(test_data),
        }

    return data_info
