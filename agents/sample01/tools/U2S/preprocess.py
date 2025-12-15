import os
import sys

import librosa
import torch
from tqdm import tqdm
import yaml

sys.path.append("../S2U/")
from dataloaders.utils import compute_spectrogram
from run_utils import load_audio_model_and_state
from steps.unit_analysis import get_feats_codes, DenseAlignment

sys.path.append("../I2U/")
from get_s5hubert_unit import get_unit


# def get_unit(path):
#     code_q3_ali = get_code_ali(audio_model, "quant3", path, device).get_sparse_ali()
#     encoded = []
#     enc_old = -1
#     for _, _, code in code_q3_ali.data:
#         assert enc_old != code
#         encoded.append(str(code))
#         enc_old = code
#     return encoded


def get_code_ali(audio_model, layer, path, device):
    y, sr = librosa.load(path, sr=None)
    mels, nframes = compute_spectrogram(y, sr)
    mels = mels[:, :nframes]
    _, _, codes, spf = get_feats_codes(audio_model, layer, mels, device)
    code_list = codes.detach().cpu().tolist()
    return DenseAlignment(code_list, spf)


if __name__ == "__main__":
    with open("../../conf/spolacq3.yaml") as y:
        config = yaml.safe_load(y)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load S2U
    exp_dir = "../../models/S2U"
    audio_model = load_audio_model_and_state(exp_dir=exp_dir)
    audio_model = audio_model.to(device)
    audio_model = audio_model.eval()

    foods = [
        "japanese_food_corner",
        "western_food_corner",
        "Udon",
        "rice_bowl",
        "Tofu",
        "sushi",
        "hamburger",
        "sandwich",
        "pizza",
        "salad",
    ]

    train_lists = []
    val_lists = []
    test_lists = []

    for food in foods:
        for i in range(4):
            for j in range(90):
                train_lists.append(f"../../data/I2U/audio/{food}/{food}_{i}_{j}.wav")

            for j in range(90, 110):
                val_lists.append(f"../../data/I2U/audio/{food}/{food}_{i}_{j}.wav")

            for j in range(110, 120):
                test_lists.append(f"../../data/I2U/audio/{food}/{food}_{i}_{j}.wav")

    train_data = [f"{path}|{" ".join(get_unit(path))}\n" for path in tqdm(train_lists)]
    val_data = [f"{path}|{" ".join(get_unit(path))}\n" for path in tqdm(val_lists)]
    test_data = [f"{path}|{" ".join(get_unit(path))}\n" for path in tqdm(test_lists)]

    os.makedirs("../../data/U2S", exist_ok=True)

    filelists_train_path = os.path.join(os.path.dirname(__file__), "../..", config["U2S"]["filelists_train"])
    filelists_val_path = os.path.join(os.path.dirname(__file__), "../..", config["U2S"]["filelists_val"])
    filelists_test_path = os.path.join(os.path.dirname(__file__), "../..", config["U2S"]["filelists_test"])

    with open(filelists_train_path, "w") as f:
        f.writelines(train_data)

    with open(filelists_val_path, "w") as f:
        f.writelines(val_data)

    with open(filelists_test_path, "w") as f:
        f.writelines(test_data)
