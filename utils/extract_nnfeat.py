from argparse import ArgumentParser
from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from model import SimpleImageCorrNet
from loader import A_Dataset, V_Dataset

parser = ArgumentParser()
parser.add_argument("modality", type=str, choices=["audio", "image"])
parser.add_argument("modelparam", type=str)
parser.add_argument("pkllist", type=str)
parser.add_argument("outpath", type=str)
args = parser.parse_args()

def load_model(parampath, modality, device):
    model = SimpleImageCorrNet(num_class = 50).to(device)
    checkpoint = torch.load(parampath)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    if modality == "audio":
        return model._sound_extract
    elif modality == "image":
        return model._visual_extract
    else:
        raise ValueError()

def load_data(dataset_path, modality):
    paths = []
    with open(dataset_path, "r") as f:
        for t in f:
            paths.append(t.strip())
    v_tr = transforms.ToTensor()
    if modality == "audio":
        DatasetClass = A_Dataset
    elif modality == "image":
        DatasetClass = V_Dataset
    else:
        raise ValueError()
    seg_loader = DataLoader(DatasetClass(paths, None, v_tr), batch_size=1, shuffle=False, num_workers=7, pin_memory=True)
    return seg_loader

def extract(model, data, device):
    seg_fea = []
    with torch.no_grad():
        for seg in tqdm(data):
            seg = seg.to(device)
            fea = model(seg)
            seg_fea.append(fea.cpu())
    seg_fea_tensor = torch.stack(seg_fea, dim=0)
    return seg_fea_tensor


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model(args.modelparam, args.modality, device)
seg_loader = load_data(args.pkllist, args.modality)
seg_fea_tensor = extract(model, seg_loader, device)

np.save(args.outpath, seg_fea_tensor.numpy())