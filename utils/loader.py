import functools
import pickle

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch


class AV_Dataset(Dataset):
    def __init__(self, pair_paths, a_tr, v_tr):
        self.pair_paths = pair_paths  # (audio, visual)
        self.a_tr = a_tr
        self.v_tr = v_tr

    def __len__(self):
        return len(self.pair_paths)

    def _a_load(self, path):
        with open(path, "rb") as f:
            feat = pickle.load(f)
        log_pad = 0.010
        feat = np.log(np.abs(feat) + log_pad).T
        if self.a_tr:
            feat = self.a_tr(feat)
        return feat

    def _v_load(self, path):
        img = Image.open(path)
        img = img.resize((224, 224))
        if self.v_tr:
            img = self.v_tr(img)
        return img

    @functools.lru_cache(maxsize=None)  # Cache loaded structures
    def __getitem__(self, index):
        a = self._a_load(self.pair_paths[index][0])
        v = self._v_load(self.pair_paths[index][1])
        return a, v

class A_Dataset(Dataset):
    def __init__(self, paths, a_tr, v_tr):
        self.paths = paths  # (audio, visual)
        self.a_tr = a_tr
        self.v_tr = v_tr

    def __len__(self):
        return len(self.paths)

    def _a_load(self, path):
        with open(path, "rb") as f:
            feat = pickle.load(f)
        log_pad = 0.010
        feat = np.log(np.abs(feat) + log_pad).T
        if self.a_tr:
            feat = self.a_tr(feat)
        return feat

    def _v_load(self, path):
        img = Image.open(path)
        img = img.resize((224, 224))
        if self.v_tr:
            img = self.v_tr(img)
        return img

    @functools.lru_cache(maxsize=None)  # Cache loaded structures
    def __getitem__(self, index):
        a = self._a_load(self.paths[index])
        return a

class V_Dataset(Dataset):
    def __init__(self, paths, a_tr, v_tr):
        self.paths = paths  # (audio, visual)
        self.a_tr = a_tr
        self.v_tr = v_tr

    def __len__(self):
        return len(self.paths)

    def _a_load(self, path):
        with open(path, "rb") as f:
            feat = pickle.load(f)
        log_pad = 0.010
        feat = np.log(np.abs(feat) + log_pad).T
        if self.a_tr:
            feat = self.a_tr(feat)
        return feat

    def _v_load(self, path):
        img = Image.open(path)
        img = img.resize((224, 224))
        if self.v_tr:
            img = self.v_tr(img)
        return img

    @functools.lru_cache(maxsize=None)  # Cache loaded structures
    def __getitem__(self, index):
        v = self._v_load(self.paths[index])
        return v