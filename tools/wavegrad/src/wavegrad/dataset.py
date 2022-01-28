# We modified this file.
# The original one is available at https://github.com/lmnt-com/wavegrad
# 
# Shinozaki Lab Tokyo Tech
# http://www.ts.ip.titech.ac.jp/
# 2022
# ==============================================================================

# Copyright 2020 LMNT, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np
import os
import random
import torch
import torchaudio
import h5py

from glob import glob
from torch.utils.data.distributed import DistributedSampler


class NumpyDataset(torch.utils.data.Dataset):
  def __init__(self, args):
    super().__init__()
    self.data = []
    with open(args.data_list, 'r') as f, h5py.File(args.audio_dir, 'r') as g:
      for record in f:
        action, wavname = record.strip().split(',')
        action = int(action)
        wav = torch.Tensor(g[wavname][()])
        self.data.append((action, wav))

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    action, signal = self.data[idx]
    return {
        #'audio': signal[0] / 32767.5,
        'audio': signal[0],
        'num_audio': action,
    }


class Collator:
  def __init__(self, params):
    self.params = params

  def collate(self, minibatch):
    num_frames = []
    for record in minibatch:
      num_frames.append(len(record['audio']))
      if self.params.wave_len > len(record['audio']):
        padlen = self.params.wave_len - len(record['audio'])
        if self.params.random_pad:
          i = random.randint(0, padlen)
          record['audio'] = np.pad(record['audio'], (i, padlen-i), mode='constant')
        else:
          record['audio'] = np.pad(record['audio'], (0, padlen), mode='constant')
      else:
        raise ValueError("wav is too long")
      start = 0
      end = start + self.params.wave_len

      record['audio'] = record['audio'][start:end]

    masks = []
    for num_frame in num_frames:
      mask = torch.cat((torch.ones(num_frame), torch.zeros(self.params.wave_len - num_frame)))
      masks.append(mask)

    audio = np.stack([record['audio'] for record in minibatch if 'audio' in record])
    num_audio = np.stack([record['num_audio'] for record in minibatch if 'num_audio' in record])
    return {
        'audio': torch.from_numpy(audio),
        'num_audio': torch.from_numpy(num_audio),
        'mask': torch.stack(masks),
    }


def from_path(args, params):
  dataset = NumpyDataset(args)
  return torch.utils.data.DataLoader(
      dataset,
      batch_size=params.batch_size,
      collate_fn=Collator(params).collate,
      shuffle=True,
      num_workers=os.cpu_count(),
      pin_memory=True)
      #drop_last=True)
