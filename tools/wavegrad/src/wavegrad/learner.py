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
import torch
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model import WaveGrad


def _nested_map(struct, map_fn):
  if isinstance(struct, tuple):
    return tuple(_nested_map(x, map_fn) for x in struct)
  if isinstance(struct, list):
    return [_nested_map(x, map_fn) for x in struct]
  if isinstance(struct, dict):
    return { k: _nested_map(v, map_fn) for k, v in struct.items() }
  return map_fn(struct)


def sample_noise_scale(num, noise_level):
  S = len(noise_level) - 1 #in training, use noise_schedule as l.

  s = torch.randint(1, S + 1, [num], device=noise_level.device)
  l_a, l_b = noise_level[s-1], noise_level[s]
  #noise_scale = l_a + torch.rand(num, device=noise_level.device) * (l_b - l_a) #continuous
  noise_scale = l_b #discrete
  return noise_scale


def forward(model, features, noise_scale, loss_fn):
  for param in model.parameters():
    param.grad = None

  audio = features['audio']
  num_audio = features['num_audio']
  mask = features['mask']

  if noise_scale.ndim == 1: noise_scale = noise_scale.unsqueeze(1)
  noise = torch.randn_like(audio)
  noisy_audio = noise_scale * audio + (1.0 - noise_scale**2)**0.5 * noise

  predicted = model(noisy_audio, num_audio, noise_scale.squeeze(1))
  #loss = loss_fn(noise, predicted.squeeze(1)) #with loss_fn(reduction='mean')
  loss = loss_fn(noise * mask, predicted.squeeze(1) * mask) / mask.sum() #with loss_fn(reduction='sum')

  return loss


def optimize(model, loss, optimizer, max_grad_norm, scaler=None):
  for param in model.parameters():
    param.grad = None
  if scaler is None:
    loss.backward()
    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
  else:
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    scaler.step(optimizer)
    scaler.update()

  return grad_norm


class WaveGradLearner:
  def __init__(self, model_dir, model, dataset, optimizer, params, *args, **kwargs):
    os.makedirs(model_dir, exist_ok=True)
    self.model_dir = model_dir
    self.model = model
    self.dataset = dataset
    self.optimizer = optimizer
    self.params = params
    self.autocast = torch.cuda.amp.autocast(enabled=kwargs.get('fp16', False))
    self.scaler = torch.cuda.amp.GradScaler(enabled=kwargs.get('fp16', False))
    self.step = 0
    
    beta = np.array(self.params.noise_schedule)
    noise_level = np.cumprod(1 - beta)**0.5
    noise_level = np.concatenate([[1.0], noise_level], axis=0)
    self.noise_level = torch.tensor(noise_level.astype(np.float32))
    #self.loss_fn = nn.L1Loss()
    self.loss_fn = nn.L1Loss(reduction='sum')
    self.summary_writer = None

  def state_dict(self):
    return {
        'step': self.step,
        'model': { k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in self.model.state_dict().items() },
        'optimizer': { k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in self.optimizer.state_dict().items() },
        'params': dict(self.params),
        'scaler': self.scaler.state_dict(),
    }

  def load_state_dict(self, state_dict):
    self.model.load_state_dict(state_dict['model'])
    self.optimizer.load_state_dict(state_dict['optimizer'])
    self.scaler.load_state_dict(state_dict['scaler'])
    self.step = state_dict['step']
    #params is not loaded automatically to enable to change parameters in different training phases.
    #self.params.override(state_dict['params'])

  def save_to_checkpoint(self, filename):
    #save_basename = f'{filename}-{self.step}.pt'
    save_basename = filename
    save_name = f'{self.model_dir}/{save_basename}'
    #link_name = f'{self.model_dir}/{filename}.pt'
    link_name = f'{self.model_dir}/weights.pt'
    torch.save(self.state_dict(), save_name)
    if os.name == 'nt':
      torch.save(self.state_dict(), link_name)
    else:
      if os.path.islink(link_name):
        os.unlink(link_name)
      os.symlink(save_basename, link_name)

  def restore_from_checkpoint(self, filename='weights'):
    try:
      checkpoint = torch.load(f'{self.model_dir}/{filename}.pt')
      self.load_state_dict(checkpoint)
      return True
    except FileNotFoundError:
      return False

  def train(self):
    device = next(self.model.parameters()).device
    while True:
      for features in tqdm(self.dataset, desc=f'Epoch {self.step // len(self.dataset)}', disable=True):
        features = _nested_map(features, lambda x: x.to(device) if isinstance(x, torch.Tensor) else x)
        loss = self.train_step(features)
        if torch.isnan(loss).any():
          raise RuntimeError(f'Detected NaN loss at step {self.step}.')
        self.step += 1
        #if self.step % 100 == 0:
        self._write_summary(self.step, features, loss)
        if self.step % (len(self.dataset) * self.params.save_interval_epoch) == 0: self.save_to_checkpoint(f'weights-{self.step // len(self.dataset)}ep.pt')
        if self.step == (len(self.dataset) * self.params.train_epoch): #if epoch == xxx, break
          self.save_to_checkpoint(f'weights-{self.step // len(self.dataset)}ep.pt')
          exit()

  def train_step(self, features):
    #for param in self.model.parameters():
    #  param.grad = None

    audio = features['audio']
    N = audio.shape[0]
    self.noise_level = self.noise_level.to(audio.device)

    #N, T = audio.shape
    #device = audio.device

    with self.autocast:
      noise_scale = sample_noise_scale(N, self.noise_level)
      loss = forward(self.model, features, noise_scale, self.loss_fn)
      #s = torch.randint(1, S + 1, [N], device=audio.device)
      #l_a, l_b = self.noise_level[s-1], self.noise_level[s]
      ##noise_scale = l_a + torch.rand(N, device=audio.device) * (l_b - l_a) #continuous
      #noise_scale = l_b #discrete
      #noise_scale = noise_scale.unsqueeze(1)
      #noise = torch.randn_like(audio)
      #noisy_audio = noise_scale * audio + (1.0 - noise_scale**2)**0.5 * noise

    self.grad_norm = optimize(self.model, loss, self.optimizer, self.params.max_grad_norm, self.scaler)

    #self.scaler.scale(loss).backward()
    #self.scaler.unscale_(self.optimizer)
    #self.grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.params.max_grad_norm)
    #self.scaler.step(self.optimizer)
    #self.scaler.update()
    return loss

  def _write_log(self, step, val, key):
    writer = self.summary_writer or SummaryWriter(self.model_dir, purge_step=step)
    writer.add_scalar(key, val, step)
    writer.flush()
    self.summary_writer = writer
    

  def _write_summary(self, step, features, loss):
    writer = self.summary_writer or SummaryWriter(self.model_dir, purge_step=step)
    #writer.add_audio('audio/reference', features['audio'][0], step, sample_rate=self.params.sample_rate)
    writer.add_scalar('train/loss', loss, step)
    writer.add_scalar('train/grad_norm', self.grad_norm, step)
    writer.flush()
    self.summary_writer = writer


def train(dataset, args, params):
  torch.backends.cudnn.benchmark = True

  model = WaveGrad(params, torch.nn.Embedding(params.num_action, 50)).cuda()
  opt = torch.optim.Adam(model.parameters(), lr=params.learning_rate)

  learner = WaveGradLearner(args.model_dir, model, dataset, opt, params, fp16=params.fp16)
  learner.restore_from_checkpoint()
  if os.path.islink(f'{args.model_dir}/weights.pt'):
    with open(args.load_filename_log, 'w') as f:
      f.write(os.readlink(f'{args.model_dir}/weights.pt'))
  learner.train()
