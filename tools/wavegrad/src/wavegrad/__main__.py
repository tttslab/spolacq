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

from argparse import ArgumentParser
import os
import subprocess

from learner import train
from params import AttrDict
from dataset import from_path as dataset_from_path

import numpy as np
import yaml

def load_yaml(yamlpath):
  with open(yamlpath, 'r') as f:
    conf = yaml.safe_load(f)
  return conf

def num_line(file):
  return int(subprocess.check_output(['wc', '-l', file]).decode().split(' ')[0])

def get_num_action(args):
  used_audios     = f'{args.rl_dir}/used_audios.txt'
  filtered_audios = f'{args.rl_dir}/used_audios_filtered.txt'
  if os.path.isfile(filtered_audios):
    num_action = num_line(filtered_audios)
  elif os.path.isfile(used_audios):
    num_action = num_line(used_audios)
  else:
    raise FileNotFoundError()
  return num_action


def main(args):
  #args: paths.
  #params: parameters such as int, bool, list.

  params = AttrDict()
  params.override(load_yaml(args.conf))
  params.num_action = get_num_action(args)

  schedule = np.linspace(float(params.betamin), float(params.betamax), int(params.betalen)).tolist()
  params.override({'noise_schedule': schedule})

  train(dataset_from_path(args, params), args, params)


if __name__ == '__main__':
  parser = ArgumentParser(description='train (or resume training) a WaveGrad model')
  parser.add_argument('rl_dir', type=str)
  parser.add_argument('model_dir',
      help='directory in which to store model checkpoints and training logs')
  parser.add_argument('conf', type=str)
  args = parser.parse_args()

  args.data_list         = f'{args.rl_dir}/wavegrad_training_list.txt'
  args.audio_dir         = f'{args.rl_dir}/audio_log.h5'
  args.load_filename_log = f'{args.model_dir}/load_filename_log.txt'

  main(args)
