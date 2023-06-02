Spoken Language Acquisition From Conversation Based On Reinforcement Learning
=====================================================



Overview
--------
Master branch contains codes for the paper:
- Continuous Action Space-based Spoken Language Acquisition Agent Using Residual Sentence Embedding and Transformer Decoder

[spolacq2.1 branch](https://github.com/tttslab/spolacq/tree/spolacq2.1) contains codes for the papers:
- [Spoken Language Acquisition Based on Reinforcement Learning and Word Unit Segmentation](https://ieeexplore.ieee.org/abstract/document/9053326)
- [Sound-Image Grounding Based Focusing Mechanism for Efficient Automatic Spoken Language Acquisition](http://www.interspeech2020.org/uploadfile/pdf/Thu-2-4-4.pdf)
- [Pronunciation adaptive self speaking agent using WaveGrad](https://aaai-sas-2022.github.io/)
- [Self-Supervised Spoken Question Understanding and Speaking With Automatic Vocabulary Learning](https://ieeexplore.ieee.org/abstract/document/9660413)
- [Automatic spoken language acquisition based on observation and dialogue](https://ieeexplore.ieee.org/abstract/document/9817627)

About Author
-------------
- [Shinozaki Lab Tokyo Tech](http://www.ts.ip.titech.ac.jp/)

Usage
-----
```
# setup for python==3.7.12 setuptools==59.5.0 wheel==0.37.1 with CUDA 11.1
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html

# download pretrained HiFi-GAN from https://drive.google.com/drive/folders/1YuOoV3lO2-Hhn1F2HJ2aQ4S0LC1JdKLd
# and place them in the following paths
# - models
#   - hifi-gan
#     - config.json
#     - do_02500000
#     - g_02500000

sh egs/setup.sh
sh egs/run_spolacq3.sh
```
If you have [a trouble with building box2d-py](https://github.com/openai/gym/issues/218), please try the following command:
```
sudo apt install xvfb xorg-dev libsdl2-dev swig cmake
```

License
-------
Copyright (C) 2020- Shinozaki Lab Tokyo Tech

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.