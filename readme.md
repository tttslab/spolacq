Spoken Language Acquisition From Conversation Based On Reinforcement Learning
=====================================================



Overview
--------
This repository contains codes for the papers:

- [Spoken Language Acquisition Based on Reinforcement Learning and Word Unit Segmentation](https://ieeexplore.ieee.org/abstract/document/9053326)
- [Sound-Image Grounding Based Focusing Mechanism for Efficient Automatic Spoken Language Acquisition](http://www.interspeech2020.org/uploadfile/pdf/Thu-2-4-4.pdf)
- [Pronunciation adaptive self speaking agent using WaveGrad](https://aaai-sas-2022.github.io/)
- [Self-Supervised Spoken Question Understanding and Speaking With Automatic Vocabulary Learning](https://ieeexplore.ieee.org/abstract/document/9660413)


Unsupervised syllable boundary detection
----------------------------------------
We use the unsupervised syllable boundary detection algorithm described in:

- O. J. Räsänen, G. Doyle, and M. C. Frank, "Unsupervised word discovery from
  speech using automatic segmentation into syllable-like units," in *Proc.
  Interspeech*, 2015.


Unsupervised segmentation and clustering
----------------------------------------
Word segmentation and clustering is performed using the
[ES-KMeans](https://github.com/kamperh/eskmeans/tree/master/eskmeans) package. 


Unsupervised sound-image correspondence learning
------------------------------------------------
We use the unsupervised sound-image correspondence learning described in:

- D. Harwath, A. Torralba, and J. Glass,
  "Unsupervised learning of spoken language with visual context," in *Proc. NIPS*, 2016.


Self-supervised pronunciation adaptation
----------------------------------------
As the agent's speech organ, we use the neural vocoder described in:

- N. Chen and Y. Zhang and H. Zen and R. J. Weiss and M. Norouzi and W. Chan,
  "WaveGrad: Estimating Gradients for Waveform Generation," in *Proc. ICLR*, 2021.

For implementation, we use the [wavegrad](https://github.com/lmnt-com/wavegrad) package.


Dependencies
------------
- [Python](https://www.python.org/)
- [Matlab](https://www.mathworks.com/): Used for syllable boundary detection.


About Author
-------------
- [Shinozaki Lab Tokyo Tech](http://www.ts.ip.titech.ac.jp/)

Usage
-----
Create your combined_sounds.wav from your own data (optional)
```
cd scripts
python prep_combine_wavs.py --path=PATH_TO_RAW_AUDIOS_FOLDER
```
Run main script for 3D homing task
```
cd egs
bash run.bash DATA_NAME(arbitrary) MAX_K
```
Run main script for food task
```
conda env create -f=spolacq.yml
conda activate spolacq
cd egs
sh runfood.sh DATA_DIR_NAME(arbitrary)
```
Run main script for food task with WaveGrad speech organ-based agent
```
conda env create -f=spolacq.yml
conda activate spolacq
cd egs
sh runwavegrad.sh DATA_DIR_NAME(arbitrary)
```
If you are having [trouble with building box2d-py](https://github.com/openai/gym/issues/218), please try
```
sudo apt install xvfb xorg-dev libsdl2-dev swig cmake
```

License
-------
Copyright (C) 2020, 2021 Shinozaki Lab Tokyo Tech

This program is free software: you can redistribute it and/or modify  
it under the terms of the GNU General Public License as published by  
the Free Software Foundation, either version 3 of the License, or  
(at your option) any later version.

This program is distributed in the hope that it will be useful,  
but WITHOUT ANY WARRANTY; without even the implied warranty of  
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the  
GNU General Public License for more details.

You should have received a copy of the GNU General Public License  
along with this program.  If not, see <https://www.gnu.org/licenses/>.


The following programs are APIs for [ES-KMeans](https://github.com/kamperh/eskmeans/tree/master/eskmeans).  
We have copied and modified some of the code available at
https://github.com/kamperh/eskmeans,  
which is released under GNU General Public License version 3.

- [exercises/exercise1.ipynb](exercises/exercise1.ipynb)
- [scripts/mfcc_downsampling.py](scripts/mfcc_downsampling.py)
- [tools/syllables/get_landmarks.py](tools/syllables/get_landmarks.py)
- [tools/syllables/get_seglist_from_landmarks.py](tools/syllables/get_seglist_from_landmarks.py)
- [utils/eskmeans_api.py](utils/eskmeans_api.py)