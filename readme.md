Spoken Language Acquisition From Conversation Based On Reinforcement Learning
=====================================================



Overview
--------
This repository contains codes for the paper [Spoken Language Acquisition Based on Reinforcement Learning and Word Unit Segmentation](https://ieeexplore.ieee.org/abstract/document/9053326). 


Unsupervised syllable boundary detection
----------------------------------------
We use the unsupervised syllable boundary detection algorithm described in:

- O. J. Räsänen, G. Doyle, and M. C. Frank, "Unsupervised word discovery from
  speech using automatic segmentation into syllable-like units," in *Proc.
  Interspeech*, 2015.


Unsupervised segmentation and clustering
----------------------------------------
Word segmentation and clustering is performed using the
[ESKMeans](https://github.com/kamperh/eskmeans/tree/master/eskmeans) package. 


Dependencies
------------
- [Python](https://www.python.org/)
- [Matlab](https://www.mathworks.com/): Used for syllable boundary detection.


About Author
-------------
- [Shinozaki Lab TITech](http://www.ts.ip.titech.ac.jp/)

Usage
-----
Create your combined_sounds.wav from your own data (optional)
```
cd scripts
python prep_combine_wavs.py --path=PATH_TO_RAW_AUDIOS_FOLDER
```
Run main script
```
cd egs
bash run.bash DATA_NAME(arbitrary) MAX_K
```