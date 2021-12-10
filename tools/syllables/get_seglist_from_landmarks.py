"""
This program is an API for ES-KMeans [H. Kamper et al., 2017].

H. Kamper, K. Livescu, and S. J. Goldwater,
"An embedded segmental K-means model for unsupervised segmentation and clustering of speech,"
in Proc. ASRU, 2017.

We have copied and modified some of the code available at
https://github.com/kamperh/eskmeans,
which is released under GNU General Public License version 3.

Shinozaki Lab Tokyo Tech
http://www.ts.ip.titech.ac.jp/
2021
"""

import argparse
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("--landmarks", type=str, default="landmarks/landmarks_syllable_seg.pkl")
parser.add_argument("--seglist", type=str, default="seglist/seglist.pkl")
args = parser.parse_args()

N_LANDMARKS_MAX = 4

def main():
    with open(args.landmarks, "rb") as f:
        landmarks = pickle.load(f)
    
    seglist = []
    for m in range(len(landmarks)):
        seglist_per_wav = []
        prev_landmark = 0
        for i in range(len(landmarks[m])):
            for j in landmarks[m][i:i + N_LANDMARKS_MAX]:
                seglist_per_wav.append((prev_landmark, j))
            prev_landmark = landmarks[m][i]
        seglist.append(seglist_per_wav)

    with open(args.seglist, "wb") as f:
        pickle.dump(seglist, f, -1)

main()