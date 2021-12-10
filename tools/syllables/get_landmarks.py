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
parser.add_argument("--sylseg", type=str, default="thetaOscillator/results/output.csv")
parser.add_argument("--landmarks", type=str, default="landmarks/landmarks_syllable_seg.pkl")
args = parser.parse_args()

def main():
    with open(args.sylseg) as f:
        lineList = f.readlines()

    landmarks = []
    for line in lineList:
        bound_list = line.split()
        landmarks_per_wav = [int(round(float(bound)*100.0)) for bound in bound_list]
        landmarks_per_wav = landmarks_per_wav[1:]
        landmarks.append(landmarks_per_wav)
    
    with open(args.landmarks, "wb") as f:
        pickle.dump(landmarks, f, -1)
    print("landmarks saved to: " + args.landmarks)

main()
