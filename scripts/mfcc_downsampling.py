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
import scipy.signal as signal

parser = argparse.ArgumentParser()
parser.add_argument("--mfcc", type=str, default="../exp/pkls/mfccs.pkl")
parser.add_argument("--seglist", type=str, default="../tools/syllables/seglist/seglist.pkl")
parser.add_argument("--embed", type=str, default="../exp/pkls/embed_dur_dic.pkl")
parser.add_argument("--n", type=int, default=10, help="number of mfccs to keep")
args = parser.parse_args()

def main():
    with open(args.mfcc, "rb") as handle:
        mfccs = pickle.load(handle)
    with open(args.seglist, "rb") as handle:
        seglist = pickle.load(handle)

    embed_dur_dic = {}

    embeddings  = []
    durations   = []

    for m in range(len(mfccs)):
        embeddings_per_wav  = []
        durations_per_wav   = []
        for i, j in seglist[m]:
            y = mfccs[m][i:j+1, :].T
            y_new = signal.resample(y, args.n, axis=1).flatten("C")
            embeddings_per_wav.append(y_new)
            durations_per_wav.append(j + 1 - i)
        embeddings.append(embeddings_per_wav)
        durations.append(durations_per_wav)

    embed_dur_dic["embeddings"] = embeddings
    embed_dur_dic["durations"]  = durations

    with open(args.embed, "wb") as f:
        pickle.dump(embed_dur_dic, f, -1)
        
main()