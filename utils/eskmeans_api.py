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
from typing import Optional

import numpy as np
from eskmeans.kmeans import KMeans
from eskmeans.eskmeans_wordseg import ESKmeans


def save(km: KMeans, data_name: Optional[str], txt_path: str, pkl_path: Optional[str]) -> None:
    print("assignments:")
    print(km.assignments)
    print(len(km.assignments))
    print("non-zero assignments")
    print(len(np.where(km.assignments != -1)[0]))
    # list of indice which corresponds to the identified words in seg_list

    result_file = open(txt_path, "w")

    result_dict = {}
    result_list = []
    for group_index in range(km.K_max):
        result_file.write("group " + str(group_index) + " indices:" + "\n")
        group_list = np.where(km.assignments == group_index)[0]
        result_dict[group_index] = group_list
        result_file.write(np.array_str(group_list) + "\n")
        result_file.write("length :" + "\n")
        result_file.write(str(len(group_list)) + "\n")
        for index in group_list:
            result_list.append(index)
    result_file.write("results:" + "\n")
    result_file.write(" ".join(str(e) for e in result_list) + "\n")
    result_file.close()
    
    if data_name:
        print("Writing: " + "results/unsup_seg_assignments.pkl")
        with open("results/" + data_name + ".pkl", "wb") as f:
            pickle.dump(result_dict, f)
    else:
        with open(pkl_path, "wb") as f:
            pickle.dump(result_dict, f)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str)
    parser.add_argument("--max_k", type=int)
    parser.add_argument("--landmarks", type=str, default="../syllables/landmarks/landmarks_syllable_seg.pkl")
    parser.add_argument("--embed", type=str, default="../../exp/pkls/embed_dur_dic.pkl")
    parser.add_argument("--txt_path", type=str, default="results/results.txt")
    parser.add_argument("--pkl_path", type=str)
    args = parser.parse_args()

    # n_slices == num of landmarks
    with open(args.landmarks, "rb") as handle:
        landmarks = pickle.load(handle)
    n_slices = [len(landmarks_per_wav) for landmarks_per_wav in landmarks]
    
    n_slices_max = 4
    n_iter = 100
    p_boundary_init = 1.0

    # get embeddings
    with open(args.embed, "rb") as handle:
        embed_dur_dic = pickle.load(handle)
    
    embeddings = embed_dur_dic["embeddings"]
    durations_raw = embed_dur_dic["durations"]

    # number of wav files
    num_wav_files = len(landmarks)

    # get Vector IDs
    vec_ids = []
    durations = []
    for m in range(num_wav_files):
        vec_ids_per_wav = -1*np.ones((n_slices[m]**2 + n_slices[m])//2, dtype=int)
        durations_per_wav = -1*np.ones((n_slices[m]**2 + n_slices[m])//2, dtype=int)
        i_embed = 0
        for cur_start in range(n_slices[m]):
            for cur_end in range(cur_start, min(n_slices[m], cur_start + n_slices_max)):
                cur_end += 1
                t = cur_end
                i = t*(t - 1)//2
                vec_ids_per_wav[i + cur_start] = i_embed
                durations_per_wav[i + cur_start] = durations_raw[m][i_embed]
                i_embed += 1
        vec_ids.append(vec_ids_per_wav)
        durations.append(durations_per_wav)

    # convert into dics
    embedding_mats = {}
    vec_ids_dict = {}
    durations_dict = {}
    landmarks_dict = {}
    for m in range(num_wav_files):
        embedding_mats[str(m)] = embeddings[m]
        vec_ids_dict[str(m)] = vec_ids[m]
        durations_dict[str(m)] = durations[m]
        landmarks_dict[str(m)] = landmarks[m]


    # Initialize model
    K_max = args.max_k
    segmenter = ESKmeans(
        K_max, embedding_mats, vec_ids_dict, durations_dict, landmarks_dict,
        p_boundary_init=p_boundary_init, n_slices_max=n_slices_max
        )

    # Perform inference
    record = segmenter.segment(n_iter=n_iter)

    # save assignment results
    save(segmenter.acoustic_model, args.data_name, args.txt_path, args.pkl_path)

if __name__ == "__main__":
    main()