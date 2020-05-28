import pickle as pickle

N_LANDMARKS_MAX = 4
landmarks_pkl_path = 'landmarks/landmarks_syllable_seg.pkl'
seglist_pkl_path = 'seglist/seglist.pkl'

def main():
    with open(landmarks_pkl_path, "rb") as f:
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

    with open(seglist_pkl_path, "wb") as f:
        pickle.dump(seglist, f, -1)

main()