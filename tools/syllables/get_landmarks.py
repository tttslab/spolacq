import pickle as pickle

def main():
    syllable_seg_result_path = 'thetaOscillator/results/output.csv'
    with open(syllable_seg_result_path) as f:
        lineList = f.readlines()

    landmarks = []
    for line in lineList:
        bound_list = line.split()
        landmarks_per_wav = [int(round(float(bound)*100.0)) for bound in bound_list]
        landmarks_per_wav = landmarks_per_wav[1:]
        landmarks.append(landmarks_per_wav)

    landmarks_pkl_path = 'landmarks/landmarks_syllable_seg.pkl'
    with open(landmarks_pkl_path, "wb") as f:
        pickle.dump(landmarks, f, -1)
    print("landmarks saved to: " + landmarks_pkl_path)

main()
