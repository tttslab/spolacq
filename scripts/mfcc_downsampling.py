import pickle as pickle
import scipy.signal as signal

n = 10 # number of mfccs to keep

mfccs_path = '../exp/pkls/mfccs.pkl'
seglist_path = '../tools/syllables/seglist/seglist.pkl'

embed_dur_dic_path = '../exp/pkls/embed_dur_dic.pkl'

def main():
    with open(mfccs_path, 'rb') as handle:
        mfccs = pickle.load(handle)
    with open(seglist_path, 'rb') as handle:
        seglist = pickle.load(handle)

    embed_dur_dic = {}

    embeddings  = []
    durations   = []

    for m in range(len(mfccs)):
        embeddings_per_wav  = []
        durations_per_wav   = []
        for i, j in seglist[m]:
            y = mfccs[m][i:j+1, :].T
            y_new = signal.resample(y, n, axis=1).flatten("C")
            embeddings_per_wav.append(y_new)
            durations_per_wav.append(j + 1 - i)
        embeddings.append(embeddings_per_wav)
        durations.append(durations_per_wav)

    embed_dur_dic['embeddings'] = embeddings
    embed_dur_dic['durations']  = durations

    with open(embed_dur_dic_path, "wb") as f:
        pickle.dump(embed_dur_dic, f, -1)
        
main()