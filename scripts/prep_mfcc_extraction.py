'''
    convert to mfcc
'''

import librosa, pickle

wavlist_path = '../tools/syllables/thetaOscillator/wavlist.list'
wavlist_root = '../tools/syllables/thetaOscillator/'

with open(wavlist_path) as f:
    lines = f.readlines()


mfccs = []
for line in lines:
    wav_path = ''
    if '/' == line[0]:
        # if absolute path
        wav_path = line.rstrip()
    else:
        # if relative path
        wav_path = wavlist_root + line.rstrip()
    x, sr = librosa.load(wav_path, sr=44100)
    mfccs_per_wav = librosa.feature.mfcc(x, sr=sr, hop_length=441, n_mfcc=20).T
    print("for wav file " + line.rstrip() + ", mfcc shape:")
    print(mfccs_per_wav.shape)
    mfccs.append(mfccs_per_wav)

with open('../exp/pkls/mfccs.pkl', 'wb') as handle:
    pickle.dump(mfccs, handle)


