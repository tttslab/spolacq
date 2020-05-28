import speech_recognition as sr 
from os import walk
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_name',   type=str)
parser.add_argument('--idx',   type=str)
args = parser.parse_args()

wav_dir_path = '../exp/segmented_wavs_' + args.data_name

_, _, filenames = next(walk(wav_dir_path), (None, None, []))

f = open(wav_dir_path + '/stt_results.txt', 'a')

r = sr.Recognizer()

for i, filename in enumerate(filenames):

    if ".wav" not in filename:
        continue

    if i < int(args.idx):
        continue
    
    with sr.AudioFile(wav_dir_path + '/' + filename) as source:
        audio = r.record(source)

    for j in range(5):
        try: 
            recog_result = r.recognize_google(audio)
            f.write(filename + ' ' + recog_result + '\n')
            break

        except sr.UnknownValueError: 
            pass

        if 4 == j:
            f.write(filename + ' ' + "Google Speech Recognition could not understand audio" + '\n')


f.close()