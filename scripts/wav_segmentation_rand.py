from pydub import AudioSegment
import argparse
import random

export_dir_path = 'segmented_wav_xyz_wx/'



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--wav_path', type=str)
    args = parser.parse_args()

    # min and max segment length (in milliseconds)
    min_len = 500
    max_len = 1500


    wav_original = AudioSegment.from_wav(args.wav_path)

    total_len = len(wav_original)

    t_start = 0
    count = 0

    while True:
        rand_duration = random.randint(min_len, max_len)

        if t_start + rand_duration >= total_len:
            wav_segment = wav_original[t_start : total_len]
            wav_segment.export(export_dir_path + str(count) + '.wav', format="wav")
            break

        wav_segment = wav_original[t_start : t_start + rand_duration]
        wav_segment.export(export_dir_path + str(count) + '.wav', format="wav")
        count += 1
        t_start = t_start + rand_duration
    


main()