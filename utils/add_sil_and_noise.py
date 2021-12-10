import argparse
import random
from pydub import AudioSegment
from pathlib import Path
from tqdm import tqdm

def add_voiceless_and_noise(audio_path: str, save_path: str, noise_db: int = 30, voiceless_len_ms: int = 1000):
    from pydub.generators import WhiteNoise
    sound = AudioSegment.from_wav(audio_path)
    noise_dbfs = sound.dBFS-noise_db
    
    def make_sil(duration: int) -> AudioSegment:
        return AudioSegment.silent(duration=duration)
    
    def make_noise(duration: int, dbfs: float) -> AudioSegment:
        return WhiteNoise().to_audio_segment(duration=duration, volume=dbfs)

    noise = make_noise(len(sound), noise_dbfs)
    sound = sound.overlay(noise)
    
    first_noise = make_noise(voiceless_len_ms, noise_dbfs)
    last_noise  = make_noise(voiceless_len_ms, noise_dbfs)
    #first_noise = make_sil(1000)
    #last_noise  = make_sil(1000)
    sound = first_noise + sound + last_noise
    sound.export(save_path, format="wav")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("load_dir", type=str)
    parser.add_argument("save_dir", type=str)
    args = parser.parse_args()

    random.seed(202006241517)

    audio_dir = args.load_dir
    save_dir = args.save_dir
    paths = [i for i in Path(audio_dir).glob("*.wav")]
    for p in tqdm(paths, desc="Add white noise"):
        savep = Path(save_dir) / p.name
        add_voiceless_and_noise(p, savep)