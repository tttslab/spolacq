import argparse
import random
import librosa
from pathlib import Path
import soundfile as sf

from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("load_dir", type=str)
parser.add_argument("save_dir", type=str)
args = parser.parse_args()

random.seed(202006241517)

def get_sampling_rate(audio_path: str) -> int:
    import wave
    with wave.open(audio_path, "rb") as f:
        return f.getframerate()

def rm_voiceless(audio_path: str, save_path: str):
    sr = get_sampling_rate(audio_path)
    sound, _ = librosa.core.load(audio_path, sr = sr)
    sound, _ = librosa.effects.trim(sound)
    sf.write(save_path, sound ,sr)

audio_dir = args.load_dir
save_dir = args.save_dir

paths = [i for i in Path(audio_dir).glob("*.wav")]
for p in tqdm(paths, desc="rm_voiceless"):
    savep = Path(save_dir) / p.name
    rm_voiceless(str(p), str(savep))