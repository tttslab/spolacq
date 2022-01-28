import math
import random

from pydub import AudioSegment
from pydub.generators import SignalGenerator

class RandomSine(SignalGenerator):
    def __init__(self, freq, **kwargs):
        super(RandomSine, self).__init__(**kwargs)
        self.freq = freq

    def generate(self):
        sine_of = (self.freq * 2 * math.pi) / self.sample_rate
        sample_n = random.randint(0, self.sample_rate-1)
        while True:
            yield math.sin(sine_of * sample_n)
            sample_n += 1

def add_sin_noise(sound: AudioSegment, noise_db: int, freq: int, sample_rate: int, clean_dbfs = None):
    from pydub.generators import Sine
    if clean_dbfs is None:
        noise_dbfs = min(sound.dBFS-noise_db, 0)
    else:
        noise_dbfs = min(clean_dbfs-noise_db, 0)
    
    def make_noise(duration: int, dbfs: float) -> AudioSegment:
        return RandomSine(freq=freq, sample_rate=sample_rate).to_audio_segment(duration=duration, volume=dbfs)

    noise = make_noise(len(sound), noise_dbfs)
    sound = sound.overlay(noise)
    
    return sound
