"""
MIT License

Copyright (c) Facebook, Inc. and its affiliates.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC


class ASR:
    def __init__(self, modelname_or_path: str, lr=1e-6):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = Wav2Vec2Processor.from_pretrained(modelname_or_path)
        self.model = Wav2Vec2ForCTC.from_pretrained(modelname_or_path).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
    
    def __call__(self, audio_input: np.ndarray) -> str:
        # pad input values and return pt tensor
        input_values = self.processor(audio_input, sampling_rate=16000, return_tensors="pt").input_values

        # INFERENCE

        # retrieve logits & take argmax
        logits = self.model(input_values.to(self.device)).logits
        predicted_ids = torch.argmax(logits, dim=-1)

        # transcribe
        transcription = self.processor.decode(predicted_ids[0])
        return transcription
    
    def save(self, path: str) -> None:
        self.processor.save_pretrained(path)
        self.model.save_pretrained(path)
    
    def train(self, audio_input: np.ndarray, target_transcription: str) -> None:
        # pad input values and return pt tensor
        input_values = self.processor(audio_input, sampling_rate=16000, return_tensors="pt").input_values

        # encode labels
        with self.processor.as_target_processor():
            labels = self.processor(target_transcription, return_tensors="pt").input_ids

        # compute loss by passing labels
        loss = self.model(input_values.to(self.device), labels=labels).loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()