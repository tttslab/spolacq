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

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import numpy as np


class ASR:
    def __init__(self, modelname_or_path: str):
        # Device and Data Type Settings
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        # Model Initialization (Using a Pre-trained Model)
        model_id = "openai/whisper-large-v3-turbo"

        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=self.torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        self.model.to(self.device)

        self.processor = AutoProcessor.from_pretrained(model_id)

        # Make pipeline
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            torch_dtype=self.torch_dtype,
            device=self.device,
        )

    def __call__(self, audio_input: np.ndarray) -> str:
        # Transcription execution
        result = self.pipe(audio_input, chunk_length_s=30, batch_size=8, return_timestamps=True)
        return result["text"]


# import os
# import random
# from typing import List

# import librosa
# import numpy as np
# import torch
# import yaml
# from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor


# class ASR:
#     def __init__(self, modelname_or_path: str, lr: float = 1e-6):
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#         self.processor = Wav2Vec2Processor.from_pretrained(modelname_or_path)

#         self.model = Wav2Vec2ForCTC.from_pretrained(modelname_or_path).to(self.device)
#         self.model.freeze_feature_encoder()
#         self.model.eval()

#         self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

#     @torch.inference_mode()
#     def __call__(self, audio_input: np.ndarray) -> str:
#         # pad input values and return pt tensor
#         input_values = self.processor(audio_input, sampling_rate=16000, return_tensors="pt").input_values

#         # INFERENCE

#         # retrieve logits & take argmax
#         logits = self.model(input_values.to(self.device)).logits
#         predicted_ids = torch.argmax(logits, dim=-1)

#         # transcribe
#         transcript = self.processor.decode(predicted_ids[0])
#         return transcript

#     def save(self, path: str) -> None:
#         self.processor.save_pretrained(path)
#         self.model.save_pretrained(path)

#     def train(self, audio_input: List[np.ndarray], target_transcript: List[str]) -> None:
#         # pad input values and return pt tensor
#         inputs = self.processor(
#             audio_input, padding=True, return_attention_mask=True, sampling_rate=16000, return_tensors="pt"
#         ).to(self.device)

#         # encode labels
#         with self.processor.as_target_processor():
#             labels = self.processor(target_transcript, padding=True, return_tensors="pt").input_ids
#             labels = labels.masked_fill(labels.eq(0), -100)

#         # compute loss by passing labels
#         loss = self.model(inputs["input_values"], attention_mask=inputs["attention_mask"], labels=labels).loss
#         self.optimizer.zero_grad()
#         loss.backward()
#         self.optimizer.step()


# obj2color = {
#     "apple": "red",
#     "banana": "yellow",
#     "carrot": "orange",
#     "cherry": "black",
#     "cucumber": "green",
#     "egg": "chicken",
#     "eggplant": "purple",
#     "green_pepper": "green",
#     "hyacinth_bean": "green",
#     "kiwi_fruit": "brown",
#     "lemon": "yellow",
#     "onion": "yellow",
#     "orange": "orange",
#     "potato": "brown",
#     "sliced_bread": "yellow",
#     "small_cabbage": "green",
#     "strawberry": "red",
#     "sweet_potato": "brown",
#     "tomato": "red",
#     "white_radish": "white",
# }


# def add_indications(name, idx):
#     preposition = "an" if name[0] in ["a", "o", "e"] else "a"
#     color = obj2color[name]
#     preposition2 = "an" if color[0] in ["a", "o", "e"] else "a"
#     ll = [name, f"{preposition} {name}", f"{preposition2} {color} {name}", f"i want {preposition} {name}"]
#     return ll[idx].replace("_", " ").upper()


# class TrainSet:
#     def __init__(self):
#         self.audios = []
#         self.target_transcripts = []

#         for food in obj2color:
#             for i in range(4):
#                 for j in range(90):
#                     path = f"../data/I2U/audio/{food}/{food}_{i}_{j}.wav"
#                     audio, _ = librosa.load(path, sr=16000)
#                     self.audios.append(audio)

#                     target_transcript = add_indications(food, i)
#                     self.target_transcripts.append(target_transcript)

#     def __len__(self):
#         return len(self.audios)


# def make_train_loader(dataset: TrainSet, batch_size: int):
#     dataset_size = len(dataset)

#     indices = random.sample(list(range(dataset_size)), k=batch_size * (dataset_size // batch_size))

#     data_loader = []
#     for batch_id in range(dataset_size // batch_size):
#         batch_audios = []
#         batch_target_transcripts = []

#         for index in indices[batch_size * batch_id : batch_size * (batch_id + 1)]:
#             batch_audios.append(dataset.audios[index])
#             batch_target_transcripts.append(dataset.target_transcripts[index])

#         data_loader.append((batch_audios, batch_target_transcripts))
#     return data_loader


# def make_val_loader():
#     data_loader = []

#     for food in obj2color:
#         for i in range(4):
#             for j in range(90, 110):
#                 path = f"../data/I2U/audio/{food}/{food}_{i}_{j}.wav"
#                 audio, _ = librosa.load(path, sr=16000)

#                 target_transcript = food.replace("_", " ").upper()

#                 data_loader.append((audio, target_transcript))

#     return data_loader


# def train(model, train_loader):
#     model.model.train()

#     for audio, target_transcript in train_loader:
#         model.train(audio, target_transcript)


# def validate(model, val_loader):
#     model.model.eval()
#     accuracy = 0

#     for audio, target_transcript in val_loader:
#         transcript = model(audio)
#         if target_transcript in transcript:
#             accuracy += 1
#     return accuracy / len(val_loader)


# if __name__ == "__main__":
#     with open("../conf/spolacq3.yaml") as y:
#         config = yaml.safe_load(y)

#     asr_dir = os.path.join(os.path.dirname(__file__), "..", config["ASR"]["dir"])

#     if not os.path.isdir(asr_dir):
#         model = ASR(config["ASR"]["model"], lr=config["ASR"]["lr"])
#         train_set = TrainSet()
#         val_loader = make_val_loader()
#         max_accuracy = 0

#         for epoch in range(1, 1 + config["ASR"]["epoch"]):
#             train_loader = make_train_loader(train_set, config["ASR"]["batch_size"])

#             train(model, train_loader)
#             accuracy = validate(model, val_loader)

#             print(f"epoch: {epoch}, accuracy: {accuracy}", flush=True)

#             if max_accuracy < accuracy:
#                 max_accuracy = accuracy
#                 model.save(asr_dir)
