import os
from typing import Any
import numpy as np
import librosa
import torch
import torchaudio
from transformers import AutoTokenizer, AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from diffusers import DiffusionPipeline, DPMSolverSinglestepScheduler
from vllm import LLM, SamplingParams
from kokoro import KPipeline

class LLMComponent:
    def __init__(self, seed: int | None = None, max_model_len: int = 1200, gpu_memory_utilization: float = 0.4, max_new_tokens: int = 150, temperature: float = 0.5, device: str = "cuda"):
        if seed is None:
            seed = 1234
        torch.random.manual_seed(seed)  # Set random seed

        # Load model
        self.llm = LLM(
            model="microsoft/Phi-4-mini-instruct",
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            download_dir="./cache/LLM",
        )
        self.tokenizer = self.llm.get_tokenizer()

        self.generation_args = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
        }

    def generate_response(self, system_prompt: str, user_input: str) -> dict[str, Any]:
        # Prepare inputs for the LLM
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,  # Generate as a raw string without tokenizing
            add_generation_prompt=True,  # Append generation prompt
        )

 

        sampling_params = SamplingParams(
            max_tokens=self.generation_args["max_new_tokens"], temperature=self.generation_args["temperature"]
        )

        output = self.llm.generate(prompt, sampling_params)

        return output[0].outputs[0].text


class SpeechRecognizerComponent:
    def __init__(self, sampling_rate: int = 22050, device: str = "cuda"):
        # Load ASR model
        self.asr_model = AutoModelForSpeechSeq2Seq.from_pretrained(
            "openai/whisper-large-v3-turbo",
            cache_dir="./cache/ASR",
        ).to(device)
        self.asr_processor = AutoProcessor.from_pretrained(
            "openai/whisper-large-v3-turbo",
            cache_dir="./cache/ASR",
        )
        self.asr_model.config.forced_decoder_ids = None
        self.asr_pipe = pipeline(
            "automatic-speech-recognition",
            model=self.asr_model,
            tokenizer=self.asr_processor.tokenizer,
            feature_extractor=self.asr_processor.feature_extractor,
            device=device,
        )
        self.sampling_rate = sampling_rate
        self.device = device
        self.resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16000).to(self.device)

    def speech_to_text(self, waveform: np.ndarray) -> str:
        # Convert speech to text
        waveform = self.resampler(torch.from_numpy(waveform).to(self.device)).cpu().numpy()
        recognized_text = self.asr_pipe(
            {"sampling_rate": 16000, "raw": waveform}, generate_kwargs={"language": "english"}
        )["text"]

        return recognized_text


class SpeechSynthesizerComponent:
    def __init__(self, sampling_rate: int = 22050, device: str = "cuda"):
        # Load TTS model
        self.pipeline = KPipeline(lang_code='a')
        default_sampling_rate = 24000
        if sampling_rate != default_sampling_rate:
            self.resampler = torchaudio.transforms.Resample(orig_freq=default_sampling_rate, new_freq=sampling_rate).to(
                device
            )
        else:
            self.resampler = None

    def text_to_waveform(self, text: str, speaker_description: str, device: str = "cuda") -> np.ndarray:
        # Generate speech
        generator = self.pipeline(text, voice=speaker_description)
        _, _, generation = next(generator)
        generation = generation.numpy().flatten()

        if self.resampler is not None:
            waveform_tensor = torch.from_numpy(generation).unsqueeze(0).to("cuda")
            waveform_tensor = self.resampler(waveform_tensor).squeeze(0)
            waveform = waveform_tensor.cpu().numpy().astype(np.float32)

        return waveform

class ImageGeneratorComponent:
    def __init__(self, num_inference_steps: int = 8, guidance_scale: int = 4, height: int = 512, width: int = 512, device: str = "cuda"):
        # Load model
        self.pipe = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            cache_dir="./cache/image_generator",
        ).to(device)
        self.pipe.enable_model_cpu_offload()
        self.pipe.load_lora_weights(
            "mann-e/Mann-E_Turbo",
            weight_name="manne_turbo.safetensors",
            cache_dir="./cache/image_generator",
        )
        self.pipe.scheduler = DPMSolverSinglestepScheduler.from_config(
            self.pipe.scheduler.config,
            use_karras_sigmas=True,
        )
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.image_height = height
        self.image_width = width

    def generate_image(self, item_name: str, item_prompt: str | None = None) -> np.ndarray:
        # Build prompt
        if item_prompt is None:
            prompt = f"A {item_name} on a white background, uncooked, realistic."
        else:
            prompt = item_prompt

        image_output_type = "np"  # Output format for the image

        # Generate image
        img_arr_float32 = self.pipe(
            prompt=prompt,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            height=self.image_height,
            width=self.image_width,
            output_type=image_output_type,
        ).images[0]
        img_arr_uint8 = (img_arr_float32 * 255).clip(0, 255).astype("uint8")

        return img_arr_uint8


def load_generative_models(
    generative_model_config: dict, sampling_rate: int = 22050, seed: int | None = None, device: str = "cuda"
) -> tuple[LLMComponent, SpeechRecognizerComponent, SpeechSynthesizerComponent, ImageGeneratorComponent]:
    image_generator_params = generative_model_config["image_generator"]
    llm_params = generative_model_config["LLM"]
    speech_recognizer = SpeechRecognizerComponent(sampling_rate, device=device)
    speech_synthesizer = SpeechSynthesizerComponent(sampling_rate, device=device)
    image_generator = ImageGeneratorComponent(num_inference_steps=image_generator_params["num_inference_steps"], guidance_scale=image_generator_params["guidance_scale"], height=image_generator_params["height"], width=image_generator_params["width"], device=device)
    llm = LLMComponent(seed=seed, max_model_len=llm_params["max_model_len"], gpu_memory_utilization=llm_params["gpu_memory_utilization"], max_new_tokens=llm_params["max_new_tokens"], temperature=llm_params["temperature"], device=device)

    return llm, speech_recognizer, speech_synthesizer, image_generator
