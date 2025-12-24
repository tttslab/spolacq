from pathlib import Path

from .generative_models import (
    load_generative_models,
    LLMComponent,
    SpeechRecognizerComponent,
    SpeechSynthesizerComponent,
    ImageGeneratorComponent,
)
from .data_logger import DataLogger


def load_common_components(
    base_data_dir: Path, generative_model_config: dict, sampling_rate: int = 22050, seed: int | None = None, device: str = "cuda"
) -> tuple[LLMComponent, SpeechRecognizerComponent, SpeechSynthesizerComponent, ImageGeneratorComponent, DataLogger]:
    common_models = load_generative_models(generative_model_config, sampling_rate, seed=seed, device=device)
    data_logger = DataLogger(
        images_dir=base_data_dir / "images",
        audio_dir=base_data_dir / "audio",
        log_dir=base_data_dir / "logs",
        sampling_rate=sampling_rate,
    )
    return (common_models, data_logger)
