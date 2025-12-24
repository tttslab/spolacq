import importlib
import json

from pathlib import Path
from .register_functions import make_env_base


def get_task_path(task_name: str) -> dict[str, Path]:
    task_dir = Path(__file__).parent / task_name

    task_config_file = task_dir / "task_config.json"
    with open(task_config_file, "r", encoding="utf-8") as file:
        task_config = json.load(file)
    task_config_name = task_config["task_config_name"]

    config_file = task_dir / task_config_name
    return {
        "task_dir": task_dir,
        "config_file": config_file,
        "task_config_file": task_config_file
    }

def get_env(
    task_name,
    data_logger,
    llm,
    speech_recognizer,
    speech_synthesizer,
    image_generator,
    llm_prompt_name,
    mode="train",
    record_mode=False,
    num_images_per_item=1,
    sampling_rate=22050,
    enable_audio_response=True,
    **kwargs,
):
    """
    Create an environment based on the specified task name.

    Parameters
    ----------
    Create an environment based on the specified task name.

    Parameters
    ----------
    task_name : str
        The name of the task (e.g., "food_task").
    data_logger : DataLogger
        The data logger instance.
    llm : LLMComponent
        The LLM instance.
    speech_recognizer : SpeechRecognizerComponent
        The speech recognition component instance.
    speech_synthesizer : SpeechSynthesizerComponent
        The text-to-speech component instance.
    image_generator : ImageGeneratorComponent
        The image generation component instance.
    llm_prompt_name : str
        The name of the LLM prompt to use.
    mode : str
        Either "train" or "test".
    record_mode : bool
        Whether to record dialogues.
    num_images_per_item : int
        Number of images per item.
    sampling_rate : int
        Audio sampling rate.
    enable_audio_response : bool
        Whether to enable audio responses.

    Returns
    -------
        Whether to enable audio responses.

    Returns
    -------
    env : gym.Env
        The environment instance.
    """
    # Validate required parameters
    if data_logger is None:
        raise ValueError("data_logger is required")
    if llm is None:
        raise ValueError("llm is required")
    if speech_recognizer is None:
        raise ValueError("speech_recognizer is required")
    if speech_synthesizer is None:
        raise ValueError("speech_synthesizer is required")
    if image_generator is None:
        raise ValueError("image_generator is required")
    if llm_prompt_name is None:
        raise ValueError("llm_prompt_name is required")

    # Get task configuration file paths
    task_path = get_task_path(task_name)
    task_dir = task_path["task_dir"]
    config_file = task_path["config_file"]
    task_config_file = task_path["task_config_file"]
    
    if not config_file.exists():
        raise FileNotFoundError(f"Task configuration file not found: {config_file}")

    # Load max_step from task_config.json if available
    max_step = 100  # Default value
    if task_config_file.exists():
        with open(task_config_file, "r") as f:
            task_config = json.load(f)
            max_step = task_config.get("max_step", max_step)

    # Import task-specific functions
    try:
        task_functions = importlib.import_module(f".{task_name}.task_functions", package="tasks")
        reward_function = getattr(task_functions, "reward_function", None)
        termination_function = getattr(task_functions, "termination_function", None)
        truncation_function = getattr(task_functions, "truncation_function", None)
        internal_state_update_function = getattr(task_functions, "internal_state_update_function", None)
        observation_function = getattr(task_functions, "observation_function", None)
        initial_internal_state_function = getattr(task_functions, "initial_internal_state_function", None)
    except ImportError as e:
        raise ImportError(f"Failed to import task functions: {e}")

    # Create base environment using make_env_base
    env = make_env_base(
        data_logger=data_logger,
        llm=llm,
        speech_recognizer=speech_recognizer,
        speech_synthesizer=speech_synthesizer,
        image_generator=image_generator,
        config_path=config_file,
        llm_prompt_name=llm_prompt_name,
        mode=mode,
        num_images_per_item=num_images_per_item,
        sampling_rate=sampling_rate,
        enable_audio_response=enable_audio_response,
        initial_internal_state_function=initial_internal_state_function,
        max_step=max_step,
        reward_function=reward_function,
        termination_function=termination_function,
        truncation_function=truncation_function,
        internal_state_update_function=internal_state_update_function,
        observation_function=observation_function,
        record_mode=record_mode,
    )

    # Apply task-specific wrappers if available
    try:
        task_wrappers = importlib.import_module(f".{task_name}.task_wrappers", package="tasks")
        wrap_task_env = getattr(task_wrappers, "wrap_task_env", None)
        if wrap_task_env:
            env = wrap_task_env(env, **kwargs)
    except (ImportError, AttributeError):
        # Ignore if no task-specific wrapper exists
        pass

    return env
