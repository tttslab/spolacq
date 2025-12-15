import gymnasium as gym
from functools import wraps

from env.dialogue_environment_state_manager import AgentStateManager
from env.dialogue_simulator import DialogueSimulator
from .wrappers import wrap_custom_functions, wrap_logging


def make_env_base(
    data_logger,
    llm,
    speech_recognizer,
    speech_synthesizer,
    image_generator,
    config_path,
    llm_prompt_name,
    mode,
    num_images_per_item,
    sampling_rate,
    enable_audio_response,
    initial_internal_state_function,
    max_step,
    reward_function,
    termination_function,
    truncation_function,
    internal_state_update_function,
    observation_function,
    record_mode=False,
):
    print("make_env_base")
    print("config_path:", config_path)
    dialogue_simulator = DialogueSimulator(
        config_path=config_path,
        llm=llm,
        speech_recognizer=speech_recognizer,
        speech_synthesizer=speech_synthesizer,
        image_generator=image_generator,
        data_logger=data_logger,
        llm_prompt_name=llm_prompt_name,
        mode=mode,
        num_images_per_item=num_images_per_item,
        sampling_rate=sampling_rate,
        enable_audio_response=enable_audio_response,
    )

    asm = AgentStateManager(
        dialogue_simulator=dialogue_simulator,
        initial_internal_state_function=initial_internal_state_function,
        max_step=max_step,
    )

    base_env = wrap_custom_functions(
        asm,
        reward_function,
        termination_function=termination_function,
        truncation_function=truncation_function,
        internal_state_update_function=internal_state_update_function,
        observation_function=observation_function,
    )

    base_env = wrap_logging(
        base_env,
        record_mode=record_mode,
        data_logger=data_logger,
        mode=mode,
    )

    return base_env