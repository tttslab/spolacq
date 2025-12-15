import numpy as np
from typing import Dict

from env.data_models import EnvironmentState, DialogueFeedback


def reward_function(current_env: EnvironmentState, action_waveform: np.ndarray, next_env: EnvironmentState) -> float :
    internal_state = current_env.internal_state
    leftimage_rgb = np.mean(current_env.dialogue_scene.images[0], axis=(0, 1))
    rightimage_rgb = np.mean(current_env.dialogue_scene.images[1], axis=(0, 1))
    leftimage_distance = np.linalg.norm(leftimage_rgb - internal_state)
    rightimage_distance = np.linalg.norm(rightimage_rgb - internal_state)

    if leftimage_distance < rightimage_distance:
        correct_image = current_env.dialogue_scene.images[0]
    else:
        correct_image = current_env.dialogue_scene.images[1]
    if next_env.dialogue_feedback.selected_item is not None and np.array_equal(
        next_env.dialogue_feedback.selected_item.image, correct_image
    ):
        reward = 1
    else:
        reward = 0
    return reward

def termination_function(current_env: EnvironmentState, next_env: EnvironmentState) -> bool :
    if next_env.dialogue_scene.scene_id == 1:
        terminated = True
    else:
        terminated = False
    return terminated

def truncation_function(next_env: EnvironmentState) -> bool :
    truncated = True
    return truncated

def internal_state_update_function(internal_state: np.ndarray, dialogue_feedback: DialogueFeedback) -> np.ndarray:
    # next_internal_state = np.random.random(3).astype(np.float32)*255
    next_internal_state = internal_state
    return next_internal_state

def observation_function(next_env: EnvironmentState) -> Dict[str, np.ndarray]:
    observation = {}
    observation["state"] = next_env.internal_state
    if len(next_env.dialogue_scene.images) >= 1:
        observation["leftimage"] = next_env.dialogue_scene.images[0]
        observation["rightimage"] = next_env.dialogue_scene.images[1]
    else:
        observation["leftimage"] = np.zeros((224, 224, 3))
        observation["rightimage"] = np.zeros((224, 224, 3))
    # observation["audio"] = next_env.dialogue_scene.prompt_waveform
    return observation

def initial_internal_state_function() -> np.ndarray :
    initial_internal_state = np.random.random(3).astype(np.float32)*255
    return initial_internal_state
