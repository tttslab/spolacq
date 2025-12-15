import numpy as np


def reward_function(current_env, action_waveform, next_env, is_print=False) -> np.ndarray:
    """
    Calculate reward based on how well the selected food's attributes match the internal state.

    Args:
        current_env: Current environment state
        next_env: Next environment state
        is_print: Whether to print debug information

    Returns:
        float: Reward value between 0 and 1
    """
    if is_print:
        print("Internal State:", current_env.internal_state)
        print("Selected Item:", next_env.dialogue_feedback.selected_item)

    # If no item was selected, return 0 reward
    if next_env.dialogue_feedback.selected_item is None:
        return 0

    # Get the selected item's attributes
    selected_item = next_env.dialogue_feedback.selected_item

    # If the selected item is not a food (e.g., it's a corner), return 0 reward
    if selected_item.attributes.get("type") != "food":
        return 0

    # Get the attribute values
    flavor_intensity = selected_item.attributes.get("flavor_intensity", 0)
    volume = selected_item.attributes.get("volume", 0)

    # Internal state should have two values corresponding to flavor_intensity and volume preferences
    internal_flavor_intensity = current_env.internal_state[0]
    internal_volume = current_env.internal_state[1]

    if is_print:
        print(f"Food Attributes - Flavor Intensity: {flavor_intensity}, Volume: {volume}")
        print(f"Internal State - Flavor Intensity: {internal_flavor_intensity}, Volume: {internal_volume}")

    # Calculate absolute differences
    flavor_diff = abs(flavor_intensity - internal_flavor_intensity)
    volume_diff = abs(volume - internal_volume)

    # Calculate average difference
    avg_diff = (flavor_diff + volume_diff) / 2

    # Calculate reward (1 - average difference)
    reward = 1 - avg_diff

    if is_print:
        print(f"Flavor difference: {flavor_diff}")
        print(f"Volume difference: {volume_diff}")
        print(f"Final reward: {reward}")

    return reward


def termination_function(current_env, next_env):
    if next_env.dialogue_scene.scene_id == 3:
        terminated = True
    else:
        terminated = False
    return terminated


def truncation_function(next_env):
    truncated = False
    return truncated


def internal_state_update_function(internal_state, dialogue_feedback):
    return internal_state


def observation_function(next_env):
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


def initial_internal_state_function():
    initial_internal_state = np.random.random(2).astype(np.float32)
    return initial_internal_state
