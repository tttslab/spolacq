import logging
import json
from pathlib import Path
from typing import Any, Optional
import numpy as np
from PIL import Image
import soundfile as sf
from collections import deque

from .data_models import DialogueScene, DialogueFeedback, EnvironmentState, SavedInteraction, Item, FeedbackItem


class DataLogger:
    def __init__(self, images_dir: Path = Path("./images"), audio_dir: Path = Path("./audio"), log_dir: Path = Path("./logs"), sampling_rate: int = 22050):
        self.images_dir: Path = Path(images_dir)
        self.audio_dir: Path = Path(audio_dir)
        self.log_dir: Path = Path(log_dir)
        self.sampling_rate = sampling_rate
        self.item_dict: dict[int, Item] = {}
        self.saved_interaction_counter: dict[str, int] = {"train": 0, "test": 0}
        self.scene_image_queues: dict[str, deque[list[Optional[str]]]] = {
            "train": deque(maxlen=2),
            "test": deque(maxlen=2),
        }
        # Create directories for saving data
        (self.images_dir / "train").mkdir(parents=True, exist_ok=True)
        (self.images_dir / "test").mkdir(parents=True, exist_ok=True)
        (self.audio_dir / "train").mkdir(parents=True, exist_ok=True)
        (self.audio_dir / "test").mkdir(parents=True, exist_ok=True)
        (self.log_dir / "train").mkdir(parents=True, exist_ok=True)
        (self.log_dir / "test").mkdir(parents=True, exist_ok=True)

        # Lists for storing SavedInteraction instances
        self.saved_interaction_lists: dict[str, list[SavedInteraction]] = {"train": [], "test": []}

    def log_interaction(
        self,
        current_env: EnvironmentState,
        action_waveform: np.ndarray,
        next_env: EnvironmentState,
        reward: float,
        mode: str = "train",
        dialogue_counter: int = 1,
    ):
        if mode not in ["train", "test"]:
            logging.warning(f"Mode '{mode}' is not recognized. Using 'train' folder.")
            mode = "train"
        (self.audio_dir / mode / str(dialogue_counter)).mkdir(parents=True, exist_ok=True)

        self.saved_interaction_counter[mode] += 1

        # Convert EnvironmentState instances to dict
        current_env_dict = self._encode_environment_state(
            current_env, True, mode=mode, dialogue_counter=dialogue_counter
        )
        next_env_dict = self._encode_environment_state(next_env, False, mode=mode, dialogue_counter=dialogue_counter)

        # Save action waveform as a WAV file and get file path
        action_waveform_path = self.save_audio(
            action_waveform, f"{self.saved_interaction_counter[mode]}_action-waveform", mode, dialogue_counter
        )

        # Create SavedInteraction instance and append it to the list
        saved_interaction = SavedInteraction(current_env_dict, action_waveform_path, next_env_dict, reward)
        self.saved_interaction_lists[mode].append(saved_interaction)

    def load_interaction(
        self, saved_interaction: SavedInteraction
    ) -> tuple[EnvironmentState, np.ndarray, EnvironmentState, float]:
        # Extract information from SavedInteraction
        current_env_dict = saved_interaction.environment_state_dict
        action_waveform_path = saved_interaction.action_waveform_path
        next_env_dict = saved_interaction.next_environment_state_dict
        reward = saved_interaction.reward

        # Convert dicts back to EnvironmentState instances
        current_env = self._decode_environment_state(current_env_dict)
        next_env = self._decode_environment_state(next_env_dict)

        # Load action waveform from WAV file
        action_waveform, sampling_rate = sf.read(action_waveform_path)
        if sampling_rate != self.sampling_rate:
            raise ValueError(f"Sampling rate of '{action_waveform_path}' must be {self.sampling_rate}")

        return current_env, action_waveform, next_env, reward

    def save_image(self, image: np.ndarray, item: Item, image_index: int = 0, mode: str = "train"):
        if mode not in ["train", "test"]:
            logging.warning(f"Mode '{mode}' is not recognized. Using 'train' folder.")
            mode = "train"
        sub_dir = self.images_dir / mode
        # Convert to PIL Image and save
        pil_img = Image.fromarray(image.astype("uint8"))  # Convert data type to uint8
        image_path = sub_dir / f"{item.id}_{item.name}_{image_index}.png"
        pil_img.save(image_path)
        logging.info(f"Image for '{item.name}' (index={image_index}) saved at '{image_path}'.")

    def load_image(self, item: Item, image_index: int = 0, mode: str = "train") -> Optional[np.ndarray]:
        # Load an image
        if mode not in ["train", "test"]:
            logging.warning(f"Mode '{mode}' is not recognized. Using 'train' folder.")
            mode = "train"
        sub_dir = self.images_dir / mode
        image_path = sub_dir / f"{item.id}_{item.name}_{image_index}.png"
        if not image_path.exists():
            logging.error(f"Image for '{item.name}' not found at '{image_path}'.")
            return None
        pil_img = Image.open(image_path)
        image = np.array(pil_img)
        logging.info(f"Image for '{item.name}' (index={image_index}) loaded from '{image_path}'.")
        return image
    
    def save_audio(self, waveform: np.ndarray, file_name: str, mode: str = "train", dialogue_counter: int = 1) -> Path:
        if mode not in ["train", "test"]:
            logging.warning(f"Mode '{mode}' is not recognized. Using 'train' folder.")
            mode = "train"
        sub_dir = self.audio_dir / mode / str(dialogue_counter)
        audio_path = sub_dir / f"{file_name}.wav"
        sampling_rate = self.sampling_rate
        sf.write(audio_path, waveform, sampling_rate)
        logging.info(f"Audio saved at '{audio_path}'.")

        return audio_path

    def save_logfile(self, file_name: str, mode: str = "train"):
        # Convert each SavedInteraction instance to a dictionary
        saved_interaction_dict_list = []
        for saved_interaction in self.saved_interaction_lists[mode]:
            saved_interaction_dict_list.append({
                "environment_state": saved_interaction.environment_state_dict,
                "action_waveform_path": str(saved_interaction.action_waveform_path),
                "next_environment_state": saved_interaction.next_environment_state_dict,
                "reward": float(saved_interaction.reward)
            })
        
        # Save list as a JSON file
        if mode not in ["train", "test"]:
            logging.warning(f"Mode '{mode}' is not recognized. Using 'train' folder.")
            mode = "train"
        sub_dir = self.log_dir / mode
        log_path = sub_dir / f"{file_name}.json"
        with open(log_path, "w", encoding='utf-8') as file:
            json.dump(saved_interaction_dict_list, file, indent=2, ensure_ascii=False)
        logging.info(f"Interaction log saved at '{log_path}'.")
        # Reset lists and counters
        self.saved_interaction_lists[mode] = []
        self.saved_interaction_counter[mode] = 0
    
    def load_logfile(self, file_name: str, mode: str = "train") -> Optional[list[SavedInteraction]]:
        # Load SavedInteraction information from JSON file
        if mode not in ["train", "test"]:
            logging.warning(f"Mode '{mode}' is not recognized. Using 'train' folder.")
            mode = "train"
        sub_dir = self.log_dir / mode
        log_path = sub_dir / f"{file_name}.json"
        if not log_path.exists():
            logging.error(f"logfile not found at '{log_path}'.")
            return None
        with open(log_path, "r", encoding="utf-8") as file:
            saved_interaction_dict_list = json.load(file)
        logging.info(f"Interaction log loaded from '{log_path}'.")

        # Convert dicts to SavedInteraction instances
        saved_interaction_list = []
        for saved_interaction_dict in saved_interaction_dict_list:
            environment_state_dict = saved_interaction_dict["environment_state"]
            action_waveform_path = Path(saved_interaction_dict["action_waveform_path"])
            next_environment_state_dict = saved_interaction_dict["next_environment_state"]
            reward = saved_interaction_dict["reward"]
            saved_interaction = SavedInteraction(
                environment_state_dict, action_waveform_path, next_environment_state_dict, reward
            )
            saved_interaction_list.append(saved_interaction)
        return saved_interaction_list
    
    def register_scene_paths(self, paths: list[Optional[Path]], mode: str):
        fixed = [str(p) if p is not None else None for p in paths]
        self.scene_image_queues[mode].append(fixed)

    def _encode_environment_state(
        self, environment_state: EnvironmentState, is_current_env: bool, mode: str = "train", dialogue_counter: int = 1
    ) -> dict[str, Any]:
        if is_current_env:
            state_label = "current"
        else:
            state_label = "next"

        # Convert DialogueScene instance to dict
        scene_id = environment_state.dialogue_scene.scene_id
        prompt_waveform_path = self.save_audio(
            environment_state.dialogue_scene.prompt_waveform,
            f"{self.saved_interaction_counter[mode]}_{state_label}-env_prompt-waveform",
            mode,
            dialogue_counter,
        )  # Save prompt waveform as WAV file
        scene_image_queue = self.scene_image_queues[mode]
        if is_current_env:
            image_paths = scene_image_queue[0] if len(scene_image_queue) >= 1 else []
        else:
            image_paths = scene_image_queue[1] if len(scene_image_queue) >= 2 else []

        dialogue_scene_dict = {
            "scene_id": scene_id,
            "prompt_waveform_path": str(prompt_waveform_path),
            "image_paths": image_paths
        }

        # Convert DialogueFeedback instance to dict
        if environment_state.dialogue_feedback is not None:
            if environment_state.dialogue_feedback.selected_item is not None:
                if is_current_env:
                    image_candidates = self.saved_interaction_lists[mode][-1].environment_state_dict["dialogue_scene"][
                        "image_paths"
                    ]
                else:
                    image_candidates = self.scene_image_queues[mode][0]
                selected_item_id = environment_state.dialogue_feedback.selected_item.id
                item_name = self.item_dict[selected_item_id].name
                matched_paths = [p for p in image_candidates if item_name in Path(p).name]
                selected_item_image_path = matched_paths[0]
            else:
                selected_item_image_path = None

            response_waveform_path = self.save_audio(
                environment_state.dialogue_feedback.response_waveform,
                f"{self.saved_interaction_counter[mode]}_{state_label}-env_response-waveform",
                mode,
                dialogue_counter,
            )  # Save response waveform as WAV file

            dialogue_feedback = {
                "selected_item_image_path": selected_item_image_path,
                "response_waveform_path": str(response_waveform_path)
            }
        else:
            dialogue_feedback = None

        # Get internal state
        internal_state = environment_state.internal_state.astype(str).tolist()

        # Combine all into dict
        environment_state_dict = {
            "dialogue_scene": dialogue_scene_dict,
            "dialogue_feedback": dialogue_feedback,
            "internal_state": internal_state,
        }
        return environment_state_dict

    def _decode_environment_state(
        self, environment_state_dict: dict[str, Any], mode: str = "train"
    ) -> EnvironmentState:
        # Reconstruct DialogueScene instance
        scene_id = environment_state_dict["dialogue_scene"]["scene_id"]

        prompt_waveform_path = environment_state_dict["dialogue_scene"]["prompt_waveform_path"]
        prompt_waveform, sampling_rate = sf.read(prompt_waveform_path)
        if sampling_rate != self.sampling_rate:
            raise ValueError(f"Sampling rate of '{prompt_waveform_path}' must be {self.sampling_rate}")

        image_paths = environment_state_dict["dialogue_scene"]["image_paths"]
        images = []
        for path in image_paths:
            if path is None:
                # Use a dummy image for missing ones
                images.append(np.zeros((512, 512, 3), dtype=np.uint8))
            else:
                pil_img = Image.open(path)
                images.append(np.array(pil_img))

        dialogue_scene = DialogueScene(scene_id, prompt_waveform, images)

        # Reconstruct DialogueFeedback instance
        if environment_state_dict["dialogue_feedback"] is not None:
            if environment_state_dict["dialogue_feedback"]["selected_item_image_path"] is not None:
                selected_item_image_path = Path(environment_state_dict["dialogue_feedback"]["selected_item_image_path"])
                selected_item_id = self.get_item_id_from_path(selected_item_image_path)
                selected_item = self.item_dict[selected_item_id]
                pil_img = Image.open(selected_item_image_path)
                feedback_item = FeedbackItem(selected_item.id, np.array(pil_img), selected_item.attributes)
            else:
                feedback_item = None

            response_waveform_path = environment_state_dict["dialogue_feedback"]["response_waveform_path"]
            response_waveform, sampling_rate = sf.read(response_waveform_path)
            if sampling_rate != self.sampling_rate:
                raise ValueError(f"Sampling rate of '{response_waveform_path}' must be {self.sampling_rate}")

            dialogue_feedback = DialogueFeedback(feedback_item, response_waveform)
        else:
            dialogue_feedback = None

        # Restore internal state
        internal_state = np.array(environment_state_dict["internal_state"], dtype=np.float32)

        # Construct EnvironmentState instance
        environment_state = EnvironmentState(dialogue_scene, dialogue_feedback, internal_state)
        return environment_state

    def get_item_id_from_path(self, path: str) -> int:
        # Extract filename and remove extension
        filename = os.path.basename(path)
        basename = os.path.splitext(filename)[0]
        # Find first underscore position
        underscore_index = basename.find("_")
        if underscore_index == -1:
            raise ValueError(f"Filename format invalid (missing underscore): {path.name}")
        
        item_id_str = basename[:underscore_index]
        if item_id_str.isdigit():
            return int(item_id_str)
        else:
            raise ValueError(f"Item ID is not numeric: {item_id_str}")
