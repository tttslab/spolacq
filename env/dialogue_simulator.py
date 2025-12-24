import yaml
import logging
import random
import json
from typing import Dict, Any, Optional, List
import numpy as np
import re
from pathlib import Path

from .generative_models import (
    LLMComponent,
    SpeechRecognizerComponent,
    SpeechSynthesizerComponent,
    ImageGeneratorComponent,
)
from .data_logger import DataLogger
from .data_models import DialogueScene, DialogueFeedback, Item, FeedbackItem


class DialogueSimulator:
    llm: LLMComponent
    speech_recognizer: SpeechRecognizerComponent
    speech_synthesizer: SpeechSynthesizerComponent
    image_generator: ImageGeneratorComponent
    data_logger: DataLogger
    current_scene: Dict[str, Any]
    current_offered_items: List[Item]
    config: Dict[str, Any]
    scenes: Dict[str, Any]
    items: Dict[str, Any]
    num_images_per_item: int
    enable_audio_response: bool
    prompt_template: str
    mode: str

    def __init__(
        self, 
        config_path: Path, 
        llm: LLMComponent, 
        speech_recognizer: SpeechRecognizerComponent,
        speech_synthesizer: SpeechSynthesizerComponent,
        image_generator: ImageGeneratorComponent,
        data_logger: DataLogger,
        llm_prompt_name: str,
        mode: str = "train",  # "train" or "test"
        num_images_per_item: int = 10,
        sampling_rate: int = 22050,
        enable_audio_response: bool = True,
    ):
        self.llm = llm
        self.speech_recognizer = speech_recognizer
        self.speech_synthesizer = speech_synthesizer
        self.image_generator = image_generator
        self.data_logger = data_logger
        self.mode = mode
        self.num_images_per_item = num_images_per_item
        self.sampling_rate = sampling_rate
        self.enable_audio_response = enable_audio_response

        # Load config file
        self.load_config(config_path)
        self.data_logger.item_dict = self._load_item_dict_from_config()
        self.prompt_template = self.load_llm_prompt(llm_prompt_name)
        self.current_scene = self._initialize_current_scene()
        self.current_offered_items = []

    def load_config(self, config_path: Path):
        # Read configuration and initialize scenes and items
        config_path = Path(config_path)
        with open(config_path, "r", encoding="utf-8") as file:
            self.config = yaml.safe_load(file)
        self.scenes = self.config.get("scenes", [])
        self.items = self.config.get("items", [])

    def _load_item_dict_from_config(self) -> Dict[int, Item]:
        item_dict = {}
        for item_data in self.items:
            attributes = item_data.get("attributes", {})
            prompt = item_data.get("prompt", None)
            converted_attributes = {}
            for key, value in attributes.items():
                if isinstance(value, (int, float)):
                    converted_attributes[key] = np.float32(value)
                elif isinstance(value, str):
                    # Remove characters other than the first '.' and digits to judge numeric strings
                    numeric_str = value.replace(".", "", 1)  # remove only the first decimal dot
                    if numeric_str.isdigit():  # treat as numeric if possible
                        converted_attributes[key] = np.float32(float(value))
                    else:
                        converted_attributes[key] = value
                else:
                    converted_attributes[key] = value
            item = Item(
                id=item_data["id"],
                name=item_data["name"],
                image=None,
                prompt=prompt,
                attributes=converted_attributes,
            )
            item_dict[item.id] = item
        return item_dict

    def _initialize_current_scene(self) -> Dict[str, Any]:
        # Set the initial scene from the config
        initial_scene_id = self.config.get("initial_scene_id", 0)
        scene = next((s for s in self.scenes if s["id"] == initial_scene_id), None)
        if not scene:
            logging.error(f"Initial scene ID {initial_scene_id} not found.")
            raise ValueError(f"Initial scene ID {initial_scene_id} not found.")
        return scene

    def get_current_scene_info(self) -> DialogueScene:
        # Get current scene information and build DialogueScene
        self.current_offered_items = self._select_items_for_prompt()

        system_prompt = self.generate_system_prompt()
        # No agent utterance when fetching scene info
        response_text = self.llm.generate_response(system_prompt=system_prompt, user_input="")

        # Parse as JSON
        json_str = self._extract_json(response_text)
        if json_str:
            try:
                response_json = json.loads(json_str)
            except json.JSONDecodeError as e:
                logging.error(f"JSON parse error: {e}")
                response_json = {}
        else:
            logging.error("No valid JSON found in LLM response.")
            response_json = {}

        if self.enable_audio_response:
            # Do not obtain item_name / next_scene_id here
            response_text_final = response_json.get("response_text", "How may I help you?")
            speakers_description = self.current_scene.get("speaker_description", "")
            response_waveform = self.speech_synthesizer.text_to_waveform(
                text=response_text_final, speaker_description=speakers_description
            )
        else:
            response_waveform = np.zeros(
                (self.sampling_rate,), dtype=np.float32
            )  # Generate 1 second of silence as a waveform

        images, paths = [], []
        for item in self.current_offered_items:
            img, img_path = self._get_item_image(item)
            images.append(img)
            paths.append(img_path)

        # Register into FIFO queue
        self.data_logger.register_scene_paths(paths, self.mode)

        logging.debug("Scene information generated. Offering items and creating DialogueScene.")

        dialogue_scene = DialogueScene(
            _scene_id=self.current_scene["id"],
            _prompt_waveform=response_waveform,
            _images=images,  # List[np.ndarray]
        )
        return dialogue_scene

    def get_dialogue_feedback(self, agent_speech_waveform: np.ndarray) -> DialogueFeedback:
        # Create DialogueFeedback from the agent's speech
        if not isinstance(agent_speech_waveform, np.ndarray):
            logging.error("agent_speech_waveform must be np.ndarray.")
            raise ValueError("agent_speech_waveform must be np.ndarray.")

        user_input = self.speech_recognizer.speech_to_text(agent_speech_waveform)
        logging.info("Received user input.")

        system_prompt = self.generate_system_prompt()
        response_text = self.llm.generate_response(system_prompt=system_prompt, user_input=user_input)

        # Parse as JSON
        json_str = self._extract_json(response_text)
        if json_str:
            try:
                response_json = json.loads(json_str)
            except json.JSONDecodeError as e:
                logging.error(f"JSON parse error: {e}")
                response_json = {}
        else:
            logging.error("No valid JSON found in LLM response.")
            response_json = {}

        # Obtain reply text, next scene, and item name
        next_scene_id = response_json.get("next_scene_id", None)
        item_name = response_json.get("item_name", None)

        if self.enable_audio_response:
            response_text_final = response_json.get("response_text", "How may I help you?")
            speakers_description = self.current_scene.get("speaker_description", "")
            response_waveform = self.speech_synthesizer.text_to_waveform(
                text=response_text_final, speaker_description=speakers_description
            )
        else:
            response_waveform = np.zeros(
                (self.sampling_rate,), dtype=np.float32
            )  # Generate 1 second of silence as a waveform

        self._handle_scene_transition(next_scene_id)

        # Match and select an item name
        feedback_item = None
        if item_name:
            # Case-insensitive match for item names
            matched_item = next(
                (item for item in self.current_offered_items if item.name.replace("_", " ").lower() == item_name.replace("_", " ").lower()), 
                None
            )
            if matched_item:
                user_selected_item = matched_item
                feedback_item = FeedbackItem(
                    _id=user_selected_item.id,
                    _image=user_selected_item.image,
                    _attributes=user_selected_item.attributes,
                )
            else:
                logging.warning(f"Item '{item_name}' not available.")

        logging.debug("Dialogue feedback generated based on user input and LLM response.")

        dialogue_feedback = DialogueFeedback(_selected_item=feedback_item, _response_waveform=response_waveform)
        return dialogue_feedback

    def load_llm_prompt(self, llm_prompt_name):
        current_dir = Path(__file__).parent
    
        # Build the path to the template file under the prompts directory
        template_path = current_dir / "prompts" / llm_prompt_name

        if not template_path.exists():
            raise FileNotFoundError(f"Prompt template not found: {template_path}")

        return template_path.read_text(encoding="utf-8")

    def generate_system_prompt(self) -> str:
        # Create system prompt from the scene information
        scene_information = "\n".join(f"- Scene ID: {s['id']}, Scene Name: {s['name']}" for s in self.scenes)

        # Join currently offered item names as "- name"
        available_items_text = "\n".join(f"- {item.name}" for item in self.current_offered_items)

        # Join available transitions as "- Available scenes: Name (ID: X)."
        scene_dict = {s["id"]: s for s in self.scenes}
        transitions = "\n".join(
            f"- Available scenes: {scene_dict[ns]['name']} (ID: {ns})."
            for ns in self.current_scene.get("possible_next_scenes", [])
            if ns in scene_dict
        )

        # Extract system guidelines and role description
        guidelines = self.current_scene.get("system_guidelines", "")
        role_description = self.current_scene.get("role_description", "a clerk")

        # Fill the template
        return self.prompt_template.format(
            role_description=role_description,
            scene_information=scene_information,
            available_items_text=available_items_text,
            transitions=transitions,
            guidelines=guidelines,
        )

    def _extract_json(self, text: str) -> Optional[str]:
        # Extract the portion surrounded by {}
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return text[start : end + 1]
        else:
            logging.error("No valid JSON segment in LLM response.")
            return None

    def _handle_scene_transition(self, next_scene_id: Optional[int]):
        # Perform scene transition
        if next_scene_id is not None:
            if self._is_scene_transition_valid(next_scene_id):
                new_scene = next((s for s in self.scenes if s["id"] == next_scene_id), None)
                if new_scene:
                    self.current_scene = new_scene
                    logging.info(f"Scene transitioned to ID {self.current_scene['id']}.")
                else:
                    logging.warning(f"Scene ID {next_scene_id} not found; no transition made.")
            else:
                logging.warning(f"Invalid scene transition request to ID {next_scene_id}.")

    def _is_scene_transition_valid(self, scene_id: int) -> bool:
        # Validate whether the next scene predicted by the LLM is valid
        return scene_id in self.current_scene.get("possible_next_scenes", [])

    def _select_items_for_prompt(self) -> List[Item]:
        # Randomly select items to present
        available_item_ids = self.current_scene.get("items", [])
        if len(available_item_ids) < 2:
            selected_item_ids = available_item_ids
        else:
            selected_item_ids = random.sample(available_item_ids, 2)
        selected_items = [self.data_logger.item_dict[item_id] for item_id in selected_item_ids]
        return selected_items

    def _get_item_image(self, item: Item) -> tuple[np.ndarray, Optional[Path]]:
        """
        Retrieve or generate an image corresponding to the Item instance and save it to a file.
        If an image is already registered, return it. Otherwise, generate, register, and save it.
        """
        # Do not save images if num_images_per_item is less than 1
        if self.num_images_per_item < 1:
            # Generate an image as np.ndarray
            image = self.image_generator.generate_image(
                item_name=item.name, item_prompt=item.prompt
            )  # np.ndarray (H, W, C)
            if not isinstance(image, np.ndarray):
                logging.error(f"Generated image for item '{item.name}' is not a numpy.ndarray.")
                raise ValueError(f"Generated image for item '{item.name}' is not a numpy.ndarray.")
            # Register the image to the Item instance
            item.image = image
            return image, None

        image_index = random.randint(0, self.num_images_per_item - 1)
        image_path = self.data_logger.images_dir / self.mode / f"{item.id}_{item.name}_{image_index}.png"
        if image_path.exists():
            image = self.data_logger.load_image(item, image_index=image_index, mode=self.mode)
            item.image = image
            return image, image_path
        else:
            # Generate an image as np.ndarray
            image = self.image_generator.generate_image(
                item_name=item.name, item_prompt=item.prompt
            )  # np.ndarray (H, W, C)
            if not isinstance(image, np.ndarray):
                logging.error(f"Generated image for item '{item.name}' is not a numpy.ndarray.")
                raise ValueError(f"Generated image for item '{item.name}' is not a numpy.ndarray.")
            # Save the generated image
            self.data_logger.save_image(image, item, image_index=image_index, mode=self.mode)

            # Register the image to the Item instance
            item.image = image

            return image, image_path
