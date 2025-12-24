import numpy as np
from pathlib import Path
from typing import Optional, Any
from dataclasses import dataclass


@dataclass
class Item:
    id: int
    name: str
    image: Optional[np.ndarray]
    prompt: Optional[str]
    attributes: dict[str, Any]


# Item class excluding name (used for feedback)
@dataclass
class FeedbackItem:
    _id: int
    _image: Optional[np.ndarray]
    _attributes: dict[str, Any]

    @property
    def id(self) -> int:
        return self._id

    @property
    def image(self) -> Optional[np.ndarray]:
        return self._image

    @property
    def attributes(self) -> dict[str, Any]:
        return self._attributes


@dataclass
class DialogueScene:
    _scene_id: int
    _prompt_waveform: np.ndarray
    _images: list[np.ndarray]

    @property
    def scene_id(self) -> int:
        return self._scene_id

    @property
    def prompt_waveform(self) -> np.ndarray:
        return self._prompt_waveform

    @property
    def images(self) -> list[np.ndarray]:
        return self._images


@dataclass
class DialogueFeedback:
    _selected_item: Optional[FeedbackItem]
    _response_waveform: np.ndarray

    @property
    def selected_item(self) -> Optional[FeedbackItem]:
        return self._selected_item

    @property
    def response_waveform(self) -> np.ndarray:
        return self._response_waveform


@dataclass
class EnvironmentState:
    _dialogue_scene: DialogueScene
    _dialogue_feedback: Optional[DialogueFeedback]
    _internal_state: np.ndarray

    @property
    def dialogue_scene(self) -> DialogueScene:
        return self._dialogue_scene

    @property
    def dialogue_feedback(self) -> Optional[DialogueFeedback]:
        return self._dialogue_feedback

    @property
    def internal_state(self) -> np.ndarray:
        return self._internal_state


@dataclass
class SavedInteraction:
    environment_state_dict: dict[str, Any]
    action_waveform_path: Path
    next_environment_state_dict: dict[str, Any]
    reward: float
