import logging
from typing import Tuple, Optional, Type, List, Any, Callable
import numpy as np
import gymnasium as gym

from .dialogue_simulator import DialogueSimulator
from .data_models import DialogueFeedback, DialogueScene, EnvironmentState


class AgentStateManager(gym.Env):
    dialogue_simulator: DialogueSimulator
    initial_internal_state_function: Callable[[], np.ndarray]
    observation_space: gym.Space
    dialogue_scene: DialogueScene
    dialogue_feedback: Optional[DialogueFeedback]
    current_internal_state: np.ndarray
    current_env: EnvironmentState
    previous_env: EnvironmentState
    max_step: int
    talk_step: int

    def __init__(
        self,
        dialogue_simulator: DialogueSimulator,
        initial_internal_state_function: Optional[Callable[[], np.ndarray]] = None,
        max_step: int = 100,
    ):
        super().__init__()
        self.dialogue_simulator = dialogue_simulator
        self.initial_internal_state_function = (
            initial_internal_state_function
            if initial_internal_state_function is not None
            else lambda: np.random.rand(3).astype(np.float32)
        )
        self.max_step = max_step
        self.action_space = VariableLengthVectorSpace(dtype=np.float32, min_length=1)

        # Observation space includes only the internal state
        self.observation_space = gym.spaces.Dict(
            {
                "internal_state": gym.spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32),
            }
        )

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[dict, dict]:
        logging.info("Resetting environment to initial state.")
        self.dialogue_simulator.current_scene = self.dialogue_simulator._initialize_current_scene()
        self.dialogue_scene = self.dialogue_simulator.get_current_scene_info()
        self.dialogue_feedback = None
        self.current_internal_state = self.initial_internal_state_function()  # Initialize internal state on reset
        self.current_env = EnvironmentState(self.dialogue_scene, self.dialogue_feedback, self.current_internal_state)
        self.talk_step = 0
        obs = {"internal_state": self.current_env.internal_state}
        return obs, {}

    def step(self, action_waveform: np.ndarray) -> Tuple[dict, float, bool, bool, dict]:
        logging.info("Processing action and updating environment.")
        self.previous_env = self.current_env

        # Obtain DialogueFeedback
        self.dialogue_feedback = self.dialogue_simulator.get_dialogue_feedback(action_waveform)

        # Get the next DialogueScene
        self.dialogue_scene = self.dialogue_simulator.get_current_scene_info()

        # Construct the next environment state
        self.current_env = EnvironmentState(self.dialogue_scene, self.dialogue_feedback, self.current_internal_state)

        # Initial reward/termination (expected to be overwritten by a Wrapper)
        reward = 0.0
        terminated = False

        self.talk_step += 1
        truncated = self.talk_step >= self.max_step

        # By default, only the internal state is observed
        # (ObservationProcessingWrapper can set this from self.current_env)
        obs = {"internal_state": self.current_env.internal_state}

        return obs, reward, terminated, truncated, {}


class VariableLengthVectorSpace(gym.Space):
    dtype: Type[Any]
    min_length: int
    max_length: Optional[int]

    def __init__(self, dtype=np.float32, min_length: int = 1, max_length: Optional[int] = None):
        """
        Custom Space class representing a variable-length vector.

        Args:
            dtype (type, optional): Data type. Defaults to np.float32.
            min_length (int, optional): Minimum vector length. Defaults to 1.
            max_length (Optional[int], optional): Maximum vector length.
                If None, there is no upper bound.
        """
        # Gymnasium's Space requires shape and dtype.
        # Since the length is variable, set shape to None or an empty tuple.
        super().__init__(shape=None, dtype=dtype)
        self.dtype = dtype
        self.min_length = min_length
        self.max_length = max_length  # If None, no upper bound

    @property
    def shape(self) -> Optional[Tuple[int, ...]]:
        """
        Returns None because the length is variable.
        If needed, shape information based on the minimum length could be provided.
        """
        return None  # No fixed shape for variable-length vectors

    def sample(self) -> np.ndarray:
        """
        Generate a random sample.
        """
        if self.max_length is not None:
            length = np.random.randint(self.min_length, self.max_length + 1)
        else:
            length = np.random.randint(self.min_length, 100)
        return self.dtype(np.random.randn(length))

    def contains(self, x) -> bool:
        """
        Check whether a sample belongs to this space.

        Args:
            x: Sample to check.

        Returns:
            bool: True if contained, False otherwise.
        """
        if not isinstance(x, (list, np.ndarray)):
            return False
        length = len(x)
        if length < self.min_length:
            return False
        if self.max_length is not None and length > self.max_length:
            return False
        if not np.issubdtype(np.array(x).dtype, np.dtype(self.dtype).type):
            return False
        return True

    def to_jsonable(self, sample_n: List[np.ndarray]) -> List[List[float]]:
        """
        Convert samples to a JSON-serializable form.

        Args:
            sample_n (List): List of samples.

        Returns:
            List[List[float]]: JSON-compatible list.
        """
        return [x.tolist() for x in sample_n]

    def from_jsonable(self, sample_n: List[List[float]]) -> List[np.ndarray]:
        """
        Convert from JSON-serializable form to samples.

        Args:
            sample_n (List[List[float]]): List of JSON-serializable samples.

        Returns:
            List[np.ndarray]: List of NumPy arrays.
        """
        return [self.dtype(x) for x in sample_n]

    def __repr__(self):
        return (
            f"VariableLengthVectorSpace(dtype={self.dtype}, min_length={self.min_length}, max_length={self.max_length})"
        )
