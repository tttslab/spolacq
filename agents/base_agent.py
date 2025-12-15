from abc import ABC, abstractmethod
from typing import Dict, Any

import numpy as np


class BaseAgent(ABC):
    agent_config: Dict[str, Any]
    def __init__(self, agent_config: Dict[str, Any]):
        super().__init__()

        self.agent_config = agent_config

    # Function to convert the action space of reinforcement learning into speech data
    @abstractmethod
    def action2speech(self, action: np.ndarray) -> np.ndarray:

        raise NotImplementedError()
