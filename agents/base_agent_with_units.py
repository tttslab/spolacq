from abc import ABC, abstractmethod
from pathlib import Path
from agents.base_agent import BaseAgent


class BaseImage2Unit(ABC):
    def get_unit(self, path: Path) -> list[str]:
        # Implement the process to get units from an image
        pass


class BaseSpeech2Unit(ABC):
    @abstractmethod
    def get_unit(self, paths: Path) -> list[str]:
        pass


class BaseUnit2Speech(ABC):
    def get_speech(self, unit: list[str]) -> str:
        # Implement the process to get audio from the units
        pass


class BaseRLUnit(ABC):
    def infer(self, unit: list[str]) -> str:
        # Implement the process to get RL action from the units
        pass


class BaseAgentWithUnits(BaseAgent, ABC):
    def __init__(self, agent_config):
        super().__init__(agent_config)

    @property
    @abstractmethod
    def i2u(self) -> BaseImage2Unit:
        """Abstract property returning an instance of Image2Unit"""
        pass

    @property
    @abstractmethod
    def s2u(self) -> BaseSpeech2Unit:
        """Abstract property returning an instance of Speech2Unit"""
        pass

    @property
    @abstractmethod
    def u2s(self) -> BaseUnit2Speech:
        """Abstract property returning an instance of Unit2Speech"""
        pass

    def pretrain(self):
        self.train_i2u()
        self.train_u2s()
        self.train_s2u()

    # When training an I2U model, override this function.
    def train_i2u(self):
        pass

    # When training a U2S model, override this function.
    def train_u2s(self):
        pass

    # When training a S2U model, override this function.
    def train_s2u(self):
        pass
