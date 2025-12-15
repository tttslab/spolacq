from typing import Dict, Any, List, TypedDict
from pathlib import Path

import gymnasium as gym

from agents.base_agent import BaseAgent
from agents.base_agent_with_units import BaseAgentWithUnits
from abc import ABC, abstractmethod

class I2UInfo(TypedDict):
    word_map_path: Path
    output_model_path: Path


class U2MInfo(TypedDict):
    u2m_model_dir: Path
    u2m_pretrain_config: dict
    data_info: dict


class M2SInfo(TypedDict):
    m2s_pretrain_config: dict


class U2SInfo(TypedDict):
    u2m: U2MInfo
    m2s: M2SInfo


class PretrainInfo(TypedDict):
    i2u: I2UInfo
    u2s: U2SInfo

class BaseTrainer(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def pretrain(
        self, 
        agent: BaseAgent, 
        image_paths: Dict[str, List[Path]], 
        audio_paths: Dict[str, List[Path]], 
        base_config: Dict[str, Any], 
        pretrain_config: Dict[str, Any], 
        agent_dir: Path, 
        agent_experiment_dir: Path
    ) -> PretrainInfo:
        raise NotImplementedError()
    
    @abstractmethod
    def train_rl(
        self, 
        agent: BaseAgent, 
        env_train: gym.Env, 
        env_test: gym.Env, 
        base_config: Dict[str, Any], 
        env_config: Dict[str, Any], 
        agent_config: Dict[str, Any], 
        pretrain_info: PretrainInfo, 
        agent_dir: Path, 
        agent_experiment_dir: Path
    ) -> None:
        raise NotImplementedError()

class BaseTrainerWithUnits(BaseTrainer):
    def __init__(self):
        super().__init__()

    def pretrain(
        self, 
        agent: BaseAgentWithUnits, 
        image_paths: Dict[str, List[Path]], 
        audio_paths: Dict[str, List[Path]], 
        base_config: Dict[str, Any], 
        pretrain_config: Dict[str, Any], 
        agent_dir: Path, 
        agent_experiment_dir: Path
    ) -> PretrainInfo:
        i2u_info = self.train_i2u(
            agent, image_paths, audio_paths,
            base_config, pretrain_config,
            agent_dir, agent_experiment_dir,
        )
        u2s_info = self.train_u2s(
            agent, image_paths, audio_paths,
            base_config, pretrain_config,
            agent_dir, agent_experiment_dir,
        )
        return {"i2u": i2u_info, "u2s": u2s_info}
    
    @abstractmethod
    def train_i2u(
        self,
        agent: BaseAgentWithUnits,
        image_paths: Dict[str, List[Path]],
        audio_paths: Dict[str, List[Path]],
        base_config: Dict[str, Any],
        pretrain_config: Dict[str, Any],
        agent_dir: Path,
        agent_experiment_dir: Path,
        ) -> I2UInfo:
        raise NotImplementedError()

    @abstractmethod
    def train_u2s(
        self,
        agent: BaseAgentWithUnits,
        image_paths: Dict[str, List[Path]],
        audio_paths: Dict[str, List[Path]],
        base_config: Dict[str, Any],
        pretrain_config: Dict[str, Any],
        agent_dir: Path,
        agent_experiment_dir: Path,
        ) -> U2SInfo:

        raise NotImplementedError()  
