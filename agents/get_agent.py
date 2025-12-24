import os
import sys
import json
from pathlib import Path
import importlib


from agents.base_agent import BaseAgent
from agents.base_trainer import BaseTrainer

def get_agent_path(agent_name: str) -> dict[str, Path]:
    agent_dir = Path(__file__).parent / agent_name
    agent_config_file = agent_dir / "agent_config.json"
    return {
        "agent_dir": agent_dir,
        "agent_config_file": agent_config_file
    }


def get_agent(agent_name: str, agent_config) -> tuple[BaseAgent, BaseTrainer]:
    agent_path = get_agent_path(agent_name)
    agent_dir = agent_path["agent_dir"]
    agent_config_file = agent_path["agent_config_file"]

    # Check if the agent directory exists
    if not agent_dir.is_dir():
        raise ValueError(f"Agent '{agent_name}' does not exist")

    try:
        with open(agent_config_file, "r") as f:
            agent_class_config = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"configuration file '{agent_config_file}' was not found")
    except json.JSONDecodeError:
        raise ValueError(f"JSON format of the configuration file '{agent_config_file}' is invalid")

    # Check if the required key exists
    required_keys = ["agent_class", "trainer_class"]
    for key in required_keys:
        if key not in agent_class_config:
            raise ValueError(f"The setting '{key}' was not found for agent '{agent_name}'")

    # Import the module
    agent_module = importlib.import_module(f".{agent_name}.agent", package="agents")
    trainer_module = importlib.import_module(f".{agent_name}.trainer", package="agents")

    # get the class
    agent_class = getattr(agent_module, agent_class_config["agent_class"])
    trainer_class = getattr(trainer_module, agent_class_config["trainer_class"])
    # Type checking
    if not issubclass(agent_class, BaseAgent):
        raise TypeError(f"'{agent_class_config['agent_class']}' does not inherit from BaseAgent")

    if not issubclass(trainer_class, BaseTrainer):
        raise TypeError(f"'{agent_class_config['trainer_class']}' does not inherit from BaseTrainer")

    agent = agent_class(agent_config)
    trainer = trainer_class()
    return agent, trainer
