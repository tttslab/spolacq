"""
The MIT License

Copyright (c) 2019 Antonin Raffin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import Schedule, GymEnv
from stable_baselines3.dqn.dqn import DQN
from stable_baselines3.dqn.policies import QNetwork, DQNPolicy

from model import SimpleImageCorrNet

class CustomFeaturesExtractor(BaseFeaturesExtractor):
    """
    Custom features extractors from an observation.
    This consists of two features extractors (visual_net and state_net).
    The "sound" can be a question audio from an environment.
    
    :param observation_space: Observation space of an agent.
    :param cluster_centers: Cluster centers of training images.
    :param load_state_dict_path: Path of the model pretrained by correspondence learning.
    :param action_mask: If specified, only those actions can be selected.
    :param num_per_group: Number of audio segments to be focused on per each cluster center.
    :param purify_rate: This is a factor that is multiplied when the action filter is updated.
    """
    
    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        cluster_centers: np.ndarray,
        load_state_dict_path: Optional[str] = None,
        action_mask: Optional[List[int]] = None,
        num_per_group: int = 100,
        purify_rate: float = 0.97,
    ):
        if "sound" in observation_space.spaces:
            super().__init__(observation_space, cluster_centers.shape[1]*4)
        else:
            super().__init__(observation_space, cluster_centers.shape[1]*3)
        
        self.corr_net = SimpleImageCorrNet(num_class=cluster_centers.shape[1])
        self.focus_net = SimpleImageCorrNet(num_class=cluster_centers.shape[1])
        self.state_net = nn.Linear(observation_space.spaces["state"].shape[0], cluster_centers.shape[1])
        if load_state_dict_path:
            self.corr_net.load_state_dict(torch.load(load_state_dict_path)["model_state_dict"])
            self.focus_net.load_state_dict(torch.load(load_state_dict_path)["model_state_dict"])
        
        # For focusing mechanism
        self.num_per_group = num_per_group
        self.purify_rate = purify_rate
        self.cluster_centers = nn.parameter.Parameter(
            torch.from_numpy(cluster_centers), requires_grad=False)
        self.action_filter = nn.parameter.Parameter(
            torch.ones((cluster_centers.shape[0], self.num_per_group)), requires_grad=False)
        
        if action_mask: self.set_action_mask(action_mask)
    
    def set_action_mask(self, action_mask: List[int]) -> None:
        self.action_mask = nn.parameter.Parameter(
            torch.full(
                (self.cluster_centers.size()[0]*self.num_per_group,),
                -float("inf"),
            ).to(self.cluster_centers.device),
            requires_grad=False,
        )
        self.action_mask[action_mask] = 0
    
    def forward(self, observation: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state = torch.tanh(self.state_net(observation["state"]))
        leftimage = torch.tanh(self.corr_net._visual_extract(observation["leftimage"]))
        rightimage = torch.tanh(self.corr_net._visual_extract(observation["rightimage"]))
        if "sound" in observation:
            sound = torch.tanh(self.corr_net._sound_extract(observation["sound"]))
        else:
            sound = torch.empty((state.size()[0], 0)).to(state.device)
        leftimage_similarity = self.focus(observation["leftimage"])
        rightimage_similarity = self.focus(observation["rightimage"])
        return torch.cat((state, leftimage, rightimage, sound), dim=1), leftimage_similarity, rightimage_similarity
    
    def focus(self, image: torch.Tensor) -> torch.Tensor:
        self.focus_net.eval()
        with torch.no_grad():
            feature = self.focus_net._visual_extract(image)
            similarity = -torch.cdist(feature, self.cluster_centers)
            similarity = F.softmax(similarity, dim=1)
            similarity = similarity / torch.max(similarity, dim=1, keepdim=True)[0] # Broadcast
            similarity = similarity.repeat_interleave(self.num_per_group, dim=1)
            similarity = similarity * torch.flatten(self.action_filter) # Broadcast
            similarity = similarity / torch.max(similarity, dim=1, keepdim=True)[0] # Broadcast
        return similarity


class FocusingActionFilterQNetwork(nn.Module):
    """
    Focusing action filter Q-network.
    This does not include features extractors.
    """
    
    def __init__(self, features_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(features_dim, features_dim//2)
        self.fc2 = nn.Linear(features_dim//2, 2)
    
    def forward(self, x: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        features, leftimage_similarity, rightimage_similarity = x
        features = F.relu(self.fc1(features))
        alpha = F.softmax(self.fc2(features), dim=1)
        return alpha[:,0][:,None]*leftimage_similarity + alpha[:,1][:,None]*rightimage_similarity


class CustomQNetwork(QNetwork):
    """
    This stacks focusing action filter Q-network
    on the received features extractor.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.q_net = FocusingActionFilterQNetwork(self.features_dim)
    
    def _predict(self, observation: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        q_values = self.forward(observation)
        if hasattr(self.features_extractor, "action_mask"):
            q_values += self.features_extractor.action_mask # Broadcast
        # Greedy action
        action = q_values.argmax(dim=1).reshape(-1)
        return action


class CustomDQNPolicy(DQNPolicy):
    """Q-network and target Q-network."""
    
    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        action_space: gym.spaces.Discrete,
        lr_schedule: Schedule,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor_class: Type[BaseFeaturesExtractor] = CustomFeaturesExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )
    
    def make_q_net(self) -> CustomQNetwork:
        # instantiate features_extractor here
        # instantiated features_extractor is included in net_args as kwarg
        net_args = self._update_features_extractor(self.net_args, features_extractor=None)
        return CustomQNetwork(**net_args).to(self.device)


class CustomDQN(DQN):
    """
    Custom DQN off-policy algorithm class.
    We added code to update an action filter.
    policy_kwargs is passed to CustomDQNPolicy.
    features_extractor_kwargs is passed to CustomFeaturesExtractor.
    
    Example of retraining RL model
    If reset_num_timesteps=False, we can continue the previous tensorboard's log.
    
    del model
    del env
    env = SpoLacq(FOODS, args.datadir, sounddic_text, lambda x: x)
    model = CustomDQN.load(args.workdir+"/dqn", env=env)
    model.load_replay_buffer(args.workdir+"/replay_buffer")
    model.learn(total_timesteps=args.total_timesteps, tb_log_name=tb_log_name, reset_num_timesteps=False)
    """
    
    def __init__(self, policy: CustomDQNPolicy, env: Union[GymEnv, str], **kwargs):
        super().__init__(policy, env, **kwargs)
    
    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for _ in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            with torch.no_grad():
                # Compute the next Q-values using the target network
                next_q_values = self.q_net_target(replay_data.next_observations)
                # Follow greedy policy: use the one with the highest value
                next_q_values, _ = next_q_values.max(dim=1)
                # Avoid potential broadcast issue
                next_q_values = next_q_values.reshape(-1, 1)
                # 1-step TD target
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # Get current Q-values estimates
            current_q_values = self.q_net(replay_data.observations)

            # Retrieve the q-values for the actions from the replay buffer
            current_q_values = torch.gather(current_q_values, dim=1, index=replay_data.actions.long())

            # Compute Huber loss (less sensitive to outliers)
            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(loss.item())

            # Optimize the policy
            self.policy.optimizer.zero_grad()
            loss.backward()
            # Clip gradient norm
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()
            
            # Update action filter
            for action, reward in zip(replay_data.actions, replay_data.rewards):
                i, j = divmod(action.item(), self.policy.q_net.features_extractor.num_per_group)
                if reward.item() > 0:
                    self.policy.q_net.features_extractor.action_filter[i,:] *= self.policy.q_net.features_extractor.purify_rate
                    self.policy.q_net.features_extractor.action_filter[i,j] = 1
                else:
                    self.policy.q_net.features_extractor.action_filter[i,j] *= self.policy.q_net.features_extractor.purify_rate

        # Increase update counter
        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))
    
    def set_action_mask(self, action_mask: List[int]) -> None:
        self.policy.q_net.features_extractor.set_action_mask(action_mask)
        self.policy.q_net_target.features_extractor.set_action_mask(action_mask)