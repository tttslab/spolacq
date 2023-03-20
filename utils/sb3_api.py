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

import multiprocessing
import sys
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from stable_baselines3 import TD3
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BaseModel, BasePolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, NatureCNN, get_actor_critic_arch
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.dqn.dqn import DQN
from stable_baselines3.dqn.policies import QNetwork, DQNPolicy
from stable_baselines3.td3.policies import TD3Policy

from model import SimpleImageCorrNet
from loader import A_Dataset

sys.path.append("../tools/dino")
from vision_transformer import VisionTransformer


class RefinedFeaturesExtractor(BaseFeaturesExtractor):
    """
    Custom features extractors from an observation.
    This consists of three features extractors (state_net, corr_net, focus_net).
    The "sound" can be a question audio from an environment.
    
    :param observation_space: Observation space of an agent.
    :param segments_path: Path of a text file describing paths of spectrograms.
    :param load_state_dict_path: Path of the model pretrained by correspondence learning.
    :param action_mask: If specified, only those actions can be selected.
    :param use_focusing: Whether to use refined focusing mechanism or not.
    """
    
    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        segments_path: str,
        load_state_dict_path: str,
        action_mask: Optional[List[int]] = None,
        use_focusing: bool = True,
    ):
        num_class = torch.load(load_state_dict_path)["model_state_dict"]["audio_net.fc.bias"].size(dim=0)
        super().__init__(
            observation_space,
            num_class*4 if "sound" in observation_space.spaces else num_class*3,
        )
        
        # Front-ends
        self.state_net = nn.Linear(observation_space.spaces["state"].shape[0], num_class)
        self.corr_net = SimpleImageCorrNet(num_class=num_class)
        self.corr_net.load_state_dict(torch.load(load_state_dict_path)["model_state_dict"])
        self.focus_net = SimpleImageCorrNet(num_class=num_class)
        self.focus_net.load_state_dict(torch.load(load_state_dict_path)["model_state_dict"])
        
        with open(segments_path) as f:
            # paths of spectrogram of audio segment (*.pkl)
            self.spec_paths = [p.strip() for p in f]

        self.use_focusing = use_focusing
        self.initialize_key()
        if action_mask: self.set_action_mask(action_mask)
    
    def initialize_key(self, num_workers: int = multiprocessing.cpu_count(), device=torch.device("cpu")) -> None:
        loader = DataLoader(
            A_Dataset(self.spec_paths, None, None),
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        self.focus_net.eval()
        with torch.no_grad():
            key = [self.focus_net._sound_extract(segment.to(device)) for segment in loader]
        self.key = nn.parameter.Parameter(torch.cat(key, dim=0), requires_grad=False)
    
    def set_action_mask(self, action_mask: List[int]) -> None:
        self.action_mask = nn.parameter.Parameter(
            torch.full(
                (self.key.size(dim=0),),
                -float("inf"),
                device=self.key.device,
            ),
            requires_grad=False,
        )
        self.action_mask[action_mask] = 0
    
    def forward(self, observation: Dict[str, torch.Tensor]) -> tuple:
        state = torch.tanh(self.state_net(observation["state"]))
        leftimage = torch.tanh(self.corr_net._visual_extract(observation["leftimage"]))
        rightimage = torch.tanh(self.corr_net._visual_extract(observation["rightimage"]))
        if "sound" in observation:
            sound = torch.tanh(self.corr_net._sound_extract(observation["sound"]))
        else:
            sound = torch.empty((state.size(dim=0), 0), device=state.device)
        
        if self.use_focusing:
            leftimage_weight = self.focus(observation["leftimage"])
            rightimage_weight = self.focus(observation["rightimage"])
            return torch.cat((state, leftimage, rightimage, sound), dim=1), leftimage_weight, rightimage_weight
        else:
            return torch.cat((state, leftimage, rightimage, sound), dim=1),
    
    def focus(self, image: torch.Tensor) -> torch.Tensor:
        self.focus_net.eval()
        with torch.no_grad():
            query = self.focus_net._visual_extract(image)
            weight = -torch.cdist(query, self.key)
            weight = F.softmax(weight, dim=1)
            weight = weight / torch.max(weight, dim=1, keepdim=True)[0] # Broadcast
        return weight


class RefinedQNetwork(nn.Module):
    """
    Refined Q-network. This does not include features extractors.
    """
    
    def __init__(self, features_dim: int, action_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(features_dim, features_dim//2)
        self.fc2 = nn.Linear(features_dim//2, 3)
        self.fc3 = nn.Linear(features_dim, features_dim//2)
        self.fc4 = nn.Linear(features_dim//2, action_dim)
    
    def forward(self, x: tuple) -> torch.Tensor:
        if len(x) == 3:
            features, leftimage_weight, rightimage_weight = x
            alpha = F.relu(self.fc1(features))
            alpha = F.softmax(self.fc2(alpha), dim=1)
            weight = F.relu(self.fc3(features))
            weight = self.fc4(weight)
            return alpha[:,0][:,None]*leftimage_weight + alpha[:,1][:,None]*rightimage_weight + alpha[:,2][:,None]*weight
        else:
            features = x[0]
            features = F.relu(self.fc3(features))
            q_values = F.softmax(self.fc4(features), dim=1)
            return q_values


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
        if isinstance(self.features_extractor, RefinedFeaturesExtractor):
            self.q_net = RefinedQNetwork(self.features_dim, self.action_space.n)
        elif isinstance(self.features_extractor, CustomFeaturesExtractor):
            self.q_net = FocusingActionFilterQNetwork(self.features_dim)
        else:
            raise ValueError(f"You must use RefinedFeaturesExtractor or CustomFeaturesExtractor as features_extractor_class.")
    
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
        features_extractor_class: Union[Type[RefinedFeaturesExtractor], Type[CustomFeaturesExtractor]] = RefinedFeaturesExtractor,
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
    
    def __init__(self, policy: Type[CustomDQNPolicy], env: Union[GymEnv, str], **kwargs):
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
            if isinstance(self.policy.q_net.features_extractor, CustomFeaturesExtractor):
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


class CustomFeaturesExtractorUsingDINO(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict,
                 image_features_dim: int = 50, state_features_dim: int = 50):
        super().__init__(observation_space, 2*image_features_dim+state_features_dim)

        self.image_encoder = VisionTransformer(patch_size=8, qkv_bias=True)
        state_dict = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth",
            map_location="cpu",
        )
        self.image_encoder.load_state_dict(state_dict)

        self.state_encoder = nn.Linear(observation_space.spaces["state"].shape[0], state_features_dim)
        self.linear = nn.Linear(768, image_features_dim)
    
    def forward(self, observation: Dict[str, torch.Tensor]):
        state_features = self.state_encoder(observation["state"])

        with torch.no_grad():
            self.image_encoder.eval()
            leftimage_features = self.image_encoder(observation["leftimage"])
            rightimage_features = self.image_encoder(observation["rightimage"])
        
        features = torch.cat((state_features, self.linear(leftimage_features), self.linear(rightimage_features)), dim=1)
        features = F.relu(features)
        return features, leftimage_features, rightimage_features


class CustomActor(BasePolicy):
    """
    Actor network (policy) for TD3.
    :param observation_space: Obervation space
    :param action_space: Action space
    :param net_arch: Network architecture
    :param features_extractor: Network to extract features
        (a CNN when using images, a nn.Flatten() layer otherwise)
    :param features_dim: Number of features
    :param activation_fn: Activation function
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        net_arch: List[List[int]],
        features_extractor: nn.Module,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
            squash_output=False,
        )

        self.net_arch = net_arch
        self.features_dim = features_dim
        self.activation_fn = activation_fn

        # action_dim = get_action_dim(self.action_space)
        # actor_net = create_mlp(features_dim, action_dim, net_arch, activation_fn, squash_output=True)
        # Deterministic action
        # self.mu = nn.Sequential(*actor_net)
        self.mu = CustomMu(net_arch, activation_fn)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_arch,
                features_dim=self.features_dim,
                activation_fn=self.activation_fn,
                features_extractor=self.features_extractor,
            )
        )
        return data

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # assert deterministic, 'The TD3 actor only outputs deterministic actions'
        features = self.extract_features(obs)
        return self.mu(features)

    def _predict(self, observation: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        # Note: the deterministic deterministic parameter is ignored in the case of TD3.
        #   Predictions are always deterministic.
        return self.forward(observation)


class CustomMu(nn.Module):
    def __init__(self, net_arch: List[List[int]], activation_fn: Type[nn.Module]):
        super().__init__()
        
        assert len(net_arch) == 2 and net_arch[0][-1] == 2
        
        self.layers_alpha = nn.ModuleList()
        self.layers_embed = nn.ModuleList()

        n_layers_alpha = len(net_arch[0])
        n_layers_embed = len(net_arch[1])

        for n in range(n_layers_alpha - 1):
            if n != n_layers_alpha - 2:
                self.layers_alpha.append(nn.Linear(net_arch[0][n], net_arch[0][n + 1]))
                self.layers_alpha.append(activation_fn())
            else:
                self.layers_alpha.append(nn.Linear(net_arch[0][n], net_arch[0][n + 1], bias=False))
        
        if net_arch[1]:
            self.use_embed = True

            for n in range(n_layers_embed - 1):
                if n != n_layers_embed - 2:
                    self.layers_embed.append(nn.Linear(net_arch[1][n], net_arch[1][n + 1]))
                    self.layers_embed.append(activation_fn())
                else:
                    self.layers_embed.append(nn.Linear(net_arch[1][n], net_arch[1][n + 1], bias=False))
                    self.layers_embed.append(nn.LayerNorm(net_arch[1][n + 1]))
        else:
            self.use_embed = False
    
    def forward(self, x):
        features, leftimage_features, rightimage_features = x
        
        alpha = features
        embed = features

        for layer in self.layers_alpha:
            alpha = layer(alpha)

        if self.use_embed:
            for layer in self.layers_embed:
                embed = layer(embed)

        if self.training:
            alpha = F.softmax(alpha, dim=1)
            image_features = alpha[:, 0:1] * leftimage_features + alpha[:, 1:2] * rightimage_features
        else:
            image_features = torch.stack([leftimage_features, rightimage_features], dim=1)  # (batch_size, 2, features dim.)
            alpha = alpha.unsqueeze(-1)  # (batch_size, 2, 1)
            index = torch.argmax(alpha, dim=1, keepdim=True)  # (batch_size, 1, 1)
            index = index.expand(image_features.size(0), 1, image_features.size(2))  # (batch_size, 1, features dim.)
            image_features = torch.gather(image_features, dim=1, index=index)
            image_features = image_features.squeeze(1)
        
        if self.use_embed:
            action = torch.cat([image_features, embed], dim=1)
        else:
            action = image_features
        return action


class CustomContinuousCritic(BaseModel):
    """
    Critic network(s) for DDPG/SAC/TD3.
    It represents the action-state value function (Q-value function).
    Compared to A2C/PPO critics, this one represents the Q-value
    and takes the continuous action as input. It is concatenated with the state
    and then fed to the network which outputs a single value: Q(s, a).
    For more recent algorithms like SAC/TD3, multiple networks
    are created to give different estimates.
    By default, it creates two critic networks used to reduce overestimation
    thanks to clipped Q-learning (cf TD3 paper).
    :param observation_space: Obervation space
    :param action_space: Action space
    :param net_arch: Network architecture
    :param features_extractor: Network to extract features
        (a CNN when using images, a nn.Flatten() layer otherwise)
    :param features_dim: Number of features
    :param activation_fn: Activation function
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param n_critics: Number of critic networks to create.
    :param share_features_extractor: Whether the features extractor is shared or not
        between the actor and the critic (this saves computation time)
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        net_arch: List[int],
        features_extractor: nn.Module,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        n_critics: int = 2,
        share_features_extractor: bool = True,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )

        # action_dim = get_action_dim(self.action_space)

        self.share_features_extractor = share_features_extractor
        self.n_critics = n_critics
        self.q_networks = []
        for idx in range(n_critics):
            # q_net = create_mlp(features_dim + action_dim, 1, net_arch, activation_fn)
            # q_net = nn.Sequential(*q_net)
            q_net = QNet(net_arch, activation_fn)
            self.add_module(f"qf{idx}", q_net)
            self.q_networks.append(q_net)

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        # Learn the features extractor using the policy loss only
        # when the features_extractor is shared with the actor
        with torch.set_grad_enabled(not self.share_features_extractor):
            features = self.extract_features(obs)
        qvalue_input = torch.cat([features[0], actions], dim=1)
        return tuple(q_net(qvalue_input) for q_net in self.q_networks)

    def q1_forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Only predict the Q-value using the first network.
        This allows to reduce computation when all the estimates are not needed
        (e.g. when updating the policy in TD3).
        """
        with torch.no_grad():
            features = self.extract_features(obs)
        return self.q_networks[0](torch.cat([features[0], actions], dim=1))


class QNet(nn.Module):
    def __init__(self, net_arch: List[int], activation_fn: Type[nn.Module]):
        super().__init__()
        assert net_arch[-1] == 1

        self.layers = nn.ModuleList()

        n_layers = len(net_arch)

        for n in range(n_layers - 1):
            if n != n_layers - 2:
                self.layers.append(nn.Linear(net_arch[n], net_arch[n + 1]))
                self.layers.append(activation_fn())
            else:
                self.layers.append(nn.Linear(net_arch[n], net_arch[n + 1], bias=False))
    
    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
        return x


class CustomTD3Policy(BasePolicy):
    """
    Policy class (with both actor and critic) for TD3.
    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :param n_critics: Number of critic networks to create.
    :param share_features_extractor: Whether to share or not the features extractor
        between the actor and the critic (this saves computation time)
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor_class: Type[BaseFeaturesExtractor] = CustomFeaturesExtractorUsingDINO,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = True,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=False,
        )

        # Default network architecture, from the original paper
        if net_arch is None:
            if features_extractor_class == NatureCNN:
                net_arch = []
            else:
                net_arch = [400, 300]

        actor_arch, critic_arch = get_actor_critic_arch(net_arch)

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.net_args = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "net_arch": actor_arch,
            "activation_fn": self.activation_fn,
            "normalize_images": normalize_images,
        }
        self.actor_kwargs = self.net_args.copy()
        self.critic_kwargs = self.net_args.copy()
        self.critic_kwargs.update(
            {
                "n_critics": n_critics,
                "net_arch": critic_arch,
                "share_features_extractor": share_features_extractor,
            }
        )

        self.actor, self.actor_target = None, None
        self.critic, self.critic_target = None, None
        self.share_features_extractor = share_features_extractor

        self._build(lr_schedule)

    def _build(self, lr_schedule: Schedule) -> None:
        # Create actor and target
        # the features extractor should not be shared
        self.actor = self.make_actor(features_extractor=None)
        self.actor_target = self.make_actor(features_extractor=None)
        # Initialize the target to have the same weights as the actor
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.actor.optimizer = self.optimizer_class(self.actor.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

        if self.share_features_extractor:
            self.critic = self.make_critic(features_extractor=self.actor.features_extractor)
            # Critic target should not share the features extactor with critic
            # but it can share it with the actor target as actor and critic are sharing
            # the same features_extractor too
            # NOTE: as a result the effective poliak (soft-copy) coefficient for the features extractor
            # will be 2 * tau instead of tau (updated one time with the actor, a second time with the critic)
            self.critic_target = self.make_critic(features_extractor=self.actor_target.features_extractor)
        else:
            # Create new features extractor for each network
            self.critic = self.make_critic(features_extractor=None)
            self.critic_target = self.make_critic(features_extractor=None)

        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic.optimizer = self.optimizer_class(self.critic.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

        # Target networks should always be in eval mode
        self.actor_target.set_training_mode(False)
        self.critic_target.set_training_mode(False)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_arch,
                activation_fn=self.net_args["activation_fn"],
                n_critics=self.critic_kwargs["n_critics"],
                lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
                share_features_extractor=self.share_features_extractor,
            )
        )
        return data

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> CustomActor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return CustomActor(**actor_kwargs).to(self.device)

    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> CustomContinuousCritic:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        return CustomContinuousCritic(**critic_kwargs).to(self.device)

    def forward(self, observation: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        return self._predict(observation, deterministic=deterministic)

    def _predict(self, observation: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        # Note: the deterministic deterministic parameter is ignored in the case of TD3.
        #   Predictions are always deterministic.
        return self.actor(observation)

    def set_training_mode(self, mode: bool) -> None:
        """
        Put the policy in either training or evaluation mode.
        This affects certain modules, such as batch normalisation and dropout.
        :param mode: if true, set to training mode, else set to evaluation mode
        """
        self.actor.set_training_mode(mode)
        self.critic.set_training_mode(mode)
        self.training = mode



class CustomDDPG(TD3):
    def __init__(
        self,
        policy: Union[str, Type[TD3Policy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-4,
        buffer_size: int = 1000,  # 1e6
        learning_starts: int = 0,
        batch_size: int = 32,
        tau: float = 0.005,
        gamma: float = 0,
        train_freq: Union[int, Tuple[int, str]] = (4, "step"),
        gradient_steps: int = 1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[ReplayBuffer] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[torch.device, str] = "auto",
        _init_setup_model: bool = True,
        clip_sentence_embedding: float = 2.5,
    ):

        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            action_noise=action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            create_eval_env=create_eval_env,
            seed=seed,
            optimize_memory_usage=optimize_memory_usage,
            # Remove all tricks from TD3 to obtain DDPG:
            # we still need to specify target_policy_noise > 0 to avoid errors
            policy_delay=2,
            target_noise_clip=0.0,
            target_policy_noise=0.1,
            _init_setup_model=False,
        )

        # Use only one critic
        if "n_critics" not in self.policy_kwargs:
            self.policy_kwargs["n_critics"] = 1

        if _init_setup_model:
            self._setup_model()
        
        self.clip_sentence_embedding = clip_sentence_embedding
    
    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "DDPG",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> OffPolicyAlgorithm:

        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            eval_env=eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            tb_log_name=tb_log_name,
            eval_log_path=eval_log_path,
            reset_num_timesteps=reset_num_timesteps,
        )
    
    def _sample_action(
        self,
        learning_starts: int,
        action_noise: Optional[ActionNoise] = None,
        n_envs: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample an action according to the exploration policy.
        This is either done by sampling the probability distribution of the policy,
        or sampling a random action (from a uniform distribution over the action space)
        or by adding noise to the deterministic output.
        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :param n_envs:
        :return: action to take in the environment
            and scaled action that will be stored in the replay buffer.
            The two differs when the action space is not normalized (bounds are not [-1, 1]).
        """
        # Select action randomly or according to policy
        if self.num_timesteps < learning_starts and not (self.use_sde and self.use_sde_at_warmup):
            # Warmup phase
            unscaled_action = np.array([self.action_space.sample() for _ in range(n_envs)])
        else:
            # Note: when using continuous actions,
            # we assume that the policy uses tanh to scale the action
            # We use non-deterministic action in the case of SAC, for TD3, it does not matter
            unscaled_action, _ = self.predict(self._last_obs, deterministic=False)

        # Rescale the action from [low, high] to [-1, 1]
        if isinstance(self.action_space, gym.spaces.Box):
            # scaled_action = self.policy.scale_action(unscaled_action)

            # Add noise to the action (improve exploration)
            if action_noise is not None:
                # scaled_action = np.clip(scaled_action + action_noise(), -1, 1)
                unscaled_action = unscaled_action + action_noise().astype(unscaled_action.dtype)
                unscaled_action = np.concatenate(
                    [
                        unscaled_action[:, :-self.policy.actor.mu.layers_embed[-1].weight.size(0)],
                        np.clip(unscaled_action[:, -self.policy.actor.mu.layers_embed[-1].weight.size(0):],
                                -self.clip_sentence_embedding, self.clip_sentence_embedding),
                    ], axis=1)

            # We store the scaled action in the buffer
            # action = self.policy.unscale_action(scaled_action)
            buffer_action = unscaled_action
            action = buffer_action
        else:
            # Discrete case, no need to normalize or clip
            buffer_action = unscaled_action
            action = buffer_action
        return action, buffer_action
    
    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)

        # Update learning rate according to lr schedule
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])

        actor_losses, critic_losses = [], []

        for _ in range(gradient_steps):

            self._n_updates += 1
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            with torch.no_grad():
                target_q_values = replay_data.rewards

            # Get current Q-values estimates for each critic network
            current_q_values = self.critic(replay_data.observations, replay_data.actions)

            # Compute critic loss
            critic_loss = sum([F.mse_loss(current_q, target_q_values) for current_q in current_q_values])
            critic_losses.append(critic_loss.item())

            # Optimize the critics
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Delayed policy updates
            if self._n_updates % self.policy_delay == 0:
                # Compute actor loss
                actor_loss = -self.critic.q1_forward(replay_data.observations, self.actor(replay_data.observations)).mean()
                actor_losses.append(actor_loss.item())

                # Optimize the actor
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        if len(actor_losses) > 0:
            self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))