# -*- coding: utf-8 -*-
"""
DQN for LAbyRL.
"""
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from utils import Env, Agent
import pandas as pd

# read stt recog results
with open("../exp/pkls/recog_results_dict.pkl", "rb") as f:
    res_dict = pickle.load(f)

env = Env(res_dict)

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# For Replay Memory
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    #plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Number of Actions')
    plt.grid()
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# DQN
class DQN(nn.Module):
    def __init__(self, in_size, out_size, hidden_size):
        super(DQN, self).__init__()
        self.l1     = nn.Linear(in_size, hidden_size)
        self.l2     = nn.Linear(hidden_size, hidden_size)
        self.l3     = nn.Linear(hidden_size, hidden_size)
        self.final  = nn.Linear(hidden_size, out_size)
        self.relu   = nn.ReLU()

    def forward(self, inputs):
        h1 = self.relu(self.l1(inputs))
        h2 = self.relu(self.l2(h1))
        h3 = self.relu(self.l3(h2))
        return self.final(h3)

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

# Hyperparameters
BATCH_SIZE = 128
GAMMA = 0.5 #0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10


# File for recording episode durations
record_file = "../exp/rl_results.csv"
# Random Seed
for seed in range(1, 6):
    random.seed(seed)
    torch.manual_seed(0)

    # Initialize Agent
    agent = Agent()
    # Get number of actions
    # n_actions = len(agent.action_space)
    n_actions = res_dict["num_words"]
    print(f"n_actions: {n_actions}")
    # Get size of state
    state_size = agent.get_state().size(1)

    # Initialize policy and target network
    hidden_size = 32
    policy_net = DQN(state_size, n_actions, hidden_size).to(device)
    target_net = DQN(state_size, n_actions, hidden_size).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    learning_rate = 1e-2
    # Currently using Adam optimizer.
    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)

    # Initialize Replay Memory
    memory_size = 10000
    memory = ReplayMemory(memory_size)
    steps_done = 0
    episode_durations = []

    # Training Loop

    num_episodes = 5
    # suc_per_100 = 0
    for i_episode in range(num_episodes):
        print(f"Episode: {i_episode}")
        # Initialize the environment and state
        agent.reset()
        
        last_state = agent.get_state().to(device)
        current_state = agent.get_state().to(device)
        state = current_state
        
        for t in count():
            # Select and perform an action
            action = select_action(state)
            x_change, y_change, z_change = env.feedback(action)
            reward, done = agent.evaluate_reward(x_change, y_change, z_change)
            
            reward = torch.tensor([reward], device=device)

            # Observe new state
            last_state = current_state
            current_state = agent.get_state().to(device)
            if not done:
                next_state = current_state
            else:
                next_state = None

            # print(f"Episode: {i_episode}, Action: {action.data}, Done: {done}, Last State: {last_state.data}, Current State: {current_state.data}, Reward: {reward}")
            # Store the transitions in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the target network)
            optimize_model()
            if done:
                episode_durations.append(t + 1)
                plot_durations()
                break
            
        # Update the target network, copying all weights and biases in DQN
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

    print(f'Seed {seed} Complete')

    plt.ioff()
    pic_name = "../exp/res_imgs/result_" + str(seed) + ".png"
    plt.savefig(pic_name)
    
    df = pd.DataFrame(episode_durations, columns=["Seed" + str(seed)]).T
    df.to_csv(record_file, index=True, header=False, mode='a')