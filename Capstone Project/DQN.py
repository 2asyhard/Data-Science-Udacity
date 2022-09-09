# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html


import math
import random
import numpy as np
from collections import namedtuple, deque
from itertools import count
from PIL import Image
import config as cfg

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import environment
from copy import deepcopy
from save_result import save_data, first_save


cfg.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self):
        return random.sample(self.memory, cfg.BATCH_SIZE)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(1, cfg.FILTER_SIZE//2, kernel_size=(2,2), stride=(1,1), padding="same")
        self.bn1 = nn.BatchNorm2d(cfg.FILTER_SIZE//2)
        self.conv2 = nn.Conv2d(cfg.FILTER_SIZE//2, cfg.FILTER_SIZE, kernel_size=(2,2), stride=(1,1), padding="same")
        self.bn2 = nn.BatchNorm2d(cfg.FILTER_SIZE)
        self.conv3 = nn.Conv2d(cfg.FILTER_SIZE, cfg.FILTER_SIZE, kernel_size=(2,2), stride=(1,1), padding="same")
        self.bn3 = nn.BatchNorm2d(cfg.FILTER_SIZE)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.

        self.head = nn.Linear(cfg.FILTER_SIZE*h*w, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        # x = x.to(cfg.device)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))


def select_action(env, i_episode):
    _state = env.state
    n_actions = env.action_size

    sample = random.random()
    eps_threshold = cfg.EPS_END + (cfg.EPS_START - cfg.EPS_END) * math.exp(-1. * i_episode / cfg.EPS_DECAY)

    env.get_valid_action_list()

    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            _state = np.array([[_state]])
            _state = torch.Tensor(_state).to(cfg.device)
            selected_action =  policy_net(_state).max(1)[1].view(1, 1)
    else:
        selected_action = torch.tensor([[random.randrange(n_actions)]], device=cfg.device, dtype=torch.long)

    while env.action_list[selected_action][0] == 0:
        selected_action = torch.tensor([[random.randrange(n_actions)]], device=cfg.device, dtype=torch.long)

    return selected_action



def optimize_model():
    # if len(memory) < cfg.BATCH_SIZE:
    #     return

    memories = deepcopy(list(memory.memory))
    random.shuffle(memories)
    for i in range(0, len(memories), cfg.BATCH_SIZE):

        transitions = memories[i:i+cfg.BATCH_SIZE]
        # transitions = memory.sample()

        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=cfg.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

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
        # next_state_values = torch.zeros(cfg.BATCH_SIZE, device=cfg.device)
        next_state_values = torch.zeros(len(transitions), device=cfg.device)
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * cfg.GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        for param in policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()


def test(i_episode):
    eps_threshold = cfg.EPS_END + (cfg.EPS_START - cfg.EPS_END) * math.exp(-1. * i_episode / cfg.EPS_DECAY)
    done = False
    env.state = deepcopy(cfg.state_test)
    for step in range(cfg.MAX_TEST_STEP):
        action = select_action(env, i_episode)
        state, _, next_state, reward, done = env.step(action.item()) # _: action
        if done:
            break
    if done:
        print(f"Success, Steps: {step}")
        # print(env.state)
    else:
        print(f"Failed")
        # print(env.state)
    print(f"Epsilon threshold: {eps_threshold}")
    print()
    return step


env = environment.Environment()
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

policy_net = DQN(cfg.max_blocks, cfg.num_stacks, env.action_size).to(cfg.device)
target_net = DQN(cfg.max_blocks, cfg.num_stacks, env.action_size).to(cfg.device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

# optimizer = optim.RMSprop(policy_net.parameters())
optimizer = optim.SGD(policy_net.parameters(),
                lr=cfg.LEARNING_RATE,
                momentum=cfg.MOMENTUM)
memory = ReplayMemory(cfg.MEMORY_SIZE)

i_episode = 1
print("training started")

import time
start_time = str(time.time())

first_save(start_time) ###


while i_episode < cfg.MAX_EPISODES:
    # Initialize the environment and state
    env.reset()

    episode_data = []
    done = False
    for step in range(cfg.MAX_STEPS):
        # print(f"Episode: {i_episode}, Step: {step}")
        # Select and perform an action
        action = select_action(env, i_episode)
        state, _, next_state, reward, done = env.step(action.item()) # _: action

        reward = torch.tensor([reward], device=cfg.device)
        state = torch.Tensor(np.array([[state]])).to(cfg.device)
        next_state = torch.Tensor(np.array([[next_state]])).to(cfg.device)
        episode_data.append([state, action, next_state, reward])

        if done:
            break

    if done:
        # only add episode data to buffer when terminal state is reached
        for [state, action, next_state, reward] in episode_data:
            memory.push(state, action, next_state, reward)
        i_episode += 1
    else:
        continue

    if i_episode % cfg.TRAIN_PERIOD == 0:
        if len(memory) > cfg.BATCH_SIZE:
            optimize_model()

    # Update the target network, copying all weights and biases in DQN
    if i_episode % cfg.TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    if i_episode % cfg.TEST_PERIOD == 0:
        print(f"Test start, episode: {i_episode}")
        steps = test(i_episode)
        save_data(start_time, i_episode, steps+1)


print('Complete')
























