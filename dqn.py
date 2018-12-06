__author__ = 'Aron'

import random
import numpy as np
import torch
from torch import optim, nn
import torch.nn.functional as F
import gym
from tqdm import tqdm as _tqdm
from ReplayMemory import ReplayMemory
from matplotlib import pyplot as plt

def tqdm(*args, **kwargs):
    return _tqdm(*args, **kwargs, mininterval=1)  # Safety, do not overflow buffer

EPS = float(np.finfo(np.float32).eps)

env = gym.envs.make("MountainCar-v0")

class QNetwork(nn.Module):

    def __init__(self, num_hidden=128):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(2, num_hidden)
        self.l2 = nn.Linear(num_hidden, 3)

    def forward(self, x):
        out = torch.relu(self.l1(x))
        out = self.l2(out)

        return out

def select_action(model, state, epsilon):

    with torch.no_grad():

        state = torch.tensor(state.astype(np.float32))

        # compute action values
        action_values = model(state)

        # determine greedy and random action
        prob_a, greedy_a = action_values.max(dim=0)
        greedy_a = greedy_a.item()

        # determine action to choose based on eps
        if random.random() < epsilon:
            return random.choice([0,1,2])

        return greedy_a

def get_epsilon(it):
    it = min(it, 999)
    linear = np.linspace(1, 0.05, 1000)
    return linear[it]

def compute_q_val(model, state, action):
    output = model(torch.tensor(state, dtype=torch.float))
    return output[np.arange(output.size(0)), action]

def compute_target(model, reward, next_state, done, discount_factor):
    # done is a boolean (vector) that indicates if next_state is terminal (episode is done)

    output = model(next_state)
    q_values, _ = output.max(dim=1)

    # mutilpy q_values with 0 for terminal states
    return reward + discount_factor * (done == False).type(torch.float) * q_values

def train(model, memory, optimizer, batch_size, discount_factor):
    # DO NOT MODIFY THIS FUNCTION

    # don't learn without some decent experience
    if len(memory) < batch_size:
        return None

    # random transition batch is taken from experience replay memory
    transitions = memory.sample(batch_size)

    # transition is a list of 4-tuples, instead we want 4 vectors (as torch.Tensor's)
    state, action, reward, next_state, done = zip(*transitions)

    # convert to PyTorch and define types
    state = torch.tensor(state, dtype=torch.float)
    action = torch.tensor(action, dtype=torch.int64)  # Need 64 bit to use them as index
    next_state = torch.tensor(next_state, dtype=torch.float)
    reward = torch.tensor(reward, dtype=torch.float)
    done = torch.tensor(done, dtype=torch.uint8)  # Boolean

    # compute the q value
    q_val = compute_q_val(model, state, action)

    with torch.no_grad():  # Don't compute gradient info for the target (semi-gradient)
        target = compute_target(model, reward, next_state, done, discount_factor)

    # loss is measured from error between current and newly expected Q values
    loss = F.smooth_l1_loss(q_val, target)

    # backpropagation of loss to Neural Network (PyTorch magic)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()  # Returns a Python scalar, and releases history (similar to .detach())


def render(start, actions, seed):
    import time

    env.seed(seed)
    state = env.reset()

    assert (state == start).all(), (state, start)

    env.render()
    ep_length = 0
    max_x = state[0]

    for action in actions:

        # perform action
        _, _, done, _ = env.step(action)
        env.render()
        time.sleep(0.05)

        ep_length += 1

    print('finished in', ep_length)
    env.close()  # Close the environ


def run_episodes(train, model, memory, env, num_episodes, batch_size, discount_factor, learn_rate):

    optimizer = optim.Adam(model.parameters(), learn_rate)

    global_steps = 0  # Count the steps (do not reset at episode start, to compute epsilon)
    episode_durations = []  #
    losses =[]
    for i in tqdm(range(num_episodes)):

        env.seed(i)
        random.seed(i)

        # initialize episode
        done = False
        state = env.reset()
        ep_length = 0
        max_x = state[0]

        # save action for rendering
        actions = []
        start = state

        # keep acting until terminal state is reached
        while not done:
            # calculate next action
            epsilon = get_epsilon(global_steps)
            action = select_action(model, state, epsilon)

            actions.append(action)

            # perform action
            next_state, reward, done, _ = env.step(action)

            # remember transition
            memory.push((state, action, reward, next_state, done))
            state = next_state

            ep_length += 1
            global_steps += 1
            max_x = max(max_x, state[0])

            # update model
            loss = train(model, memory, optimizer, batch_size, discount_factor)

        if i == 0 or ep_length < 200:
            with torch.no_grad():
                render(start, actions, i)

        episode_durations.append(ep_length)
        losses.append(loss)
        # print(episode_durations, loss)

    return episode_durations, losses

# Let's run it!
num_episodes = 500
batch_size = 64
discount_factor = 0.97
learn_rate = 5e-4
memory = ReplayMemory(10000)
num_hidden = 200
seed = 42  # This is not randomly chosen

# We will seed the algorithm (before initializing QNetwork!) for reproducability
random.seed(seed)
torch.manual_seed(seed)
env.seed(seed)

model = QNetwork(num_hidden)

episode_durations, episode_loss = run_episodes(train, model, memory, env, num_episodes, batch_size, discount_factor, learn_rate)
print(episode_durations)
print()
print(episode_loss)
