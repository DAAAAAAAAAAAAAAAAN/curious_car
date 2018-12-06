import gym
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

import constants
from ReplayMemory import ReplayMemory


class StatePredictor(nn.Module):
    def __init__(self, num_state, num_action, num_hidden, device):
        super(StatePredictor, self).__init__()
        self.num_state = num_state
        self.num_action = num_action
        self.num_hidden = num_hidden
        self.device = device

        self.model = nn.Sequential(
            nn.Linear(num_state + num_action, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_state))

        self.to(device)

    def forward(self, state, action):
        batch_size = action.size()[0]
        action_one_hot = torch.zeros((batch_size, self.num_action),
                                     device=self.device)

        action_one_hot = action_one_hot.scatter(1, action.unsqueeze(1), 1) \
            .to(torch.float)
        return self.model(torch.cat((state, action_one_hot), dim=1).to(self.device))


def train(model, memory, optimizer):
    loss = nn.MSELoss()

    if len(memory) < constants.BATCH_SIZE:
        return None

    # random transition batch is taken from experience replay memory
    transitions = memory.sample(constants.BATCH_SIZE)

    # transition is a list of 4-tuples, instead we want 4 vectors (as torch.Tensor's)
    state, action, _, next_state, _ = zip(*transitions)

    # convert to PyTorch and define types
    state = torch.tensor(state, dtype=torch.float)
    action = torch.tensor(action,
                          dtype=torch.int64)  # Need 64 bit to use them as index
    next_state = torch.tensor(next_state, dtype=torch.float)

    prediction = model(state, action)

    output = loss(prediction, next_state.to(model.device))
    output.backward()
    optimizer.step()

    return output.item()


def run_episodes(train, model, memory, env, num_episodes, writer):
    optimizer = optim.Adam(model.parameters(), lr=constants.LR_STATE_PREDICTOR)

    global_steps = 0  # Count the steps (do not reset at episode start, to compute epsilon)
    episode_durations = []  #
    for i in tqdm(range(num_episodes)):

        # initialize episode
        done = False
        state = env.reset()
        ep_length = 0

        # keep acting until terminal state is reached
        while not done:
            # calculate next action
            action = env.action_space.sample()
            # perform action
            next_state, reward, done, _ = env.step(action)

            # remeber transition
            memory.push((state, action, reward, next_state, done))
            state = next_state

            ep_length += 1
            global_steps += 1

            # updade model
            loss = train(model, memory, optimizer)

            print(f"loss: {loss}")
            # writer.add_scalar("loss", loss or 0, global_steps)

        episode_durations.append(ep_length)

    return episode_durations


def main():
    env = gym.make("MountainCar-v0")
    # writer = SummaryWriter()
    device = torch.device("cuda")

    memory = ReplayMemory(constants.REPLAY_MEMORY_SIZE)
    model = StatePredictor(num_state=env.observation_space.shape[0],
                           num_action=3,
                           num_hidden=128,
                           device=device)

    num_episodes = 1000

    run_episodes(train, model, memory, env, num_episodes, None)


if __name__ == '__main__':
    main()
