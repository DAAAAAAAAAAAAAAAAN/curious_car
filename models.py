import torch
from torch import nn

__all__ = ['StatePredictor', 'QNetwork']


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
        return self.model(
            torch.cat((state, action_one_hot), dim=1).to(self.device))


class QNetwork(nn.Module):

    def __init__(self, device="cpu", num_hidden=128):
        nn.Module.__init__(self)
        self.device = torch.device(device)
        self.l1 = nn.Linear(2, num_hidden)
        self.l2 = nn.Linear(num_hidden, 3)

        self.to(device)

    def forward(self, x):
        out = torch.relu(self.l1(x))
        return self.l2(out)
