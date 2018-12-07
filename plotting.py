import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch

from models import QNetwork

__all__ = ['visualize_policy']


def _test_policy(state):
    model = QNetwork(device="cuda")
    action = model(state.to(model.device))
    return action



def visualize_policy(policy):
    min_position = -1.2
    max_position = 0.6
    max_speed = 0.07

    x = np.linspace(min_position, max_position, 100)
    y = np.linspace(-max_speed, max_speed, 100)
    x, y = np.meshgrid(x, y)
    xx = torch.tensor(x, dtype=torch.float).view(-1)
    yy = torch.tensor(y, dtype=torch.float).view(-1)
    state = torch.stack((xx, yy), dim=1)

    output = policy(state)
    action = output.argmax(dim=1)
    cmap = plt.cm.get_cmap('coolwarm', 3)
    plt.pcolormesh(x, y, action.unsqueeze(0).view(100, 100).to("cpu").numpy(),
                   cmap=cmap)
    plt.xlabel("Position")
    plt.ylabel("Velocity")
    plt.legend([mpatches.Patch(color=cmap(b)) for b in (0, 1, 2)],
               ("Left", "Nothing", "Right"))
    plt.show()

if __name__ == '__main__':

    model = QNetwork(num_hidden=200)
    model.load_state_dict(torch.load("checkpoints/2018-12-07-1544193677-q-model.data"))

    visualize_policy(model)
