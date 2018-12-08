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



def visualize_confidence_bounds(episodes, confidence_type = "95_ci"):
    # Possible percentile work-around in seaborn: https://stackoverflow.com/questions/37767719/timeseries-plot-with-min-max-shading-using-seaborn
    print(episodes.head())
    sns.set(style="darkgrid")
    f, axes = plt.subplots(2, 1, sharex=True)

    # grid = sns.FacetGrid(episodes, row = "episode")
    # grid = sns.FacetGrid(episodes, row = "timepoint")
    # grid()


    sns.relplot(x='timepoint', y='signal', hue='event', style='event', kind='line', data = episodes)
    sns.relplot(x='timepoint', y='signal', style='event', kind='line', data = episodes)

    # plt.setp()
    plt.tight_layout(axes)
    plt.show()


def episode_data_to_dataframe(path):
    # Needs a path to a directory with csv files containing the data of different runs.
    # Random_seed_num, Episode, Max-x, max_time_step_of_episode(extrinsic), intrinsic_reward(mean), min,max,median

    all_dfs = []

    for i, filename in enumerate(os.listdir(path)):
        if filename.endswith('csv'):
            # Extaract headers from first csv
                episodes_df = pd.DataFrame.from_csv(path+filename, headers = 0)
                all_dfs.append(episodes_df)

    all_episodes = pd.concat(all_dfs, axis=0, ignore_index=True)

    return all_episodes



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
    # visualize_policy(_test_policy)
    fmri = sns.load_dataset("fmri")


    visualize_confidence_bounds(fmri)
