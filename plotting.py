import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from models import QNetwork
import pandas as pd
import seaborn as sns

__all__ = ['visualize_policy']


def _test_policy(state):
    model = QNetwork(device="cuda")
    action = model(state.to(model.device))
    return action



def visualize_confidence_bounds(episodes, confidence_type = "95_ci"):
    # Possible percentile work-around in seaborn: https://stackoverflow.com/questions/37767719/timeseries-plot-with-min-max-shading-using-seaborn
    print(list(episodes))
    print(episodes)
    sns.set(style="darkgrid")
    # f, axes = plt.subplots(2, 1, sharex=True)

    # grid = sns.FacetGrid(episodes, row = "episode")
    # grid = sns.FacetGrid(episodes, row = "timepoint")
    # grid()

    plt.subplot(211)
    sns.relplot(x='episode', y='max_x', hue='target_reward', style='target_reward', kind='line', data = episodes)
    sns.relplot(x='episode', y='total_intrinsic_reward', hue='target_reward', style='target_reward', kind='line', data = episodes)

    # sns.relplot(x='timepoint', y='signal', style='target_reward', kind='line', data = episodes)


    # plt.tight_layout(axes)
    plt.show()


def episode_data_to_dataframe(path):
    # Needs a path to a directory with csv files containing the data of different runs.
    # Random_seed_num, Episode, Max-x, max_time_step_of_episode(extrinsic), intrinsic_reward(mean), min,max,median

    all_dfs = []

    for i, filename in enumerate(os.listdir(path)):
        if filename.endswith('csv'):
            # Extract headers from first csv
                episodes_df = pd.DataFrame.from_csv(path+"/"+filename, header = 0, index_col=1)

                # add index as column per df so we can use it as x-axis
                episodes_df.reset_index(inplace=True)
                all_dfs.append(episodes_df)

    all_episodes = pd.concat(all_dfs, axis=0, ignore_index=True)
    print(all_episodes.head())

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
    print(fmri)
    path = "./experiments"
    episodes = episode_data_to_dataframe(path)

    visualize_confidence_bounds(episodes)
