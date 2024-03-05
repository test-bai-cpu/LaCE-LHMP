import utils
import warnings

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize


def plot_LT_map(LT_histogram_data):
    all_histogram = LT_histogram_data["histogram"]
    centers = LT_histogram_data["centers"]
    theta_values = []
    rho_values = []

    # Iterate over each histogram in all_histogram
    for i in range(all_histogram.shape[2]):
        histogram = all_histogram[:, :, i]
        
        # Get the row and column index of the maximum value in histogram
        row_index, column_index = np.unravel_index(np.argmax(histogram), histogram.shape)
        
        # Compute theta and rho
        theta = np.deg2rad(column_index * 10)
        rho = (row_index + 1) * 0.2
        
        theta_values.append(theta)
        rho_values.append(rho)

    # Convert lists to numpy arrays
    theta_values = np.array(theta_values)
    rho_values = np.array(rho_values)

    (u, v) = utils.pol2cart(rho_values, theta_values)
    color = theta_values
    plt.quiver(centers[:, 0], centers[:, 1], u, v, color, cmap="hsv", scale_units='xy', scale=1, width=0.004, alpha=0.5)
    
    
def plot_all_predicted_trajs(total_predicted_motion_list, observed_tracklet_length=8):
    for predicted_traj in total_predicted_motion_list:
        time_list = predicted_traj[:, 0]
        (u, v) = utils.pol2cart(predicted_traj[:, 3], predicted_traj[:, 4])

        plt.plot(predicted_traj[:, 1], predicted_traj[:, 2], 'b', alpha=1)
        for i in range(0, observed_tracklet_length):
            plt.scatter(predicted_traj[i, 1], predicted_traj[i, 2], color="limegreen", marker="o", s=10)
        plt.scatter(predicted_traj[observed_tracklet_length:, 1], predicted_traj[observed_tracklet_length:, 2], color="b", marker="o", s=10)


def plot_human_traj_v2(human_traj_data, observed_tracklet_length=8):
    plt.plot(human_traj_data[:observed_tracklet_length+1, 1], human_traj_data[:observed_tracklet_length+1, 2], color="limegreen", label="Observation", lw=3)
    plt.scatter(human_traj_data[:observed_tracklet_length+1, 1], human_traj_data[:observed_tracklet_length+1, 2], marker='o', alpha=1, color="limegreen", s=5)

    plt.plot(human_traj_data[observed_tracklet_length:, 1], human_traj_data[observed_tracklet_length:, 2], color="r", label="Ground truth", lw=3)
    plt.scatter(human_traj_data[observed_tracklet_length:, 1], human_traj_data[observed_tracklet_length:, 2], marker='o', alpha=1, color="r", s=20)
