import numpy as np
import pandas as pd
from math import dist
import json
import math
from scipy.io import loadmat


def read_LT_histogram_data(LT_histogram_file):
    mat_data = loadmat(LT_histogram_file)

    centers = mat_data['clustered_Laminar_flow'][0][0][0]
    idx = mat_data['clustered_Laminar_flow'][0][0][1]
    histogram = mat_data['clustered_Laminar_flow'][0][0][2]
    
    LT_histogram_data = {
        "centers": centers,
        "idx": idx,
        "histogram": histogram
    }

    return LT_histogram_data


# atc-wind-proj. Already convert to meter, in the data pre-process step
# In atc_data folder, the atc data are convert to frame, 1hz, means how many frames from 9:00 am japan time.
def read_wind_human_traj_data(human_traj_file):
    data = pd.read_csv(human_traj_file, header=None)
    data.columns = ["time", "person_id", "x", "y", "velocity", "motion_angle"]

    return data


def get_euclidean_distance(position_array_1, position_array_2):
    return dist(position_array_1, position_array_2)


def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y


def wrapTo2pi(circular_value):
    return np.round(np.mod(circular_value,2*np.pi), 3)


def _circfuncs_common(samples, high, low):
    # Ensure samples are array-like and size is not zero
    if samples.size == 0:
        return np.nan, np.asarray(np.nan), np.asarray(np.nan), None

    # Recast samples as radians that range between 0 and 2 pi and calculate
    # the sine and cosine
    sin_samp = np.sin((samples - low)*2.* np.pi / (high - low))
    cos_samp = np.cos((samples - low)*2.* np.pi / (high - low))

    return samples, sin_samp, cos_samp


def circmean(samples, weights, high=2*np.pi, low=0):
    samples = np.asarray(samples)
    weights = np.asarray(weights)
    samples, sin_samp, cos_samp = _circfuncs_common(samples, high, low)
    sin_sum = sum(sin_samp * weights)
    cos_sum = sum(cos_samp * weights)
    res = np.arctan2(sin_sum, cos_sum)
    res = res*(high - low)/2.0/np.pi + low
    return wrapTo2pi(res)


def circdiff(circular_1, circular_2):
    res = np.arctan2(np.sin(circular_1-circular_2), np.cos(circular_1-circular_2))
    return abs(res)