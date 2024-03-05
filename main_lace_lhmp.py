# Standard library imports
import argparse
import csv
import mmap
import os
import sys
import time
from pprint import pprint

# Related third-party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

# Local imports
import plot_figures
import utils
from trajectory_predictor_wind import TrajectoryPredictor


def get_all_person_id(human_traj_file):
    data = pd.read_csv(human_traj_file, header=None)
    data.columns = ["time", "person_id", "x", "y", "velocity", "motion_angle"]
    person_id_list = list(data.person_id.unique())
    return person_id_list


def get_human_traj_data_by_person_id(human_traj_origin_data, person_id):
    human_traj_data_by_person_id = human_traj_origin_data.loc[human_traj_origin_data['person_id'] == person_id]
    human_traj_array = human_traj_data_by_person_id[["time", "x", "y", "velocity", "motion_angle"]].to_numpy()

    return human_traj_array


def run_experiment(LT_histogram_file, histogram_divergence_file, human_traj_file, atc_date, result_file, args):
    LT_map_data = utils.read_LT_histogram_data(LT_histogram_file)
    histogram_divergence = pd.read_csv(histogram_divergence_file)
    human_traj_data = utils.read_wind_human_traj_data(human_traj_file)
    
    person_id_list = get_all_person_id(human_traj_file)

    header = ["person_id", "predict_horizon", "x", "y", "FDE_mean", "FDE_std", "FDE_min", "ADE_mean", "ADE_std", "ADE_min"]
    with open(result_file, "w", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)

    ####### Update here to test on all person_ids in this atc date
    for person_id in [9103800]:
    # for person_id in tqdm(person_id_list):
        
        ################## For save plot of each prediction ###################################
        plt.clf()
        plt.close('all')
        plt.figure(figsize=(10, 6), dpi=100)
        plt.subplot(111, facecolor='white')
        img = plt.imread("atc_map/localization_grid_white.jpg")
        plt.imshow(img, cmap='gray', vmin=0, vmax=255, extent=[-60, 80, -40, 20])
        plot_figures.plot_LT_map(LT_map_data)
        #######################################################################################
        
        human_traj_data_by_person_id = get_human_traj_data_by_person_id(human_traj_data, person_id)
        
        trajectory_predictor = TrajectoryPredictor(
            LT_map_origin_data=LT_map_data,
            histogram_divergence = histogram_divergence,
            human_traj_origin_data=human_traj_data_by_person_id,
            person_id=person_id,
            start_length=args.start_length,
            observed_tracklet_length=args.observed_tracklet_length,
            max_planning_horizon=args.planning_horizon,
            delta_t=args.delta_t,
            result_file=result_file,
            r_s=args.r_s,
            beta=args.beta,
            generate_traj_num=args.generate_traj_num,
        )
        
        if not trajectory_predictor.check_human_traj_data():
            continue

        if args.pure_CVM:
            all_predicted_trajectory_list = trajectory_predictor.predict_one_human_traj_pure_cvm()
        else:
            all_predicted_trajectory_list = trajectory_predictor.predict_one_human_traj_with_mod()

        #### Evaluate with all k traj and compute mean ADE/FDE.evaluate_ADE_FDE_result_most_likely
        if args.if_rank:
            predicted_horizon, res_FDE, res_ADE = trajectory_predictor.evaluate_ADE_FDE_result_most_likely(all_predicted_trajectory_list)
        #### Evaluate with most likely traj. For example, generate 20 trajs, and choose the most likely one, calcuated by prob of cliff.
        else: # args.if_rank is False:
            predicted_horizon, res_FDE, res_ADE = trajectory_predictor.evaluate_ADE_FDE_result_mean(all_predicted_trajectory_list)

    
        ################## For save plot of each prediction ###################################
        textstr = '\n'.join((
            "horizon = " + str(predicted_horizon),
            "ADE = " + "{:.3f}".format(round(res_ADE, 3)),
            "FDE = " + "{:.3f}".format(round(res_FDE, 3))))
        # these are matplotlib.patch.Patch properties
        props = dict(boxstyle='round', facecolor = "gainsboro", alpha=1)
        # place a text box in upper left in axes coords
        plt.text(-27, 19, textstr, fontsize=15,
                verticalalignment='top', bbox=props)

        plot_figures.plot_all_predicted_trajs(all_predicted_trajectory_list, observed_tracklet_length=args.observed_tracklet_length)
        plot_figures.plot_human_traj_v2(human_traj_data_by_person_id[args.start_length:, :], observed_tracklet_length=args.observed_tracklet_length)

        plt.rcParams['pdf.fonttype'] = 42
        plt.rcParams['ps.fonttype'] = 42
        
        plt.xlim(-30, 5)
        plt.ylim(-15, 20)
        plt.title(f"person_id: {person_id}, in date: {atc_date}")
        os.makedirs("quality_res/LT", exist_ok=True)
        plt.savefig(f"quality_res/LT/LT_{person_id}_in_date{atc_date}.png", bbox_inches='tight')
        # plt.show()
        ########################################################################################

def main(args):
    LT_histogram_file = "MoDs/LT_histogram/clustered_Laminar_flow_1024.mat"
    ## Load histogram divergence, which is divergence between LT raw histogram and laminar histogram.
    histogram_divergence_file = "MoDs/LT_histogram/histogram_divergences.csv"
    
    result_folder = f"results/{args.version}"
    os.makedirs(result_folder, exist_ok=True)
    
    ####### Specify the date of the atc data to be used for testing.
    # for atc_date in ["1028", "1031", "1104", "1107", "1111", "1114", "1118", "1121", "1125"]:
    for atc_date in ["1031"]:
        human_traj_file = f"atc_data/middle_area/{atc_date}.csv"
        result_file = f"{result_folder}/{atc_date}.csv"
        
        run_experiment(LT_histogram_file, histogram_divergence_file, human_traj_file, atc_date, result_file, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Set parameters for the lace-lhmp.")

    parser.add_argument("--r_s", type=float, default=1, help="Sampling radius size.")
    parser.add_argument("--beta", type=float, default=1, help="Bias from cvm.")
    parser.add_argument("--generate_traj_num", type=int, default=5, help="Number of trajectories to generate, which denoted as k.")
    parser.add_argument("--version", type=str, default="version1", help="Version information, to set unique version for each test.")
    parser.add_argument("--no_rank", action="store_false", dest="if_rank", help="Add no_rank to generate k trajectories and compute mean as ADE/FDE. Default setting is using rank to use most likely output config.")
    parser.add_argument("--observed_tracklet_length", type=int, default=3, help="Length of observed trajectory.")
    parser.add_argument("--planning_horizon", type=int, default=20, help="Planning horizon.")
    parser.add_argument("--delta_t", type=int, default=1, help="Prediction time step.")
    parser.add_argument("--start_length", type=int, default=0, help="Start length of the trajectory, use this for debug one long trajectory, for example, want to start from middle of it.")
    parser.add_argument("--pure_CVM", type=bool, default=False, help="Set to true to use pure CVM predictor, set to false to use Lace-LHMP.")
    
    args = parser.parse_args()
    pprint(args)
    main(args)
    
## To run this: 
# python3 main_lace_lhmp.py --r_s 1 --beta 1 --generate_traj_num 5 --version "version1" --observed_tracklet_length 3 --planning_horizon 20 --delta_t 1
