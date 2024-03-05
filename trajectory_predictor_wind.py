# Standard library imports
import math
import random
import time
import csv
import warnings
import sys
from typing import Any, Optional
from pprint import pprint

# Related third-party imports
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy
from scipy import stats

# Local imports
import utils
import evaluation


class TrajectoryPredictor:
    def __init__(
            self,
            *,
            LT_map_origin_data,
            histogram_divergence,
            human_traj_origin_data,
            atc_map_data: Optional[Any] = None,
            person_id: int = None,
            start_length: int = 0,
            observed_tracklet_length: int = 1,
            max_planning_horizon: int = 50,
            delta_t: int = 1,
            result_file: str,
            r_s : float = 1.0,
            beta : float = 1.0,
            generate_traj_num : int = 10,
    ):
        self.LT_histogram_map = LT_map_origin_data
        self.histogram_divergence = histogram_divergence
        self.human_traj_data = human_traj_origin_data
        self.atc_map = atc_map_data
        self.person_id = person_id
        self.start_length = start_length
        self.observed_tracklet_length = observed_tracklet_length
        self.max_planning_horizon = max_planning_horizon
        self.planning_horizon = None
        self.delta_t = delta_t
        self.result_file = result_file
        self.skipped_person_ids = []
        self.r_s = r_s
        self.beta = beta
        self.generate_traj_num = generate_traj_num

    def set_planning_horizon(self):
        ground_truth_time = math.floor(self.human_traj_data[-1,0] - self.human_traj_data[self.start_length + self.observed_tracklet_length,0])
        self.planning_horizon = min(ground_truth_time, self.max_planning_horizon)

    def check_human_traj_data(self):
        ## ATC data can contain person_id is -1 and traj is 0
        if self.person_id == -1:
            return False
        
        row_num = self.human_traj_data.shape[0]
        
        ## Filter out the traj which are shorter than our prediction horizon
        if row_num <= self.start_length + self.observed_tracklet_length + 1:
            return False
        if (self.human_traj_data[-1,0] - self.human_traj_data[self.start_length + self.observed_tracklet_length,0]) < self.delta_t:
            return False

        ## Filter out the traj which are stationary
        idle_threshold = 1
        idle_check_interval = 5
        distance_array = self.human_traj_data[:-idle_check_interval, 1:3] - self.human_traj_data[idle_check_interval:, 1:3]
        distance = np.sqrt(np.power(distance_array[:,0], 2) + np.power(distance_array[:,1], 2))
        if np.any(distance < idle_threshold):
            return False

        self.set_planning_horizon()
        
        return True

    def predict_one_human_traj_with_mod(self):
        total_predicted_motion_list = []
        
        ##### use ATLAS speed, more details can check lace-lhmp paper, or origin ATLAS paper (https://arxiv.org/abs/2207.09830)
        current_motion_origin = self._calculate_current_motion()
        ##### use original speed
        # current_motion_origin = np.copy(self.human_traj_data[self.start_length + self.observed_tracklet_length, :])

        for _ in range(self.generate_traj_num):
            current_motion = current_motion_origin
            predicted_motion_list = np.copy(self.human_traj_data[self.start_length:self.start_length + self.observed_tracklet_length, :])
            predicted_motion_list = np.concatenate((predicted_motion_list, np.ones((predicted_motion_list.shape[0], 1))), axis=1)

            log_likelihood_sum = 0
            for time_index in range(1, int(self.planning_horizon / self.delta_t) + 2):
                
                # sampled_velocity: (rho, theta)
                nearest_index = self._find_nearest_center_in_histogram(current_motion)
                if nearest_index is None:
                    break
                else:
                    sampled_velocity, new_prob, kl_div_value, js_div_value = self._sample_motion_from_histogram(nearest_index)
                    updated_motion = self._apply_sampled_motion_to_current_motion(
                        sampled_velocity, current_motion, time_index, kl_div_value, js_div_value
                    )
                    log_likelihood_sum += math.log(new_prob)
                    updated_motion = np.append(updated_motion, [log_likelihood_sum])
                predicted_motion_list = np.append(predicted_motion_list, [updated_motion], axis=0)
                current_motion = self._predict_with_constant_velocity_model(updated_motion)

            total_predicted_motion_list.append(predicted_motion_list)

        return total_predicted_motion_list

    def predict_one_human_traj_pure_cvm(self):
        total_predicted_motion_list = []
        
        ##### use ATLAS speed, more details can check lace-lhmp paper, or origin ATLAS paper (https://arxiv.org/abs/2207.09830)
        current_motion_origin = self._calculate_current_motion()
        ##### use original speed
        # current_motion_origin = np.copy(self.human_traj_data[self.start_length + self.observed_tracklet_length, :])

        for _ in range(1):
            current_motion = current_motion_origin
            predicted_motion_list = np.copy(self.human_traj_data[self.start_length:self.start_length + self.observed_tracklet_length, :])
            for time_index in range(1, int(self.planning_horizon / self.delta_t) + 2):
                updated_motion = current_motion
                predicted_motion_list = np.append(predicted_motion_list, [updated_motion], axis=0)
                current_motion = self._predict_with_constant_velocity_model(updated_motion)

            total_predicted_motion_list.append(predicted_motion_list)

        return total_predicted_motion_list

    def evaluate_ADE_FDE_result_mean(self, all_predicted_trajectory_list):
        res_FDE = 0
        res_ADE = 0
        human_traj_data = self.human_traj_data[self.start_length:, :]
        
        start_predict_position = self.observed_tracklet_length
        error_matrix = []
        for predicted_traj in all_predicted_trajectory_list:
            error_list = evaluation.get_error_list_with_predict_timestamp(human_traj_data, predicted_traj, start_predict_position)
            error_matrix.append([round(num, 3) for num in error_list])

        max_planning_horizon = max([len(row) for row in error_matrix])

        for time_index in range(1, max_planning_horizon + 1):
            traj_error_matrix_for_one_time_index = []
            for error_list in error_matrix:
                if len(error_list) >= time_index:
                    traj_error_matrix_for_one_time_index.append(error_list[:time_index])
            num_predicted_trajs = len(traj_error_matrix_for_one_time_index)
            traj_error_array_for_one_time_index = np.array(traj_error_matrix_for_one_time_index)
            FDE_mean = round(np.mean(traj_error_array_for_one_time_index[:,-1]), 3)
            FDE_std = round(np.std(traj_error_array_for_one_time_index[:,-1]), 3)
            FDE_min = round(np.min(traj_error_array_for_one_time_index[:,-1]), 3)
            ADE_list = np.mean(traj_error_array_for_one_time_index, axis=1)
            ADE_mean = round(np.mean(ADE_list), 3)
            ADE_std = round(np.std(ADE_list), 3)
            ADE_min = round(np.min(ADE_list), 3)

            try:
                x = human_traj_data[start_predict_position + time_index - 1, 1]
                y = human_traj_data[start_predict_position + time_index - 1, 2]
            except:
                print("person_id is: ", self.person_id)
                raise
            
            data_row = [self.person_id, round(time_index*self.delta_t, 1), x, y, FDE_mean, FDE_std, FDE_min, ADE_mean, ADE_std, ADE_min]

            res_FDE = FDE_mean
            res_ADE = ADE_mean

            with open(self.result_file, "a", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(data_row)

        max_planning_horizon = round(max_planning_horizon * self.delta_t, 1)
        
        return max_planning_horizon, res_FDE, res_ADE



    def evaluate_ADE_FDE_result_most_likely(self, all_predicted_trajectory_list):
        res_FDE = 0
        res_ADE = 0
        human_traj_data = self.human_traj_data[self.start_length:, :]
        
        start_predict_position = self.observed_tracklet_length
        error_matrix = []
        weight_matrix = []
        for predicted_traj in all_predicted_trajectory_list:
            error_list = evaluation.get_error_list_with_predict_timestamp(human_traj_data, predicted_traj, start_predict_position)
            error_matrix.append(error_list)
            weight_list = list(predicted_traj[start_predict_position+1:,-1])
            weight_matrix.append(weight_list)
            
        # print("###### error matrix is: #######")
        # for tmp_error in error_matrix:
        #     print(["{0:0.3f}".format(k) for k in tmp_error])
        # print("#################################")
        
        max_planning_horizon = max([len(row) for row in error_matrix])

        for time_index in range(1, max_planning_horizon + 1):
            traj_error_matrix_for_one_time_index = []
            weight_matrix_for_one_time_index = []
            for i in range(len(error_matrix)):
                error_list = error_matrix[i]
                weight_list = weight_matrix[i]
                if len(error_list) >= time_index:
                    traj_error_matrix_for_one_time_index.append(error_list[:time_index])
                    weight_matrix_for_one_time_index.append(weight_list[:time_index])

            num_predicted_trajs = len(traj_error_matrix_for_one_time_index)
            traj_error_array_for_one_time_index = np.array(traj_error_matrix_for_one_time_index)
            weight_matrix_for_one_time_index = np.array(weight_matrix_for_one_time_index)

            last_col_weight = weight_matrix_for_one_time_index[:, -1]
            index_of_largest = np.argmax(last_col_weight)
            most_likely_traj_error = traj_error_array_for_one_time_index[index_of_largest]

            FDE_mean = round(most_likely_traj_error[-1], 3)
            FDE_std = 0
            FDE_min = FDE_mean
            ADE_mean = round(np.mean(most_likely_traj_error), 3)
            ADE_std = 0
            ADE_min = ADE_mean
            
            try:
                x = human_traj_data[start_predict_position + time_index - 1, 1]
                y = human_traj_data[start_predict_position + time_index - 1, 2]
            except:
                print("person_id is: ", self.person_id)
                raise
            
            data_row = [self.person_id, round(time_index*self.delta_t, 1), x, y, FDE_mean, FDE_std, FDE_min, ADE_mean, ADE_std, ADE_min]

            res_FDE = FDE_mean
            res_ADE = ADE_mean

            with open(self.result_file, "a", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(data_row)
        
        max_planning_horizon = round(max_planning_horizon * self.delta_t, 1)

        return max_planning_horizon, res_FDE, res_ADE

    def _calculate_current_motion(self):
        current_motion_origin = self.human_traj_data[self.start_length + self.observed_tracklet_length, :]
        ## parameter used in ATLAS benchmark
        sigma = 1.5
        current_speed = 0
        current_orientation = 0
        g_t = [1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- t ** 2 / (2 * sigma ** 2)) for t in range(1, self.observed_tracklet_length + 1)]
        g_t = [g/sum(g_t) for g in g_t]
        g_t = np.flip(g_t)
        raw_speed_list = self.human_traj_data[self.start_length : self.start_length + self.observed_tracklet_length, 3]
        raw_orientation_list = self.human_traj_data[self.start_length : self.start_length + self.observed_tracklet_length, 4]
        weighted_speed_list = raw_speed_list * g_t
        current_speed = np.sum(weighted_speed_list)
        wrapped_orientation = utils.wrapTo2pi(raw_orientation_list)
        current_orientation = utils.circmean(wrapped_orientation, g_t)
        current_motion = np.concatenate((current_motion_origin[0:3], [current_speed, current_orientation]))
        return current_motion

    def _predict_with_constant_velocity_model(self, updated_motion):
        new_position = self._get_next_position_by_velocity(updated_motion[1:3], updated_motion[3:5])
        new_timestamp = np.array([round(updated_motion[0] + self.delta_t, 1)])
        predicted_motion = np.concatenate((new_timestamp, new_position, updated_motion[3:5]))
        return predicted_motion

    # center_in_histogram = self._find_nearest_center_in_histogram(current_motion)
    def _find_nearest_center_in_histogram(self, current_motion):
        current_position = [current_motion[1], current_motion[2]]
        distances = np.linalg.norm(self.LT_histogram_map["centers"] - current_position, axis=1)
        nearest_index = np.argmin(distances)
        
        if distances[nearest_index] > self.r_s:
            return None
        
        return nearest_index

    def _sample_motion_from_histogram(self, nearest_index):
        selected_histogram = self.LT_histogram_map["histogram"][:, :, nearest_index]
        
        ###### Get divergence value of that cluster ######
        cluster_num = nearest_index + 1
        div_row = self.histogram_divergence[self.histogram_divergence['cluster_number'] == cluster_num].iloc[0]
        kl_div_value = div_row['kl_divs']
        js_div_value = div_row['js_divs']
        #### here, in 1024 file, kl is from: 0.2149 1.1439 and js is from 0.0552 0.3086
        ###################################################
        
        ########### print for histogram ############
        # print("####### Infor of histogram")
        # print(selected_histogram.shape)
        # with np.printoptions(precision=4, suppress=True):
        #     pprint(selected_histogram[3,:])
        #############################################

        flattened_hist = selected_histogram.flatten()
        
        if np.all(flattened_hist == 0):
            print("-------------------------------------------------------------")
            print("Warning: In this index ", nearest_index, ", all values are 0.")
            sampled_index = np.random.choice(np.arange(flattened_hist.size))
        else:
            normalized_hist = flattened_hist / flattened_hist.sum()
            sampled_index = np.random.choice(a=np.arange(normalized_hist.size), p=normalized_hist)
            
        rho_index, theta_index = np.unravel_index(sampled_index, selected_histogram.shape)

        theta = np.deg2rad((theta_index)*10)
        rho = (rho_index+1)*0.2
        sampled_velocity = np.array([rho, theta])
        new_prob = selected_histogram[rho_index, theta_index]
        
        return sampled_velocity, new_prob, kl_div_value, js_div_value

    def _apply_sampled_motion_to_current_motion(self, sampled_velocity, current_motion, time_index, kl_div_value, js_div_value, if_only_use_sample_velocity=False):
        ## Did not use js_div_value in this version
        current_velocity = current_motion[3:5]
        result_rho = current_velocity[0]
        
        sampled_orientation = utils.wrapTo2pi(sampled_velocity[1])
        current_orientation = utils.wrapTo2pi(current_velocity[1])

        delta_theta = utils.circdiff(sampled_orientation, current_orientation)
        delta_rho = np.abs(sampled_velocity[0] - current_velocity[0])
        
        if if_only_use_sample_velocity:
            param_lambda = 1
        else:
            param_lambda = self._apply_adaptive_kernel(delta_theta, kl_div_value)
            

        result_theta = utils.circmean([sampled_orientation, current_orientation], [param_lambda, 1-param_lambda])

        ########### If use speed from histogram ###########
        # param_lambda_rho = self._apply_adaptive_kernel(delta_rho, kl_div_value)
        # sampled_rho = sampled_velocity[0]
        # current_rho = current_velocity[0]
        # result_rho = (sampled_rho - current_rho) * param_lambda_rho + current_rho
        ###################################################

        predicted_motion = np.concatenate(
            (current_motion[0:3], [result_rho, result_theta])
        )

        return predicted_motion

    def _apply_gaussian_kernel(self, x):
        beta = self.beta

        return np.exp(-beta*x**2)

    def _apply_adaptive_kernel(self, x, kl_div_value):
        beta = np.power(10, kl_div_value)
        
        return np.exp(-beta*x**2)

    def _get_next_position_by_velocity(self, current_position, current_velocity):
        next_position_x = current_position[0] + current_velocity[0] * np.cos(current_velocity[1]) * self.delta_t
        next_position_y = current_position[1] + current_velocity[0] * np.sin(current_velocity[1]) * self.delta_t

        return np.array([next_position_x, next_position_y])
