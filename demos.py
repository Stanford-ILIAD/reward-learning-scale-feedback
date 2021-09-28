import algos
import numpy as np
from simulation_utils import create_env, get_feedback, run_algo, load_trajectories, get_feedback_no_sim
import pandas as pd
import time


def create_validation_data(trajectories, user, validation_resolution=0.1):
    """

    :param trajectories:
    :param user:
    :param validation_resolution:
    :return:
    """
    # trajectories are already randomly generated. So let's use them in a deterministic way to keep results consistent between iterations.
    # we will compute the log-likelihood for queries: trajectory n vs trajectory n+1 for all odd n in the trajectory set.
    PQ = []
    Up = []
    for i in range(0, len(trajectories['feature_set']), 2):
        phi_A = trajectories['feature_set'][i]
        phi_B = trajectories['feature_set'][i + 1]
        up_i = get_feedback_no_sim(validation_resolution, user, phi_A, phi_B)
        PQ.append(phi_A - phi_B)
        Up.append(up_i)
    return np.array(PQ), np.array(Up)


def get_trajectory_for_weight(simulation_object, weight):
    """
    :param weight:
    :return:
    """
    print(simulation_object.name+" - get trajectory for w=", weight)
    controls, features, _ = simulation_object.find_optimal_path(weight)
    weight = list(weight)
    features = list(features)
    return {"w": weight, "phi": features, "controls": controls}

def get_current_error(w_samples, alpha_samples, iter, user, solver, slider_step_size, trajectories, simulation_object,
                      PQ_val, Up_val, validation_resolution=0.1, slider=.0, correct=True):
    """
    Compute performance measures and save them in a dictionary
    :param w_samples: current samples from the weihgt distribution
    :param alpha_samples: current samples for the saturation
    :param iter: iteration
    :param user: the simulated user dict
    :param solver: name of the active query function
    :param slider_step_size:
    :param trajectories: sample trajectories
    :param simulation_object:
    :param PQ_val: validation queries
    :param Up_val: validation responses
    :param validation_resolution: slider_step_size for the validation set
    :param slider: slider value of the last feedback
    :param correct: was the last feedback 'correct', i.e., did the user choose the path with higher reward
    :return:
    """
    w_exp = np.mean(w_samples, axis=0)
    w_exp = np.divide(w_exp, np.linalg.norm(w_exp))
    w_user = np.divide(user['w'], np.linalg.norm(user['w']))
    features_set = trajectories['feature_set']
    min_reward = np.dot(features_set[np.argmin(features_set @ w_user.T)], w_user)
    phi_user = features_set[np.argmax(features_set @ w_user.T)]
    phi_exp = features_set[np.argmax(features_set @ w_exp.T)]
    relative_reward = min((np.dot(phi_exp, w_user) - min_reward) / (np.dot(phi_user, w_user) - min_reward), 1)
    alignment = np.dot(w_exp, w_user)

    # compute log-likelihood
    loglikelihood = 0
    for i in range(len(w_samples)):
        lprob, delta = algos.logprob(validation_resolution, PQ_val, Up_val, user['noise_std'], alpha_samples[i],
                                     w_samples[i], trajectories, test_phase=True)
        loglikelihood += lprob / len(Up_val)
    loglikelihood /= len(w_samples)
    return [{'iter': iter,
             'alignment': alignment,
             'reward': relative_reward,
             'loglikelihood': loglikelihood,
             'w_est': w_exp,
             'w_opt': w_user,
             'solver': str(solver) + '_res:' + str(slider_step_size),
             '$alpha$': str(user['alpha']),
             'sigma': str(user['noise_std']),
             'slider': float(abs(slider)),
             'correct': float(correct)
             }]


def save_erros(data_frame, identifier):
    """
    save list of error dict in file.
    :param data_frame:
    :param identifier:
    :return:
    """
    import os, errno
    print("save error log")
    filename = '/' + identifier + '.csv'
    try:
        os.makedirs("simulation_data/")
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    data_frame.to_csv("simulation_data/" + filename)


def run(task, trajectories, resolutions, generated_users, acquisitions):
    """

    :param task:
    :param resolutions:
    :param user:
    :param acquisition:
    :return:
    """

    simulation_object = create_env(task)
    num_samples = 200
    identifier = task + str(round(time.time() * 100))  # id for the trial used as the filename
    df = pd.DataFrame()
    for user_id in range(len(generated_users)):  # repeat experiment for different users
        user = generated_users[user_id]
        validation_resolution = 0.1  # for consistency we should use a single value regardless of what the resolution of the user's slider is
        PQ_val, Up_val = create_validation_data(trajectories, user, validation_resolution)

        for acquisition in acquisitions:
            for resolution in resolutions:

                print('trial', user_id + 1, 'aquisition', acquisition, 'res', resolution,
                      ' user:', user)
                d = simulation_object.num_of_features
                i = 0
                PQ = []
                Up = []
                w_samples = np.random.randn(num_samples, d)
                w_samples = w_samples / np.linalg.norm(w_samples, axis=1).reshape(-1, 1)
                alpha_samples = np.random.rand(num_samples)
                delta_samples = algos.compute_delta(trajectories,
                                                    w_samples.T)  # simple trick to get delta's of multiple w's
                sample_logprobs = np.zeros(num_samples)  # because there is no initial data

                error_hist = get_current_error(w_samples, alpha_samples, 0, user, acquisition, resolution, trajectories,
                                               simulation_object, PQ_val, Up_val,
                                               validation_resolution)  # error logging
                while i < 20:  # number of learning iterations
                    input_id1, input_id2, score = run_algo(acquisition, simulation_object, trajectories, resolution,
                                                           user['noise_std'], w_samples, alpha_samples, delta_samples,
                                                           sample_logprobs, PQ, Up)
                    phi_A, phi_B = trajectories['feature_set'][input_id1], trajectories['feature_set'][input_id2]
                    up_i = get_feedback_no_sim(resolution, user, phi_A, phi_B)
                    reward_diff = np.dot(phi_A - phi_B, user['w'])
                    correct = reward_diff >= 0 and up_i <= 0 or reward_diff <= 0 and up_i >= 0 or reward_diff == 0 and up_i == 0
                    PQ.append(phi_A - phi_B)
                    Up.append(up_i)
                    print('iteration', i+1, 'user scale feedback', up_i)
                    i += 1
                    w_samples, alpha_samples, delta_samples, sample_logprobs \
                        = algos.estimate_w_and_delta(trajectories, resolution, PQ, Up, user['noise_std'], w_samples[-1],
                                                     alpha_samples[-1], num_samples=num_samples)
                    error_hist += get_current_error(w_samples, alpha_samples, i, user, acquisition, resolution,
                                                    trajectories,
                                                    simulation_object, PQ_val, Up_val, validation_resolution,
                                                    slider=up_i, correct=correct)  # error logging

                for elem in error_hist:
                    df = df.append(pd.DataFrame([elem]), sort=True)
                save_erros(df, identifier)
