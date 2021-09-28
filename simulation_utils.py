import numpy as np
import scipy.optimize as opt
import algos
from models import Driver, DriverExtended, Fetch
import warnings


def get_feedback(simulation_object, resolution, user, input_A, input_B):
    simulation_object.feed(input_A)
    phi_A = np.array(simulation_object.get_features())
    simulation_object.feed(input_B)
    phi_B = np.array(simulation_object.get_features())
    pq = - phi_A + phi_B
    u = None
    while u == None:
        if not (user is None):
            thres = user['delta']*user['alpha']
            psi = np.clip(np.dot(user['w'], pq) / thres, -1, 1)
            u = np.clip(psi + np.random.randn() * user['noise_std'], -1, 1)
            break
        selection = input('A/B to watch, [-1,1] to vote: ').lower()
        if selection == 'a':
            simulation_object.feed(input_A)
            simulation_object.watch(1)
        elif selection == 'b':
            simulation_object.feed(input_B)
            simulation_object.watch(1)
        else:
            try:
                u = float(selection)
                if not -1 <= u <= 1:
                    u = None
            except ValueError:
                continue
                
    if np.isclose(resolution, 2):
        up = np.sign(u)
    else:
        up = np.round(u / resolution) * resolution
        up = np.round(up, 5)
    return phi_A, phi_B, up
    
def get_feedback_no_sim(resolution, user, phi_A, phi_B):
    pq = - phi_A + phi_B
    u = None
    while u == None:
        if not (user is None):
            thres = user['delta']*user['alpha']
            psi = np.clip(np.dot(user['w'], pq) / thres, -1, 1)
            u = np.clip(psi + np.random.randn() * user['noise_std'], -1, 1)
            break
        print('A: ' + str(phi_A))
        print('B: ' + str(phi_B))
        selection = input('[-1,1] to vote: ').lower()
        try:
            u = float(selection)
            if not -1 <= u <= 1:
                u = None
        except ValueError:
            continue
            
    if np.isclose(resolution, 2):
        up = np.sign(u)
    else:
        up = np.round(u / resolution) * resolution
        up = np.round(up, 5)
    return up

def load_trajectories(task, num_trajectories):
    """
    load pre sampled trajectories from file
    :param task:
    :param num_trajectories:
    :return:
    """
    A = np.load('ctrl_samples/' + task + '.npz')
    if A['input_set'].shape[0] < num_trajectories:
        warnings.warn(str(num_trajectories) + ' trajectories were requested, but the dataset contains only ' + str(A['input_set'].shape[0]) + ' trajectories. Returning the dataset.')
        return A
    B = {}
    B['input_set'] = A['input_set'][:num_trajectories]
    B['feature_set'] = A['feature_set'][:num_trajectories]
    B['w_set'] = A['w_set'][:num_trajectories]

    return B
    

def create_env(task):
    if task == 'lds':
        return LDS()
    elif task == 'driver':
        return Driver()
    elif task == 'driverextended':
        return DriverExtended()
    elif task == 'tosser':
        return Tosser()
    elif task == 'fetch':
        return Fetch()
    else:
        print('There is no task called ' + task)
        exit(0)


def run_algo(acquisition, simulation_object, trajectories, resolution, noise_std, w_samples, alpha_samples, delta_samples, sample_logprobs, PQ, Up):
    if acquisition == 'random':
        return algos.random(trajectories)
    elif acquisition == 'information':
        return algos.infogain(simulation_object, trajectories, resolution, noise_std, w_samples, alpha_samples, delta_samples)
    elif acquisition == 'regret':
        return algos.maxregret(simulation_object, trajectories, resolution, noise_std, w_samples, alpha_samples, delta_samples, sample_logprobs, PQ, Up)
    else:
        assert False, 'There is no acquisition called ' + acquisition


def func(ctrl_array, *args):
    simulation_object = args[0]
    w = np.array(args[1])
    simulation_object.set_ctrl(ctrl_array)
    features = simulation_object.get_features()
    return -np.mean(np.array(features).dot(w))
    
def best_id_out_of_dataset(trajectories, w):
    return np.argmax(trajectories['feature_set'] @ w, axis=0)

def compute_best(simulation_object, w, iter_count=10):
    u = simulation_object.ctrl_size
    lower_ctrl_bound = [x[0] for x in simulation_object.ctrl_bounds]
    upper_ctrl_bound = [x[1] for x in simulation_object.ctrl_bounds]
    opt_val = np.inf
    for _ in range(iter_count):
        temp_res = opt.fmin_l_bfgs_b(func, x0=np.random.uniform(low=lower_ctrl_bound, high=upper_ctrl_bound, size=(u)), args=(simulation_object, w), bounds=simulation_object.ctrl_bounds, approx_grad=True)
        if temp_res[1] < opt_val:
            optimal_ctrl = temp_res[0]
            opt_val = temp_res[1]
    print(-opt_val)
    return optimal_ctrl

def play(simulation_object, optimal_ctrl):
    simulation_object.set_ctrl(optimal_ctrl)
    keep_playing = 'y'
    while keep_playing == 'y':
        keep_playing = 'u'
        simulation_object.watch(1)
        while keep_playing != 'n' and keep_playing != 'y':
            keep_playing = input('Again? [y/n]: ').lower()
    return optimal_ctrl
