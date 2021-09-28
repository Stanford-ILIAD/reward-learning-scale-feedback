# There is a memory leak issue when we loop over trajectories.
# So let's go with repeating python commands instead. In the command below, D is the total number of trajectories we want to generate:
# Windows: FOR /L %i IN (0,1,D-1) DO python input_sampler.py [task_name] %i

from simulation_utils import create_env
import os
from demos import *
from config import CFG

num_samples = 2000  # need to generate more samples than will be used in the end since for some weights the planner returns the same trajectory
feature_set = []
for i in range(0, num_samples):
    print('precompute trajectory', i + 1, '/', num_samples)
    print('total number of non-redundant trajectories', len(feature_set))
    task = CFG['task']
    idx = i

    simulation_object = create_env(task)
    z = simulation_object.feed_size
    lower_input_bound = [x[0] for x in simulation_object.feed_bounds]
    upper_input_bound = [x[1] for x in simulation_object.feed_bounds]
    input = np.random.uniform(low=lower_input_bound, high=upper_input_bound, size=(z,))
    simulation_object = create_env(task)
    simulation_object.feed(list(input))

    if not os.path.isdir('ctrl_samples'):
        os.mkdir('ctrl_samples')
    if not os.path.isdir('videos'):
        os.mkdir('videos')

    d = simulation_object.num_of_features
    if idx > 0:
        data = np.load('ctrl_samples/' + task + '.npz')
        feature_set = data['feature_set']
        input_set = data['input_set']
        w_set = data['w_set']
    else:
        feature_set = np.zeros((0, d))
        w_set = np.zeros((0, d))
        input_set = np.zeros((0, z))

    w = np.random.randn(10)
    w = w / np.linalg.norm(w)
    user_traj = get_trajectory_for_weight(simulation_object, w)
    features = user_traj['phi']
    print("features", features)
    control_input = user_traj['controls']
    feature_set = np.vstack((feature_set, features))
    w_set = np.vstack((w_set, w))
    input_set = np.vstack((input_set, control_input))

    # filter suboptimal trajectories
    best_trajectories = np.argmax(feature_set @ w_set.T, axis=0)
    print('best traj', len(best_trajectories), best_trajectories)
    best_trajectories = list(set(best_trajectories))
    print('best traj - filtered', len(best_trajectories), best_trajectories)
    feature_set = feature_set[best_trajectories]
    input_set = input_set[best_trajectories]
    w_set = w_set[best_trajectories]

    np.savez('ctrl_samples/' + simulation_object.name + '.npz', feature_set=feature_set, input_set=input_set,
             w_set=w_set)
    print('Trajectory ' + str(idx) + ' has been saved!')
    print('total number of non-redundant trajectories', len(feature_set))
