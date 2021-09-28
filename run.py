import demos
import numpy as np
from simulation_utils import load_trajectories, create_env
from algos import compute_delta
from config import CFG

# Load sampled trajectories
trajectories = load_trajectories(CFG['task'], 200)
simulation_object = create_env(CFG['task'])

# Generate different simulated users
generated_users = []
weights= []
for _ in range(5):
    w = np.random.randn(simulation_object.num_of_features)  # draw : uniformly random user
    w = w / np.linalg.norm(w)
    weights.append(w)
    delta = compute_delta(trajectories, w)
    for sigma in CFG['sigma_values']:
        for alpha in CFG['alpha_values']:
            user = {'w': w, 'alpha': alpha, 'noise_std': sigma, 'delta':delta}
            generated_users.append(user)

slider_step_size = CFG['slider_step_size']
demos.run(CFG['task'], trajectories, slider_step_size, generated_users, CFG['acquisitions'])
