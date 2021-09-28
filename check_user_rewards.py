
import sys
import numpy as np
import random
import os
from simulation_utils import load_trajectories
import matplotlib.pyplot as plt
import seaborn as sns
from algos import compute_delta
from simulation_utils import get_feedback_no_sim
import pandas as pd
import matplotlib

task = 'driverextended'

matplotlib.rcParams["legend.frameon"] = False
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 22}
matplotlib.rc('font', **font)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['text.usetex'] = True

# Load sampled trajectories
trajectories = load_trajectories(task, 200)
features_set = trajectories['feature_set']
w_set = trajectories['w_set']
print(features_set)
weights= []
rewards = []
align = []
best_sampled_path=[]
for _ in range(1000):
    w = np.random.randn(5)  # option 1 : uniformly random user
    # option 2 : powered user - make large weights larger and small weights smaller
    pow=1
    w = []
    for i in range(10):
        val = np.random.randn()
        val = np.power(val,pow) if val>=0 else -np.power(-val,pow)
        w.append(val)
    w_id = random.choice(range(len(trajectories['w_set'])))
    # w = trajectories['w_set'][w_id]

    w = w / np.linalg.norm(w)
    weights += list(w)
    rewards_w = np.dot(features_set, w.T)
    min_rew = np.min(rewards_w)
    rewards_w -= min_rew
    rewards_w =  list(rewards_w / np.max(rewards_w))
    rewards += rewards_w
    aligns_w = np.dot(w_set,w.T)
    align += list(aligns_w)

    best_path = np.argmax(features_set @ w.T)
    best_sampled_path.append(best_path)
    print('w',list(w))


    print(rewards_w)

# print('len rewards', len(rewards))
# fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 3))
# ax1.hist(weights, bins=100)
# ax1.title.set_text('weights, User Power'+str(pow))
#
# ax2.hist(rewards, bins=100)
# ax2.title.set_text('rewards, User Power'+str(pow))
#
# ax3.hist(best_sampled_path, bins=100)
# ax3.title.set_text('best path id, User Power'+str(pow))
#
# ax4.hist(align, bins=100)
# ax4.title.set_text('align, User Power'+str(pow))


print('plot scale values')
alphas = [1.0, .75, .5, .25]
df = pd.DataFrame()
for alpha in alphas:
    w = trajectories['w_set'][1]
    scale_positions = []
    delta = compute_delta(trajectories, w)
    # alpha = .5
    resolution =.01
    user  = {'w': w, 'alpha': alpha, 'noise_std': .0, 'delta':delta}

    num_traj = len(features_set)
    num_traj = 20
    for i in range(num_traj):
        for j in range(i, num_traj):
            psi = get_feedback_no_sim(resolution, user, features_set[i], features_set[j])
            scale_positions+=[{'alpha':alpha, 'psi':abs(psi)}]
    df = df.append(pd.DataFrame(scale_positions), sort=False)
fig, ax = plt.subplots(1, 1, figsize=(12, 4))

ax = sns.boxplot(x="psi", y="alpha", orient='h', data=df, palette='Set3')
ax = sns.swarmplot(x="psi", y="alpha", orient='h', data=df, color=".2")

# ax = sns.boxplot(y="psi", x="alpha",  data=df, palette='Set3')
# ax = sns.swarmplot(y="psi", x="alpha", data=df, color=".2")

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.set_xlabel("Absolute slider position $|\psi|$", fontsize=28)
ax.set_ylabel("Saturation $\\alpha$", fontsize=28)
ax.tick_params(axis='x', labelsize=24)
ax.tick_params(axis='y', labelsize=24)
# ax.invert_yaxis()
ax.set_ylim([-0.5 , 3.5])
ax.set_xlim([0, 1])
fig.tight_layout()
plt.savefig("Saturation_example_noiseless")
plt.show()

