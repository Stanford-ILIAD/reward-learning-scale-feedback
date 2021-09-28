import numpy as np
import random
import scipy.optimize as opt
import scipy.stats as ss
import cvxpy as cp

def random(trajectories): # just in case we want random queries to come from the same query database
    input_set = trajectories['input_set']
    # print('numsamples', len(input_set))
    id_input1, id_input2 = np.random.choice(input_set.shape[0], size=(2,), replace=False)
    return id_input1, id_input2, 0

def infogain(simulation_object, trajectories, resolution, noise_std, w_samples, alpha_samples, delta_samples):
    z = simulation_object.feed_size
    feature_set = trajectories['feature_set']
    w_samples = np.array(w_samples)
    alpha_samples = np.array(alpha_samples)
    M = w_samples.shape[0]
    num_traj = feature_set.shape[0]
    rew_set = feature_set @ w_samples.T
    rew_diff_tensor = np.zeros((num_traj, num_traj, M))
    for i in range(rew_set.shape[0]):
        rew_diff_tensor[i] = rew_set[i] - rew_set

    psi_set = np.clip(rew_diff_tensor / (alpha_samples * delta_samples), -1., 1.)
    P = np.zeros((num_traj, num_traj, M, int(2.0/resolution + 1)))
    P[:,:,:,0] = ss.norm.cdf(-1 - psi_set, scale=noise_std)
    P[:,:,:,-1] = ss.norm.cdf(psi_set - 1, scale=noise_std)
    idx = 1
    for u in np.arange(-1 + resolution, 1 - resolution / 2., resolution):
        P[:,:,:,idx] = ss.norm.cdf(u - psi_set + resolution/2., scale=noise_std) - ss.norm.cdf(u - psi_set - resolution/2., scale=noise_std)
        idx += 1
    denom = P.sum(axis=2).reshape(P.shape[0],P.shape[1],1,-1)
    obj = np.nansum(P * np.log2(M * P / denom), axis=(2,3)) / M
    id_input1, id_input2 = np.unravel_index(obj.argmax(), [num_traj,num_traj])
    return id_input1, id_input2, obj[id_input1, id_input2]

# def maxregret2(simulation_object, trajectories, resolution, noise_std, w_samples, alpha_samples, delta_samples, sample_logprobs, PQ, Up):
#     """
#     modified implementation by Nils. the sampled trajectories are (approx.) optimal for some weight. weights saved with
#     trajectories
#     :param simulation_object:
#     :param trajectories:
#     :param resolution:
#     :param noise_std:
#     :param w_samples:
#     :param alpha_samples:
#     :param delta_samples:
#     :param sample_logprobs:
#     :param PQ:
#     :param Up:
#     :return:
#     """
#     print('max reg query')
#     z = simulation_object.feed_size
#     features_set = trajectories['feature_set']
#     weights = np.array(trajectories['w_set'])
#     max_regret = -np.Inf
#     best_pair = [-1, -1]
#     best_alpha=-1
#     alphas = np.linspace(0.1,1,10)
#     for alpha1 in alphas:
#         for alpha2 in alphas:
#             # print(alpha)
#             if len(PQ) >0:
#                 PQ = np.array(PQ)
#                 Up = np.array(Up)
#                 # alphas = np.random.random(len(weights))
#                 logpropbs1 = [logprob(resolution, PQ, Up, noise_std, alpha1, weights[idx], trajectories)[0]for idx in range(len(weights))]
#                 logpropbs2 = [logprob(resolution, PQ, Up, noise_std, alpha2, weights[idx], trajectories)[0]for idx in range(len(weights))]
#             else:
#                 logpropbs1=[0]*len(weights)
#                 logpropbs2=[0]*len(weights)
#
#             for w1_id in range(weights.shape[0]):
#                 for w2_id in range(w1_id+1, weights.shape[0]):
#                     logp1 = logpropbs1[w1_id]
#                     logp2 = logpropbs2[w2_id]
#                     features1 = features_set[w1_id]
#                     features2 = features_set[w2_id]
#                     regret1 = np.dot(features1, weights[w1_id]) - np.dot(features2, weights[w1_id])
#                     regret2 = np.dot(features2, weights[w2_id]) - np.dot(features1, weights[w2_id])
#                     obj = np.exp(logp1 + logp2) * (regret1 + regret2)
#                     if obj > max_regret:
#                         max_regret = obj
#                         best_pair = [w1_id, w2_id]
#                         best_alpha = (alpha1,alpha2)
#     print('max dist regret', obj, 'alpha', best_alpha)
#     id_input1 = best_pair[0]
#     id_input2 = best_pair[1]
#     return id_input1, id_input2, max_regret
#
# def compute_posterior_alt(w, delta, PQ, Up):
#     alpha_posts= []
#     for alpha in np.linspace(0,1,11):
#         prob = 1
#         for idx in range(len(PQ)):
#             lhs = np.dot(w,PQ[idx])
#             rhs = Up[idx]*alpha*delta
#             if Up[idx] == -1:
#                 likelihood = .8 if lhs <= rhs else .2
#             elif Up[idx] == 1:
#                 likelihood = .8 if lhs >= rhs else .2
#             else:
#                 print("choicy choice", abs(lhs - rhs))
#                 likelihood = .8 if abs(lhs - rhs) <= 0.1 else .2
#             # print('feedback', Up[idx], lhs, rhs, likelihood)
#             prob *= likelihood
#         # print('alpha',alpha,'prob', prob)
#         alpha_posts.append(prob)
#     return np.mean(alpha_posts)

def maxregret(simulation_object, trajectories, resolution, noise_std, w_samples, alpha_samples, delta_samples, sample_logprobs, PQ, Up):
    """
    Implementation by Erdem for random sampled trajectories
    :param simulation_object:
    :param trajectories:
    :param resolution:
    :param noise_std:
    :param w_samples:
    :param alpha_samples:
    :param delta_samples:
    :param sample_logprobs:
    :param PQ:
    :param Up:
    :return:
    """
    z = simulation_object.feed_size
    features_set = trajectories['feature_set']
    best_trajectories = np.argmax(features_set @ w_samples.T, axis=0)
    max_regret = -np.Inf
    best_pair = [-1,-1]
    for w1_id in range(w_samples.shape[0]):
        for w2_id in range(w1_id+1, w_samples.shape[0]):
            logp1 = sample_logprobs[w1_id]
            logp2 = sample_logprobs[w2_id]
            features1 = features_set[best_trajectories[w1_id]]
            features2 = features_set[best_trajectories[w2_id]]
            regret1 = np.dot(features1, w_samples[w1_id]) - np.dot(features1, w_samples[w2_id])
            #regret1 = np.dot(features1, w_samples[w1_id]) - np.dot(features2, w_samples[w1_id])
            regret2 = np.dot(features2, w_samples[w2_id]) - np.dot(features2, w_samples[w1_id])
            #regret2 = np.dot(features2, w_samples[w2_id]) - np.dot(features1, w_samples[w2_id])
            obj = np.exp(logp1 + logp2) * (regret1 + regret2)
            if obj > max_regret:
                max_regret = obj
                best_pair = [w1_id, w2_id]
    id_input1 = best_trajectories[best_pair[0]]
    id_input2 = best_trajectories[best_pair[1]]
    return id_input1, id_input2, max_regret

def compute_delta(trajectories, w):
    rew_set = np.matmul(trajectories['feature_set'], w)
    return np.max(rew_set, axis=0) - np.min(rew_set, axis=0)

def compute_max_delta(trajectories):
    # NOTE: This is for \norm{w} <= 1
    norms = np.linalg.norm(trajectories['psi_set'], axis=1)
    return np.max(norms)

def logprob(resolution, PQ, Up, noise_std, alpha, w, trajectories, test_phase=False):
    delta = compute_delta(trajectories, w)
    if (not test_phase) and (alpha < 0. or alpha > 1. or np.linalg.norm(w) > 1): 
        return -np.inf, delta
    Psi = np.clip(- PQ @ w / (delta * alpha), -1., 1.)
    strict_mask = np.isclose(np.abs(Up), 1)
    Up_strict = Up[strict_mask]
    Psi_strict = Psi[strict_mask]
    Up_weak = Up[np.logical_not(strict_mask)]
    Psi_weak = Psi[np.logical_not(strict_mask)]
    
    # numerical trick to handle numerical problems (i.e., with direct computation, cdf values might be too small)
    # instead of log(|a-b|), we compute log(|exp(c + log(a)) - exp(c + log(b))|) - c. The constant c increases robustness against the numerical issue.
    loga = ss.norm.logcdf(-np.abs(Up_weak - Psi_weak) + resolution/2., scale=noise_std)
    logb = ss.norm.logcdf(-np.abs(Up_weak - Psi_weak) - resolution/2., scale=noise_std)
    c = -(loga + logb) / 2
    logcdf_weak = np.log(np.abs(np.exp(c + loga) - np.exp(c + logb))) - c
    lprob = np.sum(logcdf_weak) + np.sum(ss.norm.logcdf(-np.abs(Up_strict - Psi_strict) + resolution/2., scale=noise_std))
    #if np.abs(lprob + 0.075798) < 0.001:
    #    import pdb; pdb.set_trace()
    return lprob, delta
    
def sample(trajectories, resolution, PQ, Up, noise_std, initial_w_sample, initial_alpha_sample, num_samples=200, burnin=2000, thin=100, step_size=0.01):
    burnin = 1000
    thin = 50
    step_size = 0.05

    orig_step_size = step_size
    PQ = np.array(PQ)
    Up = np.array(Up)
    if len(PQ.shape) == 1: # the user has responded to only one question
        d = len(PQ)
        PQ = np.reshape(PQ, [1,d])
    else:
        d = PQ.shape[1]
        
    curr_w = initial_w_sample
    curr_alpha = initial_alpha_sample
    curr_logprob, curr_delta = logprob(resolution, PQ, Up, noise_std, curr_alpha, curr_w, trajectories)
    w_samples = [curr_w]
    alpha_samples = [curr_alpha]
    delta_samples = [curr_delta]
    sample_logprobs = [curr_logprob]
    i = 0
    while i < num_samples * thin + burnin:
        next_w = curr_w.copy()
        next_w += np.random.randn(d) * step_size
        next_alpha = curr_alpha + (np.random.rand()-0.5) * step_size
        next_logprob, next_delta = logprob(resolution, PQ, Up, noise_std, next_alpha, next_w, trajectories)
        if i==0 and np.isinf(next_logprob):
            step_size *= 2
            step_size = np.minimum(step_size, 0.2)
            continue
        elif i==0:
            step_size = orig_step_size
        if np.log(np.random.rand()) < next_logprob - curr_logprob:
            curr_w = next_w.copy()
            curr_alpha = next_alpha
            curr_delta = next_delta
            curr_logprob = next_logprob
        w_samples.append(curr_w.copy())
        alpha_samples.append(curr_alpha)
        delta_samples.append(curr_delta)
        sample_logprobs.append(curr_logprob)
        i += 1
    w_samples = np.array(w_samples[burnin+1::thin])
    alpha_samples = np.array(alpha_samples[burnin+1::thin])
    delta_samples = np.array(delta_samples[burnin+1::thin])
    sample_logprobs = np.array(sample_logprobs[burnin+1::thin]) 
    return w_samples, alpha_samples, delta_samples, sample_logprobs


def estimate_w_and_delta(trajectories, resolution, PQ, Up, noise_std, initial_w_sample, initial_alpha_sample,num_samples):
    PQ = np.array(PQ)
    Up = np.array(Up)
    if len(PQ.shape) == 1: # the user has responded to only one question
        d = len(PQ)
        PQ = np.reshape(PQ, [1,d])
    else:
        d = PQ.shape[1]
        
    if np.isclose(noise_std, 0):
        psi_pos = Up[Up >= 0]
        psi_neg = Up[Up < 0]
        PQ_pos = PQ[Up >= 0]
        PQ_neg = PQ[Up < 0]


        w = cp.Variable(d)
        delta_times_alpha = cp.Variable(1)
        print(PQ_pos , psi_pos, delta_times_alpha)
        if len(PQ) == 0:
            prob = cp.Problem(cp.Minimize(delta_times_alpha),
                              [
                               w[0] == 1.0, w <= 100, delta_times_alpha >= 0])
        elif len(PQ_pos) == 0:
            -PQ_neg @ w - psi_neg * delta_times_alpha <= 0,
            w[0] == 1.0, w <= 100, delta_times_alpha >= 0
            prob = cp.Problem(cp.Minimize(delta_times_alpha),
                                [-PQ_neg @ w - psi_neg*delta_times_alpha <= 0,
                                 w[0] == 1.0, w <= 100, delta_times_alpha >= 0])
        elif len(PQ_neg) == 0:
            prob = cp.Problem(cp.Minimize(delta_times_alpha),
                              [-PQ_pos @ w - psi_pos * delta_times_alpha >= 0,
                               w[0] == 1.0, w <= 100, delta_times_alpha >= 0])
        else:
            prob = cp.Problem(cp.Minimize(delta_times_alpha),
                              [-PQ_pos @ w - psi_pos * delta_times_alpha >= 0,
                               -PQ_neg @ w - psi_neg * delta_times_alpha <= 0,
                               w[0] == 1.0, w <= 100, delta_times_alpha >= 0])
        prob.solve()
        return w.value, delta_times_alpha.value
        
    else:
        w_samples, alpha_samples, delta_samples, sample_logprobs = sample(trajectories, resolution, PQ, Up, noise_std, initial_w_sample, initial_alpha_sample, num_samples=num_samples)
        return w_samples, alpha_samples, delta_samples, sample_logprobs