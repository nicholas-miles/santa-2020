
import random 
import numpy as np
from scipy.stats import beta

current_action = None
total_reward = 0

est_ucb_percentile = 0.5

decay = None
num_bandits = None
num_steps = None

post_a, post_b, opponent_pulls, my_pulls = [None] * 4
opponent_log = []

explore_steps = 200
explore_max_pct = 1
explore_min_pct = 0.1


def bandit_quality():
    rands = beta.ppf(est_ucb_percentile, post_a, post_b)
    
    return rands

def opponent_quality():
    global opponent_pulls
    ucbs = bandit_quality()
    opp_quality = np.corrcoef(opponent_pulls[(opponent_pulls > 1) & (my_pulls > 1)], ucbs[(opponent_pulls > 1) & (my_pulls > 1)])[0,1]
    
    return max(0.25, opp_quality)

def opponent_consistency():
    global opponent_log
    n_lookback = 8
    
    opp_consistency = 1 - ((len(set(opponent_log[-n_lookback:])) - 1) / n_lookback)
    
    return opp_consistency

def best_bandit():
    trust = opponent_quality()
    
    bandits = bandit_quality()
    sorted_bandits = np.sort(bandits)
    opp_prob = beta.ppf(est_ucb_percentile, (opponent_pulls - 1).clip(min=1), 1)
    
    combined_prob = ((bandits + (trust * opp_prob)) / (1 + trust)) * (decay ** (opponent_pulls + my_pulls))
    for c in range(len(combined_prob)):
        if my_pulls[c] > 5 or opponent_pulls[c] < 1:
            combined_prob[c] = bandits[c] * (decay ** (opponent_pulls[c] + my_pulls[c]))
    max_index = np.argmax(combined_prob)
    
    return max_index

def ucb_with_opponent_successes(observation, configuration):
    # observation values 
    ## remainingOverageTime - 60s of overage time is allotted. Any time spent over 0.25s on an individual step is removed from remainingOverageTime
    ## step - We do 2000 trials, this tells us where within those trials we are
    ## reward - Total reward so far (sum of previous pull results)
    ## lastActions - Which machine was pulled by each agent in the last round [us, them]
    
    global current_action, total_reward, max_std, min_std, decay, num_bandits, num_steps, post_a, post_b, opponent_pulls, my_pulls, explore_steps, explore_max_pct, explore_min_pct
    
    my_index = observation.agentIndex
    their_index = 1 - observation.agentIndex
    
    if observation.step == 0:
        decay = configuration.decayRate
        num_bandits = configuration.banditCount
        num_steps = configuration.episodeSteps
        
        post_a, post_b = np.ones((2, num_bandits))
        opponent_pulls = np.zeros(num_bandits)
        my_pulls = np.zeros(num_bandits)
        
        current_action = random.randrange(num_bandits)

    else:
        last_reward = observation['reward'] - total_reward
        total_reward = observation['reward']
        
        post_a[current_action] += last_reward
        post_b[current_action] += 1 - last_reward
        
        my_pulls[observation.lastActions[my_index]] += 1
        opponent_pulls[observation.lastActions[their_index]] += 1
        opponent_log.append(observation.lastActions[their_index])
    
    if observation.step > 0 and last_reward == 0:
        max_index = best_bandit()
        current_action = int(max_index)

    return current_action
