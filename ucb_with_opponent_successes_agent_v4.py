
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
explore_rate = 0.3


def bandit_quality(method='mean'):
    """
    Estimates the quality of each bandit using a variety of methods (UCB, mean or sample mean)
    In short, what is the nth percentile of the beta distribution over my belief of the quality of each vending machine.
    """
    if method == 'ucb':
        rands = beta.ppf(est_ucb_percentile, post_a, post_b)
    elif method == 'mean':
        rands = beta.mean(post_a, post_b)
    elif method == 'sample_mean':
        rands = beta.rvs(post_a, post_b, shape=(3, num_bandits)).mean()
    
    return rands

def opponent_quality():
    """
    Estimates the quality of my opponent. Currently uses the correlation between their pulls and my beliefs, excluding
    any bandits that either they or I did not pull yet.
    """
    global opponent_pulls
    ucbs = bandit_quality()
    only_pulled_ucbs = ucbs[(opponent_pulls > 1) & (my_pulls > 1)]
    only_pulled_opponent_data = opponent_pulls[(opponent_pulls > 1) & (my_pulls > 1)]
    
    if len(only_pulled_ucbs) > 5 and len(only_pulled_opponent_data) > 5:
        opp_quality = np.corrcoef(only_pulled_opponent_data, only_pulled_ucbs)[0,1]
    else:
        opp_quality = 0.25
    
    return opp_quality

def opponent_consistency():
    """
    Alternative to quality. Is my opponent repeatedly pulling the same machines or not
    """
    global opponent_log
    n_lookback = 8
    
    opp_consistency = 1 - ((len(set(opponent_log[-n_lookback:])) - 1) / n_lookback)
    
    return opp_consistency

def best_bandit():
    """
    Combines my knowledge about the quality of each bandit with information about which bandits my opponent is pulling.
    """
    # Retrieve my belief in the opponent's skill and my prior about each bandit's quality
    bandits = bandit_quality()
    trust = opponent_quality()
    
    # I assume that every pull after the first one is a success for my opponent
    opp_prob = beta.ppf(est_ucb_percentile, (opponent_pulls - 1).clip(min=1), 1)
    
    # Combine my beliefs with my opponents action. Weighted average using trust, decayed by the combined number of pulls
    combined_prob = ((bandits + (trust * opp_prob)) / (1 + trust)) * (decay ** (opponent_pulls + my_pulls))
    
    # If I have pulled a bandit often, or they have never pulled it, ignore their actions
    for c in range(len(combined_prob)):
        if my_pulls[c] > 5 or opponent_pulls[c] < 1:
            combined_prob[c] = bandits[c] * (decay ** (opponent_pulls[c] + my_pulls[c]))
            
    # Find the bandit with maximal probability of success
    max_index = np.argmax(combined_prob)
    
    return max_index

def ucb_with_opponent_successes(observation, configuration):
    # observation
    ## remainingOverageTime - 60s of overage time is allotted. Any time spent over 0.25s on an individual step is removed from remainingOverageTime
    ## step - We do 2000 trials, this tells us where within those trials we are
    ## reward - Total reward so far (sum of previous pull results)
    ## lastActions - Which machine was pulled by each agent in the last round [us, them]
    
    global current_action, total_reward, max_std, min_std, decay, num_bandits, num_steps, post_a, post_b, opponent_pulls, my_pulls, explore_steps, explore_max_pct, explore_min_pct
    
    # Figure out who I am and who they are
    my_index = observation.agentIndex
    their_index = 1 - observation.agentIndex
    
    # For the initial stage, set my global variables and pull a random bandit
    if observation.step == 0:
        # Initialization
        decay = configuration.decayRate
        num_bandits = configuration.banditCount
        num_steps = configuration.episodeSteps
        post_a, post_b = np.ones((2, num_bandits))
        opponent_pulls = np.zeros(num_bandits)
        my_pulls = np.zeros(num_bandits)
        
        # Select a random bandit
        current_action = random.randrange(num_bandits)

    else:
        # Update my priors and knowledge about my opponent's actions
        last_reward = observation['reward'] - total_reward
        total_reward = observation['reward']
        post_a[current_action] += last_reward
        post_b[current_action] += 1 - last_reward
        my_pulls[observation.lastActions[my_index]] += 1
        opponent_pulls[observation.lastActions[their_index]] += 1
        opponent_log.append(observation.lastActions[their_index])
    
        # If my last pull wasn't a success, figure out what my highest probability of success is.
        # Otherwise, keep pulling the same one!
        if last_reward == 0:
            if observation.step < explore_steps and random.random() < explore_rate:
                current_action = random.randrange(num_bandits)
            else:
                max_index = best_bandit()
                current_action = int(max_index)

    return current_action
