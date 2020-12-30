
import random 
import numpy as np
from scipy.stats import beta

current_action = None
total_reward = 0

max_std = 3
min_std = 0

decay = None
num_bandits = None
num_steps = None

post_a, post_b, pulls = [None] * 3

def random_until_failure(observation, configuration):
    # observation values 
    ## remainingOverageTime - 60s of overage time is allotted. Any time spent over 0.25s on an individual step is removed from remainingOverageTime
    ## step - We do 2000 trials, this tells us where within those trials we are
    ## reward - Total reward so far (sum of previous pull results)
    ## lastActions - Which machine was pulled by each agent in the last round [us, them]
    
    global current_action, total_reward, max_std, min_std, decay, num_bandits, num_steps, post_a, post_b, pulls
    
    if observation.step == 0:
        decay = configuration.decayRate
        num_bandits = configuration.banditCount
        num_steps = configuration.episodeSteps
        
        post_a, post_b = np.ones((2, num_bandits))
        pulls = np.zeros(num_bandits)

    else:
        last_reward = observation['reward'] - total_reward
        total_reward = observation['reward']
        
        # Update Gaussian posterior
        post_a[current_action] += last_reward
        post_b[current_action] += 1 - last_reward
        
        for action in observation.lastActions:
            post_a[action] = (post_a[action] - 1) * decay + 1
        
    if observation.step < 200:
        current_action = random.randrange(num_bandits)
    
    elif last_reward == 0:
        num_stds = (max_std - min_std) /  (num_steps - observation.step)
        rands = beta.mean(post_a, post_b) + (num_stds * beta.std(post_a, post_b))
        max_index = np.argmax(rands)
        current_action = int(max_index)

    return current_action
