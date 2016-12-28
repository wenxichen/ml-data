#!/usr/bin/env python

'''
Created by Wenxi Chen.
Implementation of the simple bandit algorithm on page 33 of 
"Reinforcement Learning: An Introduction" (Richard S. Sutton and Andrew G. Barto).
The code also compare the epsilon greedy with greedy with optimistic initial values.
Simple bandit is a stationary problem, in this case with 10 actions.
'''

import numpy as np
import matplotlib.pyplot as plt

# initialize the expected reward given actions
q_stars = np.zeros((10))
for i in xrange(10):
    q_stars[i] = np.random.normal()

def run_bandit(epsilon, step):
    ''' run the simple bandit algorithm '''
    
    Steps = range(step)
    Average_reward = []
    
    for i in Steps:
        random_selection = np.random.choice(2, size=None, replace=True, p=[1-epsilon, epsilon])
        
        if random_selection:
            action = np.random.choice(10)
        else:
            action = np.argmax(Q_a)
        reward = np.random.normal(q_stars[action])
        
        N_a[action] += 1
        Q_a[action] += (1.0/N_a[action]) * (reward - Q_a[action])
        
        optimal_action = np.argmax(Q_a)
        optimal_expected_reward = Q_a[optimal_action]
        Average_reward.append(optimal_expected_reward)
    
    return Steps, Average_reward

# run greedy epsilon, epsilon = 0.01
Q_a = np.zeros((10))
N_a = np.zeros((10))
Steps, Average_reward = run_bandit(0.01, 2000)

# run greedy epsilon, epsilon = 0.1
Q_a = np.zeros((10))
N_a = np.zeros((10))
Steps1, Average_reward1 = run_bandit(0.1, 2000)

# run greedy using optimistic initial values, epsilon = 0
Q_a = np.ones((10)) * 5
N_a = np.zeros((10))
Steps_oi, Average_reward_oi = run_bandit(0, 2000)


# plot the average optimal reward against the step
plt.plot(Steps[10:], Average_reward[10:], label="epislon-greedy, 0.01", color="blue")
plt.plot(Steps1[10:], Average_reward1[10:], label="epislon-greedy, 0.1", color="green")
plt.plot(Steps_oi[10:], Average_reward_oi[10:], label="optimistic initial values", color="red")
plt.xlabel('Steps')
plt.ylabel('average reward')
plt.title("Average Optimal Reward")
legend = plt.legend(loc='lower right', shadow=True)
plt.show() 
