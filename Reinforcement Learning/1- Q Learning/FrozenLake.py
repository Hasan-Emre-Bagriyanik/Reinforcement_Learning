# -*- coding: utf-8 -*-
"""
Created on Thu May 18 05:23:21 2023

@author: Hasan Emre
"""



import gym 
import numpy as np
import random
import matplotlib.pyplot as plt

env = gym.make("FrozenLake-v0").env

# makes enviroment deterministic
# from gym.envs.registration import register
# register(
#         id = "FrozenLakeNotSlippery-v0",
#         entry_point= "gym.envs.toy_text:FrozenLakeEnv",
#         kwargs={"map_name": "4x4", "is_slippery": False},
#         max_episode_steps=100,
#         reward_threshold=0.78
#     )
env.render()

#%%
# Q learning
q_table = np.zeros([env.observation_space.n, env.action_space.n])

# Hyperparameter
alpha = 0.8
gamma = 0.95
epsilon = 0.1

# Plotting Metrix
reward_list = []



episode_number = 20000
for i in range(1,episode_number):
    
    # initialize enviroment
    state = env.reset()
    
    reward_count = 0
    
    
    while True:
        # exploit vs explore to find action
        # %10 = explore(rastgele yere gidecek), %90 exploit (daha önceden gittiği yere gidecek)
        if random.uniform(0,1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])
        
        # action process and take reward/ observation
        next_state, reward, done, _ = env.step(action)
        
        # Q learning function
        
        old_value = q_table[state,action] #old_value 
        next_max = np.max(q_table[next_state])  #next_max 
        
        next_value = (1-alpha)*old_value + alpha*(reward + gamma*next_max)
        
        # Q table update 
        q_table[state,action] = next_value
        
        
        # update state
        state = next_state
        
        reward_count += reward
        
        if done: 
            break;
        
        
        
        
    if i%10 == 0:
        
        reward_list.append(reward_count)
        print("Episode: {}, reward {}". format(i, reward_count))
        
plt.plot(reward_list)






