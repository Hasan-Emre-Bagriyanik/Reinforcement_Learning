# -*- coding: utf-8 -*-
"""
Created on Thu May 11 15:14:56 2023

@author: Hasan Emre
"""

import gym
import time

env = gym.make("Taxi-v3").env
env.reset()  # reset env and return random initial state
env.render() # show

#%%

print("State space: ", env.observation_space) # 500  Kaç tane state olduğunu gösterir.

print("Action space: ",env.action_space) # 6  Kaç tane harekete sahip olduğunu gösterir.

# taxi row, taxi column, passenger index, destination 
state = env.encode(3,1,2,2)
print("State number: ",state) # 330

env.s = state
env.render()

#%%

"""
Acitons;

0: Move south (down)
1: Move north (up)
2: Move east (right)
3: Move west (left)
4: Pickup passenger
5: Drop off passenger


"""
# probability, next_state, reward, done 
# bu yukarıdakiler çıktıya göre sıralanmıştır

env.P[331]

#%%  Agent

# episode

total_reward_list = []

for j in range(5):

    env.reset()
    time_step = 0
    total_reward = 0
    list_visualize = []

    while True:
        
        time_step += 1
        
        #choose aciton
        action = env.action_space.sample()
        
        
        # perform action and reward 
        state, reward, done, _=  env.step(action) # state = next_state
        
        # total reward
        total_reward += reward
        
        # visualize
        list_visualize.append({"frame": env, 
                               "state":state,
                               "action":action,
                               "reward":reward,
                               "Total Reward": total_reward})
        #env.render()
    
        if done:
            total_reward_list.append(total_reward)
            break;

#%% 
for i , frame in enumerate(list_visualize):
    print(frame["frame"].render())
    print("Timestep: ", i+1)
    print("State: ", frame["state"])
    print("Action: ", frame["action"])
    print("Reward: ", frame["reward"])
    print("Total Reward: ", frame["Total Reward"])
    





