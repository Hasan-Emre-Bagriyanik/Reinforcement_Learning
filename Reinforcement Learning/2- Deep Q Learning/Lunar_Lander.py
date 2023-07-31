# -*- coding: utf-8 -*-
"""
Created on Sun May 21 12:02:33 2023

@author: Hasan Emre
"""

import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import random


class DQLAgent:
    
    def __init__(self,env):
        # parameter / hyperparameter
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n
        
        self.gamma = 0.99
        self.learning_rate = 0.0001
        
        self.epsilon = 1 # explore
        self.epsilon_decay = 0.9993
        self.epsilon_min = 0.1
        
        self.memory = deque(maxlen=4000)
        
        self.model = self.build_model()
        self.target_model = self.build_model()
    
    def build_model(self):
        # ANN for deep q learning 
        model = Sequential()
        model.add(Dense(64, input_dim = self.state_size, activation = "relu"))
        model.add(Dense(64, activation = "relu"))
        model.add(Dense(self.action_size, activation="linear"))
        model.compile(loss = "mse", optimizer = Adam(lr = self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        # storage
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        # acting: explore(keşfetmek) and exploit
        if random.uniform(0, 1) <= self.epsilon:
            return env.action_space.sample()
        else:
            act_values = self.model.predict(state)
            return np.argmax(act_values[0])
    
    def replay(self, batch_size):
        # training
        if len(agent.memory) < batch_size:
            return
        # vectorized method for experience replay
        minibatch = random.sample(self.memory, batch_size)
        minibatch = np.array(minibatch)
        not_done_indices = np.where(minibatch[:, 4] == False)
        y = np.copy(minibatch[:, 2])
    
        # If minibatch contains any non-terminal states, use separate update rule for those states
        if len(not_done_indices[0]) > 0:
            predict_sprime = self.model.predict(np.vstack(minibatch[:, 3]))
            predict_sprime_target = self.target_model.predict(np.vstack(minibatch[:, 3]))
    
            # Non-terminal update rule
            y[not_done_indices] += np.multiply(self.gamma, np.squeeze(predict_sprime_target[not_done_indices, np.argmax(predict_sprime[not_done_indices], axis=1)]))
    
        actions = np.array(minibatch[:, 1], dtype=int)
        y_target = self.model.predict(np.vstack(minibatch[:, 0]))
        y_target[range(batch_size), actions] = y
        self.model.fit(np.vstack(minibatch[:, 0]), y_target, epochs=1, verbose=0)

    # yukarıdaki replay fonksiyonu aşağıya göre daha hızlı çalışıyor 
    
    # def replay(self, batch_size):
    #     # training 
    #     if len(self.memory) < batch_size:
    #         return
    #     minibatch = random.sample(self.memory,batch_size)
    #     for state, action, reward, next_state, done in minibatch:
    #         if done:
    #             target = reward
    #         else:
    #             target = reward + self.gamma*np.amax(self.model.predict(next_state)[0])
            
    #         train_target = self.model.predict(state)
    #         train_target[0][action] = target
    #         self.model.fit(state,train_target, verbose=0)
      
    
    def adaptiveEGreedy(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def targetModelUpdate(self):
        self.target_model.set_weights(self.model.get_weights())
    

    
if __name__ == "__main__":
    
    # initialize env and agent
    
    
    env = gym.make("LunarLander-v2")
    agent = DQLAgent(env)
    state_number = env.observation_space.shape[0]
    
    episodes = 50 # 100 değeri daha iyi sonuç verecektir 
    batch_size = 32
    
    for e in range(episodes):
        
        # initialize environment
        state = env.reset()
        
        state = np.reshape(state, [1,state_number])
        
        total_reward = 0
        
        for time in range(1000):
            
            # action 
            action = agent.act(state) # Select an action
            
            # step
            next_state, reward, done , _ = env.step(action)
            next_state = np.reshape(next_state, [1,state_number])
            
            # remember
            agent.remember(state, action, reward, next_state, done)
            
            # update state
            state = next_state
            
            # replay
            agent.replay(batch_size)
            
            total_reward += reward
            
            if done:
                agent.targetModelUpdate()
                break
            
        # adjust epsilon
        agent.adaptiveEGreedy()
        
        # running average of past 100 episodes
        print("Episode: {}, Reward: {}".format(e,total_reward))
            
#%%  test visualize
import time

trained_model = agent
state = env.reset()
state = np.reshape(state, [1,env.observation_space.shape[0]])
time_t = 0

while True:
    env.render()
    action = trained_model.act(state)
    next_state, reward, done, _ = env.step(action)
    next_state = np.reshape(state, [1,8])
    state = next_state
    time_t += 1
    print(time_t)
    time.sleep(0.05)
    if done:
        break
print("Done")
















        
        
    
    