import numpy as np
import random
import sys
import os


TIME_SLOTS = 1
NUM_CHANNELS = 2
NUM_USERS = 3
ATTEMPT_PROB = 0.6
GAMMA = 0.90
PU=[1,2,4,5,9]

class env_network:
    def __init__(self,num_users,num_channels,attempt_prob):
        self.ATTEMPT_PROB = attempt_prob
        self.NUM_USERS = num_users
        self.NUM_CHANNELS = num_channels
        self.REWARD = 1

        self.action_space = np.arange(self.NUM_CHANNELS+1)
        self.users_action = np.zeros([self.NUM_USERS],np.int32) 
        self.users_observation = np.zeros([self.NUM_USERS],np.int32)
        
        
     # randomally generated an array of size num_users from action space   
    def sample(self):
        x =  np.random.choice(self.action_space,size=self.NUM_USERS)
        return x
    
    
    def step(self,action):
        #channel allocation frequeny : K+1
        channel_alloc_frequency = np.zeros([self.NUM_CHANNELS + 1],np.int32)
        obs = []
        reward = np.zeros([self.NUM_USERS])
        j = 0
        #print(PU)
        for  each in action:
            prob = random.uniform(0,1)
            if prob <= self.ATTEMPT_PROB:
                if each in PU:
                    self.users_action[j] = 0
                    reward[j]=-1
                else:
                    self.users_action[j] = each  
                    channel_alloc_frequency[each]+=1
            j+=1
            #deallocate all the channels wich have assigned more than one user
        for i in range(1,len(channel_alloc_frequency)):
            if channel_alloc_frequency[i] > 1:
                channel_alloc_frequency[i] = 0
        for i in range(len(action)):
            self.users_observation[i] = channel_alloc_frequency[self.users_action[i]]
            if self.users_action[i] == 0 :
                self.users_observation[i] = 0
            if self.users_observation[i] == 1:
                reward[i] = 1
            obs.append((self.users_observation[i],reward[i]))
        residual_channel_capacity = channel_alloc_frequency[1:]
        residual_channel_capacity = 1-residual_channel_capacity
        obs.append(residual_channel_capacity)
        return obs



