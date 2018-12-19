#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 20:41:53 2018

@author: dead
ACTOR-CRITIC Network for Robot Locomotion

"""

#from __future__ import print_function
from keras.models import load_model, Sequential, Model
from keras.initializers import RandomUniform, RandomNormal
from keras.regularizers import l2
from keras.layers.wrappers import TimeDistributed
from keras.activations import softplus
from keras import backend as K
from keras.layers import Dense, Dropout, Input, MaxPooling1D,Conv1D,Conv2D,LSTM, Flatten, Reshape, GaussianNoise
from keras.layers.merge import Add, Multiply, Concatenate
from keras.optimizers import Adam
from keras import backend as K
import matplotlib.pyplot as plt
import time
import numpy as np
import gym, queue
import gym.wrappers
import random
from collections import deque
import tensorflow as tf
import os
#os.chdir("/home/dead/Documents/Master_Research/ddpg/")
from rl.random import OrnsteinUhlenbeckProcess
from tqdm import tqdm

NOISE = 1.0
EPSILON = 0.15
BATCH = 8192
EPOCHS = 10
ENTROPY= 5 * 1e-3
DUMMY_ACTION, DUMMY_VALUE = np.zeros((1, 4)), np.zeros((1, 1))

# Proximal Policy Optimization loss function
def PPO_Loss(advantage,old_pred):
    def loss(y_true, y_pred):
        var = K.square(NOISE)
        pi = 3.14159
        d = K.sqrt(2*pi*var)
        prob_num = K.exp(- K.square(y_true - y_pred) / (2 * var))
        old_prob_num = K.exp(- K.square(y_true - old_pred) / (2 * var))

        prob = prob_num/d
        old_prob = old_prob_num/d
        r = prob/(old_prob + 1e-10)
        return -K.mean(K.minimum(r * advantage, K.clip(r,min_value=1-EPSILON, max_value=1+EPSILON)*advantage)) +\
                                                                ENTROPY* (prob*K.log(prob + 1e-8))
    return loss

class AGENT:
    def __init__(self, nx, ny, s_link):
        #self.env = environment
        self.nx = nx[0]  #  Observation array length   nx = env.observation_space.shape[0]
        self.ny = ny[0]  #   Action space length       ny = env.action_space.n
        self.lr_actor = 0.0001 # Actor_Learning rate
        self.lr_critic = 0.0002 # Critic_Learning rate
        self.gamma = 0.99
        self.alpha = 0.1
        self.s_link =s_link 
        #self.sess = sess
        self.deck = deque(maxlen=4000)
        self.val = False
        
        self.e = 1.0
        self.e_= 0.01
        self.dc= 0.999
        self.tau = 0.001
        self.weight_decay =0.001
        self.los = []
        self.layers=[256,128]
        
        # Network init_functions
        self.k_r = l2(self.weight_decay)
        self.initializer = "glorot_uniform"
        self.final_initializer = RandomUniform(minval = -0.003, maxval = 0.003)
        self.parameters={'lr_actor': self.lr_actor,'lr_critic':self.lr_critic,'gamma':self.gamma,
                         'alpha':self.alpha, 'tau':self.tau,'dc':self.dc}
        
        if os.path.isfile('./' + self.s_link):
            self.message = ["\nLOAD existing keras model....", self.model.summary()]
            self.model = load_model(self.s_link)
        else:
            self.message = '\nBuilding New Model >_'
            ### Build Networks Actor-Critic ###
            self.actor = self.Actor()
            self.critic = self.Critic()     
            self.sample_critic = self.sample_Critic()
            
        # Empty the lists
        self.avg_actor=[]
        self.avg_critic=[]
        self.epsilon = []
        self.reward = []
        self.ep_rewards = []
        self.ep_obs = []
        self.ep_act = []
        self.ep_obs_new = []
        self.ep_flags=[] 
        
    def choose_action(self,state,episode):
        # Use epsilon-greedy algorithm
        if np.random.rand() <= self.e : 
            # Predict a random action             
            if self.e >= self.e_:
                # Discound epsilon
                self.e *= self.dc
                self.epsilon.append(self.e)
                
            values = []
            action_=np.random.uniform(-1,1,(100,4))                                
            for i in range(100):
                # Selcet the best action among range
                values.append(self.sample_critic.predict([state.reshape(1,-1), action_])[0][0])
            action = action_[np.argmax(values)]
            action_matrix = pred_action = action
            return action , action_matrix, pred_action      
        
        # Else: use policy action
        if episode % 100 !=0 :
            action = self.actor.predict([state.reshape(1,-1), DUMMY_VALUE, DUMMY_ACTION])[0] + np.random.normal(0,NOISE,4)
            action_matrix = pred_action = action
        else:
            action = self.actor.predict([state.reshape(1,-1),DUMMY_VALUE, DUMMY_ACTION])[0]
            action_matrix = pred_action = action          
        return action, action_matrix, pred_action
                        
    def storing(self, observation, action, reward, observation_new, flags ):
        # Storing for replay-expirience
        self.deck.append((observation, action, reward, observation_new, flags ))
        self.qqq.put(np)
        self.ep_rewards.append(reward)
        self.ep_obs.append(observation)
        self.ep_act.append(action)
        self.ep_obs_new.append(observation_new)
        self.ep_flags.append(flags)
        
    def save(self):
        self.actor.save('actor'+self.s_link)
        self.critic.save('critic'+self.s_link)
        
    def sample_Critic(self):                     
        state_input = Input(shape=(self.nx,))
        action_input = Input(shape=(self.ny,))
        #
        h0 = Dense(self.layers[1], activation='relu',kernel_regularizer=self.k_r)(state_input)
        h1 = Dense(self.layers[1], activation='tanh',kernel_regularizer=self.k_r)(action_input)
        #
        conc = Concatenate()([h0,h1])
        #conc = Flatten()(conc)
        h2 = Dense(64, activation='relu', kernel_regularizer=self.k_r)(conc)
        out = Dense(1, activation='linear', kernel_regularizer=self.k_r,\
                    kernel_initializer=self.final_initializer)(h2)
        #       
        model  = Model(input=[state_input, action_input], output=out)
        adam  = Adam(lr=self.lr_critic)
        model.compile(loss="mse", optimizer=adam)
        return model   
    
    def Actor(self):                     
        state_input = Input(shape=(self.nx,), name='obse')
        advantage = Input(shape=(1,), name='avanta')
        old_pred = Input(shape=(self.ny,), name='old-pred')
        #
        h0 = Dense(self.layers[0], activation='relu', kernel_regularizer=self.k_r)(state_input)
        h1 = Dense(self.layers[1], activation='relu',kernel_regularizer=self.k_r)(h0)
        #
        out = Dense(self.ny, activation='tanh',kernel_regularizer=self.k_r,\
                    kernel_initializer=self.final_initializer)(h1)
        #
        model = Model(input=[state_input, advantage, old_pred], output=out)
        adam = Adam(lr=self.lr_actor)
        model.compile(loss=[PPO_Loss(advantage, old_pred)], optimizer=adam)
        return model

    def Critic(self):                     
        state_input = Input(shape=(self.nx,))
        #
        h0 = Dense(self.layers[0], activation='relu',kernel_regularizer=self.k_r)(state_input)
        h1 = Dense(self.layers[1], activation='relu',kernel_regularizer=self.k_r)(h0)
        #
        out = Dense(1, activation='linear', kernel_regularizer=self.k_r,\
                    kernel_initializer=self.final_initializer)(h1)
        #       
        model  = Model(input=[state_input], output=out)
        adam  = Adam(lr=self.lr_critic)
        model.compile(loss="mse", optimizer=adam)
        return model         
            
    def discount_rewards(self, rewards):        
        for j in range(len(rewards) - 2, -1, -1):
            rewards[j] += rewards[j + 1] * self.gamma
        avg = np.mean(rewards)
        var = np.std(rewards)
        discounted_rewards = (rewards - avg)/var
        return discounted_rewards        
                
    def TRAIN(self,obs, action, pred, rewards, advantage, old_pred):     
        actor_loss = []
        critic_loss = []
        epoch=[]
        # TRain in EPOCHS
        start_train = time.time()
        for i in range(EPOCHS):
            epoch_start = time.time()
            actor_loss.append(self.actor.train_on_batch([obs, advantage, old_pred], [action]))
            critic_loss.append(self.critic.train_on_batch([obs], [rewards]) )
            self.sample_critic.train_on_batch([obs,action], [rewards])
            epoch.append(time.time()-epoch_start)
        end = time.time() - start_train
        
        self.avg_actor.append(np.mean(actor_loss))
        self.avg_critic.append(np.mean(critic_loss))
        print('Training Time: %.3fs | Avg: %.3fs/epoch' % (end, np.mean(epoch)) )
        return end, self.avg_actor, self.avg_critic
                        
 
if __name__ == '__main__':
                
    rendering = input("Visualize rendering ? [y/n]:  ")  
    s_link = "BipedalWalker_model.h5"  
    RENDER_REWARD_MIN = 5000
    RENDER_ENV = False
    if rendering == 'y': RENDER_ENV = True  #flag for rendering the environment
    EPISODES = 100000    # Number of episodes
    env = gym.make('BipedalWalker-v2')  #video_callable=lambda episode_id: episode_id%10==0
    #env = gym.wrappers.Monitor(env, directory+'lala3',  force=True)

    # Environment Parameters
    nx = env.observation_space.shape # network-input  
    ny = env.action_space.shape # network-out
    #sess = tf.Session()
    #K.set_session(sess)
    agent = AGENT(nx,ny, s_link)
    seed = np.random.seed(1)
         
    print('\n+++++++++++++++++++++++++++++++++')
    print('BipedalWalker-v2 Starts...  >_')
    print('+++++++++++++++++++++++++++++++++') 
    print(agent.message)    
    print("Environment Observation_space: ", env.observation_space)
    print("Environment Action_space: ", env.action_space) 
    print("Number of Episodes: " + str(EPISODES))
    print('-----------------------------------')
    
    print("\n:::::Algorithm_Parameters::::::")
    print(list(agent.parameters.items()))
    # Initialize Simulation
    batch = [[], [], [], []]  
    counter = 0
    mean100 =[]
    # Start running the SIMULATION       
    for i in range(EPISODES):         
        observation = env.reset()         
        observation = observation.reshape(1,-1)                
        start = time.time()
        ep_r = 0
        #noise=OrnsteinUhlenbeckProcess(theta=.15, mu=0., sigma=.3,size=4)
        while True:
            if RENDER_ENV: env.render()
            act, action_matrix, predicted_action = agent.choose_action(observation,i)
            act.reshape((4,))
            
            if counter >= BATCH:
                # UPDATE NETWORKS TRAIN IN BATCHES
                r = agent.discount_rewards(batch[3]) # Discound and Normalize rewards   
                r = r.reshape(-1,1)
                obs, action, pred = np.vstack(batch[0]), np.array(batch[1]), np.array(batch[2])
                #pred = np.reshape(pred, (pred.shape[0], pred.shape[2]))
                old_pred = pred
                pred_values = agent.critic.predict(obs)
                advantage = r - pred_values
                # Train on batches #
                train_time, avg_actor_loss, avg_critic_loss = agent.TRAIN(obs, action, pred, r, advantage, old_pred) 
                batch=[[],[],[],[]]
                counter=0                

            batch[0].append(observation)
            batch[1].append(action_matrix)
            batch[2].append(predicted_action)
            # ENVIRONMENT STEP
            observation_new, reward, flag, info = env.step(np.clip(act,-1,1))   
            observation_new.reshape(1,-1)
            batch[3].append(reward)

            ep_r += reward
            counter +=1
            observation = observation_new
            
            if flag:
                # Break episode
                agent.ep_rewards.append(ep_r)
                print('Episode: %i' % i, '| Reward: %.2f' % ep_r , ' | Trajectories: %i' % counter)
                
                if i %100 ==0:
                    agent.save()
                    mean100.append(np.mean(agent.ep_rewards[-100:]))
                    print('Average Reward last 100-Episodes: %.3f' % mean100[-1])
                break
    env.close()
    
    # Export        
    np.save("rewards_over_time", agent.ep_rewards)
    np.save("mean100", mean100)      
                
    plt.figure(figsize=(10,8))
    plt.plot(agent.avg_actor, label='Avg.ActorLoss')
    plt.plot(agent.avg_critic, label='Avg.CriticLoss')
    plt.xlabel("Training Time-Stamp")
    plt.ylabel("Average Error")
    plt.title("Epsilon Vs Episodes")
    plt.savefig("Plots/Epsilon.png")  
                
    plt.figure(figsize=(10,8))
    plt.plot(agent.epsilon)
    plt.xlabel("Episodes")
    plt.ylabel("Epsilon value")
    plt.title("Epsilon Vs Episodes")
    plt.savefig("Plots/Epsilon.png") 

    plt.figure(figsize=(10,8))            
    plt.plot(agent.ep_rewards) 
    plt.xlabel("Episodes")
    plt.ylabel("Rewards")
    plt.title("Rewards per Episode")
    plt.legend(loc=0)
    plt.savefig("Plots/Rewards.png")         
    
    plt.figure(figsize=(10,8))
    plt.plot(mean100)
    plt.xlabel("100_episodes")
    plt.ylabel  ("Mean_value")
    plt.title('Average Reward per 100 episodes')
    plt.savefig("Plots/mean_100.png")       
            
            
            
           
            
            
            
            

            
            
            
            
            
            
            


