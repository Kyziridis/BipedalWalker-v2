#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 20:41:53 2018

@author: dead
ACTOR-CRITIC Network for Robot Locomotion

"""

#from __future__ import print_function
from keras.models import load_model, Sequential, Model
from keras.initializers import RandomUniform
from keras.regularizers import l2
from keras.layers.wrappers import TimeDistributed
from keras import backend as K
from keras.layers import Dense, Dropout, Input, MaxPooling1D,Conv1D,Conv2D,LSTM, Flatten, Reshape, GaussianNoise
from keras.layers.merge import Add, Multiply, Concatenate
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import time
import numpy as np
import gym
import random
from collections import deque
import tensorflow as tf
import os
#os.chdir("/home/dead/Documents/Master_Research/ddpg/")
from OrnsteinUhlenbeckProcess import OrnsteinUhlenbeckProcess
from tqdm import tqdm
    
class AGENT:
    def __init__(self, nx, ny, s_link, sess):
        self.nx = nx[0]  #  Observation array length   nx = env.observation_space.shape[0]
        self.nx_lidar = 10
        self.nx_obs = 14
        self.ny = ny[0]  #   Action space length       ny = env.action_space.n
        self.lr_actor = 1e-4 # Actor_Learning rate
        self.lr_critic = 3e-4 # Critic_Learning rate
        self.gamma = 0.99
        self.alpha = 0.1
        self.s_link =s_link 
        self.sess = sess
        self.deck = deque(maxlen=4000)
        self.e = 1.0
        self.e_= 0.01
        self.dc= 0.9999
        self.tau = 1e-3
        self.weight_decay = 0.0001
        self.los = []
        self.layers=[512,256]
        self.k_r = l2(self.weight_decay)
        self.k_init = RandomUniform()
        self.b_init = RandomUniform(minval=-0.003, maxval=0.003)
        self.parameters={'lr_actor': self.lr_actor,'lr_critic':self.lr_critic,'gamma':self.gamma,
                         'alpha':self.alpha, 'tau':self.tau,'dc':self.dc}
        
        
        if os.path.isfile('./' + self.s_link):
            self.message = ["\nLOAD existing keras model....", self.model.summary()]
            self.model = load_model(self.s_link)
        else:
            self.message = '\nBuilding New Model >_'
            ### Setting Actor ###
            #######################################################
            self.actor_lidar_input, self.actor_state_input, self.actor_local = self.Actor()      
            _,_, self.actor_target = self.Actor()
            # Open a placeholder for gradiends
            self.actor_critic_grads = tf.placeholder(tf.float32, [None,self.ny])
            actor_local_weights = self.actor_local.trainable_weights
            self.actor_grads = tf.gradients(self.actor_local.output, actor_local_weights, self.actor_critic_grads)
            grads = zip(self.actor_grads, actor_local_weights)
            self.optimize = tf.train.AdamOptimizer(self.lr_actor).apply_gradients(grads)
            ######################################################################
            
            ### Setting Critic ###
            ####################################################################################         
            self.critic_lidar_input, self.critic_state_input, self.critic_action_input, self.critic_local = self.Critic()      
            _,_, _, self.critic_target = self.Critic()
            # Open placeholder for gradients
            self.critic_grads = tf.gradients(self.critic_local.output,  self.critic_action_input)
            
            # Initialize for later grafient calculations
            self.sess.run(tf.global_variables_initializer())
            
        # Empty the lists
        self.ep_rewards, self.ep_obs, self.ep_act, self.ep_obs_new, self.ep_flags=[], [], [],[], []
        
    def choose_action(self,observation):
        state = observation[0][:14].reshape((1,14))
        lidar = observation[0][14:].reshape((1,10))
        # Use epsilon-greedy algorithm
        if np.random.rand() <= self.e : 
            action_ = []
            values = []
            for i in range(10000):
                action_.append(np.random.uniform(-1,1,4).reshape(1,-1))
                values.append(self.critic_target.predict([lidar,state,action_[-1]])[0][0])
                
            action = action_[np.argmax(values)]
            return action       
        
        action = self.actor_local.predict([lidar,state])
        acts_, values=[], []
        for i in range(10000):
            gaussian_noise = np.random.normal(0, 0.01 , 4)
            acts_.append(action + gaussian_noise)
            values.append(self.critic_target.predict([lidar,state,acts_[-1]])[0][0])
            
        action = acts_[np.argmax(values)]
        return action
                    
    def storing(self, observation, action, reward, observation_new, flags ):
        # Storing for replay-expirience
        self.deck.append((observation, action, reward, observation_new, flags ))
        self.ep_rewards.append(reward)
        self.ep_obs.append(observation)
        self.ep_act.append(action)
        self.ep_obs_new.append(observation_new)
        self.ep_flags.append(flags)
        
    def save(self,name):
        # Save network configuration
        self.actor_local.save(name)
        self.critic_local.save(name)

    def Actor(self):                     
        # Build Network for Actor
        #input=[lidar_input,state_input]
        lidar_input = Input(shape=(self.nx_lidar,))
        lidar_conv = Dense(self.layers[0], activation='relu', kernel_regularizer=self.k_r)(lidar_input)
        #pool = MaxPooling1D(4)(lidar_conv)
        #flat = Flatten()(lidar_conv)
               
        state_input = Input(shape=(self.nx_obs,))
        state_h1 = Dense(self.layers[0], activation='relu', kernel_regularizer=self.k_r)(state_input)
        gauss = GaussianNoise(1.0)(state_h1)
        #gauss = Flatten()(gauss)
        
        merged = Concatenate()([lidar_conv,gauss])
        #merged_reshaped = Reshape((256,1))(merged)
        merged_lstm = Dense(self.layers[1],activation='relu')(merged)
        gauss_ = GaussianNoise(1.0)(merged_lstm)
        output = Dense(self.ny, activation='tanh', kernel_initializer=self.k_init,\
                       bias_initializer=self.b_init)(gauss_)
        
        model = Model(input=[lidar_input,state_input], output=output)
        adam = Adam(lr=self.lr_actor)
        model.compile(loss='mse', optimizer=adam)
        return lidar_input,state_input, model

    def Critic(self):                     
        # Build Network for Critic       
        #input=[lidar_input,state_input,action_input]
        lidar_input = Input(shape=(self.nx_lidar,))
        lidar_conv = Dense(self.layers[0], activation='relu', kernel_regularizer=self.k_r)(lidar_input)
        #flat= Flatten()(lidar_conv)
        
        state_input = Input(shape=(self.nx_obs,))
        state_h1 = Dense(self.layers[0], activation='relu', kernel_regularizer=self.k_r)(state_input)
        #state_h1 = Flatten()(state_h1)
               
        merge1 = Concatenate()([lidar_conv,state_h1])
        #merged_dense = Dense(self.layers[0], activation='relu')(merge1)

        action_input = Input(shape=(self.ny,))
        #action_h1    = Dense(64, activation='relu')(action_input)
        
        merge2 = Concatenate()([merge1,action_input])
        #merge2reshaped = Reshape((320,1))(merge2)
        merge_lstm = Dense(self.layers[1], activation='relu')(merge2)
        output= Dense(1,activation='linear', kernel_initializer=self.k_init,\
                      bias_initializer=self.b_init)(merge_lstm)
        
        model  = Model(input=[lidar_input,state_input,action_input], output=output)
        adam  = Adam(lr=self.lr_critic)
        model.compile(loss="mse", optimizer=adam)
        return lidar_input,state_input, action_input, model
    

    def _train_critic(self, sample_indx):
        traj = sample_indx
        for observation, act, reward, obs_new, done in zip(traj[0],traj[1],traj[2],traj[3],traj[4]):  
            Q_target = np.array(reward).reshape(1,-1)
            act = act.reshape(1,-1)
            state = observation[0][:14].reshape((1,14))
            lidar = observation[0][14:].reshape((1,10))
            state_new = obs_new[0][:14].reshape((1,14))
            lidar_new = obs_new[0][14:].reshape((1,10))
            if not done:
                target_action = self.actor_target.predict([lidar_new,state_new])
                future_reward = self.critic_target.predict([lidar_new,state_new, target_action])[0][0]
                current_reward = self.critic_target.predict([lidar,state,act])[0][0]
                Q_target =(1-self.alpha)*Q_target + self.alpha* (self.gamma * future_reward - current_reward)
                Q_target = Q_target.reshape(1,-1)
            self.critic_local.fit(x=[lidar,state,act],\
                                  y=Q_target, verbose=0, epochs=1)   
            
            
    def _train_actor(self, sample_indx):
        traj = sample_indx
        for observation, act, reward, observation_new, _ in zip(traj[0],traj[1],traj[2],traj[3],traj[4]):
            state = observation[0][:14].reshape((1,14))
            lidar = observation[0][14:].reshape((1,10))
            act = act.reshape(1,-1)

            predicted_action = self.actor_local.predict([lidar,state])
            grads = self.sess.run(self.critic_grads, feed_dict = {
                    self.critic_lidar_input : lidar,
                    self.critic_state_input: state,
                    self.critic_action_input: predicted_action})[0]
            
            self.sess.run(self.optimize, feed_dict={
                    self.actor_lidar_input: lidar,
                    self.actor_state_input: state,
                    self.actor_critic_grads: grads})   
            
            
    def _update_actor_target(self):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target"""

        actor_local_weights  = self.actor_local.get_weights()
        actor_target_weights =self.actor_target.get_weights()
        
        for i in range(len(actor_target_weights)):
            actor_target_weights[i] = self.tau*actor_local_weights[i] + (1-self.tau)*actor_target_weights[i]
        self.actor_target.set_weights(actor_target_weights)          
            
    def _update_critic_target(self):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target"""
        critic_local_weights  = self.critic_local.get_weights()
        critic_target_weights = self.critic_target.get_weights()
		
        for i in range(len(critic_target_weights)):
            critic_target_weights[i] = self.tau*critic_local_weights[i] + (1-self.tau)*critic_target_weights[i]
        self.critic_target.set_weights(critic_target_weights)		

    def update_target(self):
        self._update_actor_target()
        self._update_critic_target()

    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in range(len(rewards)):
            if rewards[t] != 0:
                running_add = 0
            running_add = running_add * self.gamma + rewards[t]
            discounted_rewards[t] = running_add
        avg = np.mean(discounted_rewards)
        var = np.std(discounted_rewards)
        discounted_rewards = (discounted_rewards - avg)/var
        return discounted_rewards        
            
    def TRAIN(self):     
        discounted_rewards = self.discount_rewards(self.ep_rewards)
        sample_indx = [self.ep_obs, self.ep_act, discounted_rewards, self.ep_obs_new, self.ep_flags]                
        start_train = time.time()
        # Train Critic
        self._train_critic(sample_indx) # TRain the network critic
        # Train Actor
        self._train_actor(sample_indx)
        #
        # Update Weights 
        self.update_target() # Update the netokr local and target weights for actor AND critic
        end = time.time() - start_train
        # Empty the lists
        self.ep_rewards, self.ep_obs, self.ep_act, self.ep_obs_new, self.ep_flags=[], [], [],[], []
        return end

if __name__ == '__main__':
                
    rendering = input("Visualize rendering ? [y/n]:  ")  
    s_link = "BipedalWalker_model.h5"  
    RENDER_REWARD_MIN = 5000
    RENDER_ENV = False
    if rendering == 'y': RENDER_ENV = True  #flag for rendering the environment
    EPISODES = 100000    # Number of episodes
    
    env = gym.make('BipedalWalker-v2')
    env = env.unwrapped
    
    # Network Parameters
    nx = env.observation_space.shape # network-input  
    ny = env.action_space.shape # network-out
    #BATCH = 16
    sess = tf.Session()
    K.set_session(sess)
    agent = AGENT(nx,ny, s_link, sess)
    
    rewards_over_time = []
    error = []
    epsilon = []
    rew_var = []
    rew_mean = []
    mean_100 = []
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
    w = 0
        
    # Start running the SIMULATION   
    tqdm_run=tqdm(range(EPISODES), desc='Reward', leave=True, unit='episode')    
    for i in tqdm_run: 
        observation = env.reset()         
        observation = observation.reshape(1,-1)                
        start = time.time()
        counter = 0
        noise=OrnsteinUhlenbeckProcess(theta=0.15, mu=0, sigma=1, x0=0, dt=1e-2, n_steps_annealing=100, size=4)
        
        # Start EPISODE
        while True:            
            if RENDER_ENV:
                env.render()
                
            counter +=1 
            action = agent.choose_action(observation)
            action = np.clip(action+noise.generate(counter), -1,1)
            action = action.reshape((4,))
            observation_new, reward, flag, inf = env.step(action)
            observation_new = observation_new.reshape((1,24))                    
            # Store new information
            agent.storing(observation, action, reward, observation_new, flag)
            observation = observation_new         
            # Measure the time
            end = time.time()
            # Set time constrain: 40secs stop episode
            time_space = end - start
            if time_space > 40:
                flag = True
                            # Sum the episode rewards
            ep_rew_total = sum(agent.ep_rewards)
            mean = np.mean(agent.ep_rewards)
            var = np.var(agent.ep_rewards)
            if ep_rew_total < -300:
                flag = True
            
            if flag==True:
                # Append rewards history
                rewards_over_time.append(ep_rew_total)
                rew_mean.append(mean)
                rew_var.append(var)
                max_reward = np.max(rewards_over_time)
                episode_max = np.argmax(rewards_over_time)
                agent.ep_rewards=[]
                
                # Winning Statement
                if ep_rew_total >=300 :
                    w = w + 1
                    agent.save(s_link)                
                # Print output per 100 episodes
                if i % 10 == 0:  
                    print("\n")                      
                    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                    print("Episode: ", i)
                    print('Steps: ' + str(counter))
                    print("Time: ", np.round(time_space, 2),"secs")
                    print("Reward:", ep_rew_total)
                    print("Maximum Reward: " + str(max_reward) + "  on Episode: " + str(episode_max))
                    print("Times win: " + str(w))
                    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")
                
                if i % 100==0:
                    print("Mean reward of the past 100 episodes: ", str(np.mean(rewards_over_time[-100:])))
                    mean_100.append(np.mean(rewards_over_time[-100:]))
                    f = open('results.txt','a')
                    f.write('\n' + str(np.mean(rewards_over_time[-100:])))
                    f.close()
                break
               #if max_reward > RENDER_REWARD_MIN: RENDER_ENV = True        
           
            # END_IF
        # END_WHILE
        # Start training the Neural Network
        training_time = agent.TRAIN()
        agent.deck.clear()
        tqdm_run.set_description("Reward: " + str(ep_rew_total))
        tqdm_run.refresh()
            
        # epsilon Greedy             
        if agent.e >= agent.e_:
            agent.e *= agent.dc
            epsilon.append(agent.e)
            
    # EndFor EPISODES 
    
    np.save("rewards_over_time", rewards_over_time)
    np.save("mean100", mean_100)         
                
    plt.figure(figsize=(10,8))
    plt.plot(epsilon)
    plt.xlabel("Episodes")
    plt.ylabel("Epsilon value")
    plt.title("Epsilon Vs Episodes")
    plt.savefig("Epsilon.png") 

    plt.figure(figsize=(10,8))            
    plt.plot(rewards_over_time, label="Rewards")
    plt.plot(rew_mean, label="Mean")  
    plt.xlabel("Episodes")
    plt.ylabel("Rewards")
    plt.title("Rewards per Episode")
    plt.legend(loc=0)
    plt.savefig("Rewards.png")         
    
    plt.figure(figsize=(10,8))
    plt.plot(mean_100)
    plt.xlabel("100_episodes")
    plt.ylabel  ("Mean_value")
    plt.title('Average Reward per 100 episodes')
    plt.savefig("mean_100.png")       
            
            
            
           
            
            
            
            
            
            
            
            
            
            
            
            
            


