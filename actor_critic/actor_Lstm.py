"""
Created on Wed Apr 25 20:41:53 2018

@author: dead
ACTOR-CRITIC Network for Robot Locomotion

Two deep q-learning networks with two layers each one and mini-batch experience replay update rule.
Actor_local >> predicts the actual action for the next state
Critic_local >> predicts the q_value

Episode reset 20 secs.

--------
-Actor_Local: 
    Input: obs_new
    H0   : 128 relu
    H1   : 64 relu
    Out  : 4  tanh
    bias_init: Uniform(-0.003, 0.003)

-Actor_Target: Same as Actor_local used for predicting the next action for the Q-learning rule.

--------   
-Critict_Local:
    Input1 : obs_new
    H0     : 128 relu
    H1     : 64 relu
    
    Input2 : pred_action
    H2     : 64 relu
    
    merge  : 32 relu
    Out    : 1  linear
   
-Critic_Target: Same as critic_local used for predicting the q_target value based on the predicted action.    

--------
Update Rule Q-learning:
    - mini-batch(16)
    - Q_target = (1-a)reward + a*gamma*mean(predict(obs+obs_new) )

-------    
Soft-Update Netowrk-Weights:
    - θ_target = τ*θ_local + (1 - τ)*θ_target
    
-------
Parameters:        
    BATCH = 16
    self.e = 1.0
    self.e_= 0.01
    self.dc= 0.9999
    self.tau = 1e-3
    self.weight_decay = 0.0001
    self.lr_actor = 1e-4 # Actor_Learning rate
    self.lr_critic = 3e-4 # Critic_Learning rate
    self.gamma = 0.99
    self.alpha = 0.1
------- 

"""

#from __future__ import print_function
import pandas as pd
from keras.models import load_model, Sequential, Model
from keras.initializers import RandomUniform
from keras.regularizers import l2
from keras.layers.wrappers import TimeDistributed
from keras import backend as K
from keras.layers import Dense, Dropout, Input, MaxPooling1D,Conv1D,LSTM, Flatten, Reshape
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

    
class AGENT:
    def __init__(self, nx, ny, s_link, sess, batch):
        self.nx = nx[0]  #  Observation array length   nx = env.observation_space.shape[0]
        self.nx_lidar = 10
        self.nx_obs = 14
        self.ny = ny[0]  #   Action space length       ny = env.action_space.n
        self.lr_actor = 1e-4 # Actor_Learning rate
        self.lr_critic = 3e-4 # Critic_Learning rate
        self.batch = batch
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
        self.parameters={'lr_actor': self.lr_actor,'lr_critic':self.lr_critic,'gamma':self.gamma,
                         'alpha':self.alpha, 'tau':self.tau,'dc':self.dc,'Batch':self.batch}
        
        
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
            self.actor_grads = tf.gradients(self.actor_local.output, actor_local_weights, -self.actor_critic_grads)
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
        self.ep_rewards=[]
        
    def choose_action(self,observation):
        # Use epsilon-greedy algorithm
        if np.random.rand() <= self.e : 
            action = np.random.uniform(-1,1,4)
            return action       
        state = observation[0][:14].reshape((1,14))
        lidar = observation[0][14:].reshape((1,10,1))
        action = self.actor_local.predict([lidar,state])
        return action
                    
    def storing(self, observation, action, reward, observation_new, flags ):
        # Storing for replay-expirience
        self.deck.append((observation, action, reward, observation_new, flags ))
        self.ep_rewards.append(reward)
        
    def save(self,name):
        # Save network configuration
        self.actor_local.save(name)
        self.critic_local.save(name)

    def Actor(self):                     
        # Build Network for Actor
        #input=[lidar_input,state_input]
        lidar_input = Input(shape=(self.nx_lidar,1))
        lidar_conv = Conv1D(64, 4, activation='relu')(lidar_input)
        pool = MaxPooling1D(4)(lidar_conv)
        flat = Flatten()(pool)
               
        state_input = Input(shape=(self.nx_obs,))
        state_h1 = Dense(192, activation='relu')(state_input)
        
        merged = Concatenate()([flat,state_h1])
        merged_reshaped = Reshape((256,1))(merged)
        merged_lstm = LSTM(256,activation='relu',input_shape=(1,256,1))(merged_reshaped)
        output = Dense(self.ny, activation='tanh')(merged_lstm)
        
        model = Model(input=[lidar_input,state_input], output=output)
        adam = Adam(lr=self.lr_actor)
        model.compile(loss='mse', optimizer=adam)
        return lidar_input,state_input, model

    def Critic(self):                     
        # Build Network for Critic       
        #input=[lidar_input,state_input,action_input]
        lidar_input = Input(shape=(self.nx_lidar,1))
        lidar_conv = Conv1D(64, 4, activation='relu',input_shape=(self.nx_lidar,1))(lidar_input)
        pool = MaxPooling1D(4)(lidar_conv)
        flat= Flatten()(pool)
        
        state_input = Input(shape=(self.nx_obs,))
        state_h1 = Dense(192, activation='relu')(state_input)
        
        action_input = Input(shape=(self.ny,))
        action_h1    = Dense(64, activation='relu')(action_input)
        
        merge1 = Concatenate()([flat,state_h1])
        merged_dense = Dense(256, activation='relu')(merge1)
        
        merge2 = Concatenate()([merged_dense,action_h1])
        merge2reshaped = Reshape((320,1))(merge2)
        merge_lstm = LSTM(320, activation='relu',input_shape=(1,320,1))(merge2reshaped)
        output= Dense(1,activation='linear')(merge_lstm)
        
        model  = Model(input=[lidar_input,state_input,action_input], output=output)
        adam  = Adam(lr=self.lr_critic)
        model.compile(loss="mse", optimizer=adam)
        return lidar_input,state_input, action_input, model
    

    def _train_critic(self, sample_indx):
        for observation, act, reward, obs_new, done in sample_indx:  
            Q_target = np.array(reward).reshape(1,-1)
            act = act.reshape(1,-1)
            state = observation[0][:14].reshape((1,14))
            lidar = observation[0][14:].reshape((1,10,1))
            state_new = obs_new[0][:14].reshape((1,14))
            lidar_new = obs_new[0][14:].reshape((1,10,1))
            if not done:
                target_action = self.actor_target.predict([lidar_new,state_new])
                future_reward = self.critic_target.predict([lidar_new,state_new, target_action])[0][0]
                Q_target =(1-self.alpha)*Q_target +  self.alpha* self.gamma * future_reward
                Q_target = Q_target.reshape(1,-1)
            self.critic_local.fit(x=[lidar,state,act],\
                                  y=Q_target, verbose=0, epochs=1)   
            
            
    def _train_actor(self, sample_indx):
        for observation, act, reward, observation_new, _ in sample_indx:
            state = observation[0][:14].reshape((1,14))
            lidar = observation[0][14:].reshape((1,10,1))

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
            
      
    def TRAIN(self, batch):

        if len(self.deck) < batch:
            return
        # Random sample
        sample_indx = random.sample(self.deck, batch)
        
        # Initialize Dictionary for time
        time_all = {}
        start_train = time.time()
               
        # Train Critic
        start_critic = time.time()
        self._train_critic(sample_indx) # TRain the network critic
        end_critic = time.time()-start_critic
                
        # Train Actor
        start_actor= time.time()
        self._train_actor(sample_indx)
        end_actor = time.time() - start_actor
                
        # Update Weights of 
        start_update = time.time()
        self.update_target() # Update the netowkr local and target weights for actor AND critic
        end_update = time.time() - start_update
        
        # Erase episode rewards list
        self.ep_rewards= []
                     
        if self.e >= self.e_:
            self.e *= self.dc
            
        end = time.time() - start_train
        time_all={'Critic_Train': round(end_critic,1) ,\
                  'Actor_Train':round(end_actor,1),\
                  'Weights_Update': round(end_update,1) ,\
                  'Train_Overall':round(end,1) }
        return time_all

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
    BATCH = 16
    sess = tf.Session()
    K.set_session(sess)
    agent = AGENT(nx,ny, s_link, sess, BATCH)
    
    rewards_over_time = []
    error = []
    epsilon = []
    rew_var = []
    rew_mean = []
    mean_100 = []
    seed = np.random.seed(666)
         
    print('\n+++++++++++++++++++++++++++++++++')
    print('BipedalWalker-v2 Starts...  >_')
    print('+++++++++++++++++++++++++++++++++')
    print("><><><><><><><><><><><><><><><><>")
    print("-----------------------------------")       
    print(agent.message)    
    print("Environment Observation_space: ", env.observation_space)
    print("Environment Action_space: ", env.action_space) 
    print("Number of Episodes: " + str(EPISODES))
    print('-----------------------------------')
    
    print("\n:::::Algorithm_Parameters::::::")
    print(list(agent.parameters.items()))
    w = 0
        
    # Start running the episodes        
    for i in range(EPISODES): 
        observation = env.reset()         
        observation = observation.reshape(1,-1)                
        start = time.time()
        
        while True:            
            if RENDER_ENV:
                env.render()
            
            action = agent.choose_action(observation)
            action = action.reshape((4,))
            observation_new, reward, flag, inf = env.step(np.clip(action,-1,1))
            observation_new = observation_new.reshape((1,24))                    
            # Store new information
            agent.storing(observation, action, reward, observation_new, flag)   
            observation = observation_new         
            # Measure the time
            end = time.time()
            time_space = end - start
            
            if time_space > 30:
                flag = True
          
            # Sum the episode rewards
            ep_rew_total = sum(agent.ep_rewards)
            mean = np.mean(agent.ep_rewards)
            var = np.var(agent.ep_rewards)
            if ep_rew_total < -300:
                flag = True
            
            if flag==True:
                rewards_over_time.append(ep_rew_total)
                rew_mean.append(mean)
                rew_var.append(var)
                max_reward = np.max(rewards_over_time)
                episode_max = np.argmax(rewards_over_time)
                if ep_rew_total >=300 :
                    w = w + 1
                    agent.save(s_link)
                                        
                print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                print("Episode: ", i)
                print("Time: ", np.round(time_space, 2),"secs")
                print("Reward:", ep_rew_total)
                print("Maximum Reward: " + str(max_reward) + "  on Episode: " + str(episode_max))
                print("Times win: " + str(w))
                
                if i % 100 ==0:
                    print("Mean reward of the past 100 episodes: ", str(np.mean(rewards_over_time[-100:])))
                    mean_100.append(np.mean(rewards_over_time[-100:]))
                    f = open('results.txt','a')
                    f.write('\n' + str(np.mean(rewards_over_time[-100:])))
                    f.close()
                
                # Start training the Neural Network
                training_time = agent.TRAIN(BATCH)
                print("Time: " + str(list(training_time.items())))
                
                epsilon.append(agent.e)
                
                if max_reward > RENDER_REWARD_MIN: RENDER_ENV = True
                
                break
    
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
    plt.plot(rew_var, label="Variance")    
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
            
            
            
           
            
            
            
            
            
            
            
            
            
            
            
            
            


