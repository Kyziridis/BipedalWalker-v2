#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 20:41:53 2018

@author: dead
ACTOR-CRITIC Network for Robot Locomotion

"""

#from __future__ import print_function
import tensorflow as tf
import scipy.signal
import math
import numba as nb
from tensorflow import keras
from keras.models import load_model, Sequential, Model
from keras.initializers import RandomUniform, RandomNormal
from keras.regularizers import l2
from keras.layers.wrappers import TimeDistributed
from keras.activations import softplus
from keras import backend as K
from keras.layers import Dense, Dropout, Input, MaxPooling1D,Conv1D,Conv2D,LSTM, Flatten, Reshape, GaussianNoise, Lambda
from keras.layers.merge import Concatenate
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
from helpers import ProcessRewards
import matplotlib.pyplot as plt
import time
import numpy as np
import gym, queue
import gym.wrappers
from collections import deque
import os
#os.chdir("/home/dead/Documents/Master_Research/ddpg/")

LAMBDA = 0.95
NOISE = 1.0
EPSILON = 0.2
BATCH = 8192
EPOCHS = 15
ENTROPY= 5 * 1e-3
SIGMA_FLOOR = 0
MINIBATCH = 32
LR = 0.0001
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
        return -K.mean(K.minimum(r * advantage, K.clip(r,1-EPSILON,1+EPSILON)*advantage))
    return loss

def Actor_loss(advantage,old_pred):
    def loss(y_true,y_pred):
        var = K.square(NOISE)
        pi = 3.141592
        d = K.sqrt(2*pi*var)
        dist = y_pred
        dist_ = old_pred
        ratio = K.maximum(dist,1e-6)/K.maximum(dist_,1e-6)
        ratio = K.clip(ratio,0,10)
        surrogate1 = advantage*ratio
        surrogate2 = advantage*K.clip(ratio,1-EPSILON,1+EPSILON)
        error = -K.mean(K.minimum(surrogate1,surrogate2))
        return error
    return loss

def Critic_loss(old):
    def loss(y_true,y_pred):
        clipped = old + K.clip(y_pred - old, -EPSILON, EPSILON)
        loss_q1 = K.square(y_true-clipped)
        loss_q2 = K.square(y_true - y_pred)
        error = K.mean(K.maximum(loss_q1,loss_q2))*0.5
        return error
    return loss

class AGENT:
    def __init__(self, nx, ny, s_link):
        #self.env = environment
        self.s_state = nx
        self.nx = nx[0]  #  Observation array length   nx = env.observation_space.shape[0]
        self.ny = ny[0]  #   Action space length       ny = env.action_space.n
        self.gamma = 0.99
        self.alpha = 0.1
        self.s_link =s_link 
        self.lr=LR
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
        self.parameters={'lr':self.lr,'gamma':self.gamma,
                         'alpha':self.alpha, 'tau':self.tau,'dc':self.dc}
        
        self.sess = tf.Session()
        self.actions = tf.placeholder(tf.float32, [None, 4], 'action')
        self.state = tf.placeholder(tf.float32, [None] + list(self.s_state), 'state')
        self.advantage = tf.placeholder(tf.float32, [None, 1], 'advantage')
        self.rewards = tf.placeholder(tf.float32, [None, 1], 'discounted_r')

        self.dataset = tf.data.Dataset.from_tensor_slices({"state": self.state, "actions": self.actions,
                                                           "rewards": self.rewards, "advantage": self.advantage})
        self.dataset = self.dataset.shuffle(buffer_size=10000)
        self.dataset = self.dataset.batch(MINIBATCH)
        self.dataset = self.dataset.cache()
        self.dataset = self.dataset.repeat(EPOCHS)
        self.iterator = self.dataset.make_initializable_iterator()
        self.batch = self.iterator.get_next()        
        
        self.sess.run(tf.global_variables_initializer()) # initialize 
        
        if os.path.isfile('./' + self.s_link):
            self.message = ["\nLOAD existing keras model....", self.model.summary()]
            self.model = load_model(self.s_link)
        else:
            self.message = '\nBuilding New Model >_'
            ### Build Networks Actor-Critic ###
            # Actors
            self.actor_old  = self.Actor()
            self.actor_new = self.Actor()
            self.actor_ = self.Actor()
            # Critics
            self.critic_old = self.Critic()     
            self.critic_new = self.Critic()
            self.critic_ = self.Critic()
            
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
     
    #@nb.jit    
    def choose_action(self,state,episode):      
        if episode % 100 !=0 :
            action = self.actor_.predict([state.reshape(1,-1), DUMMY_VALUE, DUMMY_ACTION])[0] + np.random.normal(0,NOISE,4)
            action_matrix = pred_action = action
        else:
            action = self.actor_.predict([state.reshape(1,-1),DUMMY_VALUE, DUMMY_ACTION])[0]
            action_matrix = pred_action = action    
        
        #old_act = self.actor_old.predict([self.batch["state"], DUMMY_VALUE, DUMMY_ACTION])[0]
        #new_act = self.actor_new.predict([self.batch["state"], DUMMY_VALUE, DUMMY_ACTION])[0]
        #old_q = self.critic_old.predict([self.batch["state"], DUMMY_VALUE, DUMMY_ACTION])[0]
        #new_q = self.critic_new.predict([self.batch["state"], DUMMY_VALUE, DUMMY_ACTION])[0]    
        value  = self.critic_.predict([state.reshape(1,-1), DUMMY_VALUE])[0]   
        return action, action_matrix, pred_action, value#, old_act, new_act, old_q, new_q
    
    #@nb.jit(nopython=True)                    
    def storing(self, observation, action, reward, observation_new, flags ):
        # Storing for replay-expirience
        self.deck.append((observation, action, reward, observation_new, flags ))
        self.ep_rewards.append(reward)
        self.ep_obs.append(observation)
        self.ep_act.append(action)
        self.ep_obs_new.append(observation_new)
        self.ep_flags.append(flags)
    
    #@nb.jit(nopython=True)   
    def save(self):
        self.actor_.save_weights('actor'+self.s_link)
        self.critic_.save_weights('critic'+self.s_link)
        
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
        model  = Model(inputs=[state_input, action_input], outputs=out)
        adam  = Adam(lr=LR)
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
        
        var = Lambda(lambda x: K.var(x))(out)
        out = Lambda(lambda x: K.random_normal((self.ny,), out , var ))(out)
        #out = K.get_value(out)
        #
        model = Model(inputs=[state_input, advantage, old_pred], outputs=out)
        adam = Adam(lr=LR)
        model.compile(loss=[PPO_Loss(advantage=advantage, old_pred=old_pred)], optimizer=adam)
        lrate = LearningRateScheduler(self.step_decay)
        callbacks_list = [lrate]
        return model

    def Critic(self):                     
        state_input = Input(shape=(self.nx,))
        old_value = Input(shape=(1,))
        #
        h0 = Dense(self.layers[0], activation='relu',kernel_regularizer=self.k_r)(state_input)
        h0 = Dropout(0.4)(h0)
        h1 = Dense(self.layers[1], activation='relu',kernel_regularizer=self.k_r)(h0)
        h1 = Dropout(0.5)(h1)
        #
        out = Dense(1, activation='linear', kernel_regularizer=self.k_r,\
                    kernel_initializer=self.final_initializer)(h1)
        #       
        model  = Model(inputs=[state_input, old_value], outputs=out)
        adam  = Adam(lr=0.0)
        model.compile(loss=[Critic_loss(old=old_value)], optimizer=adam)
        lrate = LearningRateScheduler(self.step_decay)
        callbacks_list = [lrate]
        return model      
    
    # learning rate schedule
    def step_decay(self,epoch):
        	initial_lrate = LR
        	drop = 0.15
        	epochs_drop = 10.0
        	lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
        	return lrate
        
    #@nb.jit    
    def discount(self,x, gamma, terminal_array=None):
        if terminal_array is None:
            return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]
        else:
            y, adv = 0, []
            terminals_reversed = terminal_array[1:][::-1]
            for step, dt in enumerate(reversed(x)):
                y = dt + gamma * y * (1 - terminals_reversed[step])
                adv.append(y)            
        return np.array(adv)[::-1]    
    
    def update_nets(self):
        # Get weights from current policy
        weights_a = self.actor_new.get_weights()
        self.actor_old.set_weights(weights_a)
        weights_c = self.critic_new.get_weights()
        self.critic_old.set_weights(weights_c)
        # Update Current NEts Players:
        self.actor_.set_weights(weights_a)
        self.critic_.set_weights(weights_c)
    
    #@nb.jit
    def TRAIN(self,obs, action, pred, rewards, advantage, old_pred, pred_values):     
        actor_loss = []
        critic_loss = []
        epoch=[]
        # Update Networks
        self.update_nets()
        # TRain in EPOCHS
        start_train = time.time()
        lrate = LearningRateScheduler(self.step_decay)
        callbacks_list = [lrate]
        for i in range(EPOCHS):
            epoch_start = time.time()
            # Main PLayer train
            actor_loss.append(self.actor_.train_on_batch([obs, advantage, old_pred], [action]))
            critic_loss.append(self.critic_.train_on_batch([obs, pred_values], [rewards]) )
            self.actor_old.train_on_batch([obs, advantage, old_pred], [action])
            self.actor_new.train_on_batch([obs, advantage, old_pred], [action])
            self.critic_old.train_on_batch([obs, pred_values], [rewards])
            self.critic_new.train_on_batch([obs, pred_values], [rewards])
            
            epoch.append(time.time()-epoch_start)
        end = time.time() - start_train
        
        self.avg_actor.append(np.mean(actor_loss))
        self.avg_critic.append(np.mean(critic_loss))
        message = 'Training Time: %.3fs | Avg: %.3fs/epoch' % (end, np.mean(epoch))
        #print('Training Time: %.3fs | Avg: %.3fs/epoch' % (end, np.mean(epoch)) )
        return end, self.avg_actor, self.avg_critic, message
                        
if __name__ == '__main__':
                
    rendering = input("Visualize rendering ? [y/n]:  ")  
    s_link = "BipedalWalker_model.h5"  
    RENDER_REWARD_MIN = 5000
    RENDER_ENV = False
    if rendering == 'y': RENDER_ENV = True  #flag for rendering the environment
    EPISODES = 10000    # Number of episodes
    env = gym.make('BipedalWalker-v2')  #video_callable=lambda episode_id: episode_id%10==0
    #env = gym.wrappers.Monitor(env, directory+'lala3',  force=True)

    # Environment Parameters
    nx = env.observation_space.shape # network-input  
    ny = env.action_space.shape # network-out
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
    batches = [[], [], [], [],[], []]  
    counter = 0
    mean100 =[]
    traint_t = 0
    proc = ProcessRewards()
    # Start running the SIMULATION       
    for i in range(EPISODES):         
        observation = env.reset()         
        observation = observation.reshape(1,-1)                
        start = time.time()
        ep_r, ep_t = 0, 0
        #noise=OrnsteinUhlenbeckProcess(theta=.15, mu=0., sigma=.3,size=4)
        while True:
            if RENDER_ENV: env.render()
            act, action_matrix, predicted_action, q_value = agent.choose_action(observation,i)
            #act, action_matrix, predicted_action, q_value, old_act, new_act, old_q, new_q = agent.choose_action(observation,i)
            act.reshape((4,))
            
            if counter >= BATCH:
                # UPDATE NETWORKS TRAIN IN BATCHES
                traint_t +=1
                print('Mpika!!!! train_call : %.i' % traint_t)
                r = np.array(batches[3])
                r = r.reshape(-1,1)
                proc.update(r) 
                r = np.clip(r/proc.std, -10,10)
                #r = r.reshape(-1,1)
                obs, action, pred = np.vstack(batches[0]), np.array(batches[1]), np.array(batches[2])
                old_pred = pred
                v_final = [q_value * (1-flag)]
                pred_values =np.array(batches[5] + v_final ) # Get Q-values
                terminals = np.array(batches[4] + [flag])
                terminals = terminals.reshape(-1,1)
                
                deltas = r + agent.gamma * pred_values[1:]*(1-terminals[1:]) - pred_values[:-1]
                advantage = agent.discount(deltas, agent.gamma * LAMBDA, terminals)
                returns = advantage + np.array(batches[5])
                advantage = (advantage - advantage.mean()) / np.maximum(advantage.std(), 1e-6)

                # Train on batches #
                train_time, avg_actor_loss, avg_critic_loss, message = agent.TRAIN(obs, action, pred, returns, advantage,\
                                                                                   old_pred, np.array(batches[5])) 
                print(message)
                batches = [[], [], [], [],[], []]
                counter=0        

            batches[0].append(observation)
            batches[1].append(action_matrix)
            batches[2].append(predicted_action)
            # ENVIRONMENT STEP
            observation_new, reward, flag, info = env.step(np.clip(act,-1,1))   
            observation_new.reshape(1,-1)
            batches[3].append(reward)
            batches[4].append(flag)
            batches[5].append(q_value)

            ep_r += reward
            ep_t += 1
            counter +=1
            observation = observation_new
            
            if flag:
                # Break episode
                agent.ep_rewards.append(ep_r)
                print('Episode: %i' % i, ' | Reward: %.2f' % ep_r , ' | Trajectories: %i' % ep_t)
                
                if i %100 ==0:
                    agent.save()
                    mean100.append(np.mean(agent.ep_rewards[-100:]))
                    print('Average Reward last 100-Episodes: %.3f' % mean100[-1])
                break
    env.close()
    
    os.mkdir("./XPS")
    os.mkdir("./Plots")
    # Export        
    np.save("XPS/rewards_over_time", agent.ep_rewards)
    np.save("XPS/mean100", mean100)      
                
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
            
            
            
           
            
            
            
            

            
            
            
            
            
            
            


