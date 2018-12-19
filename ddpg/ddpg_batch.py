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
from keras.optimizers import Adam, RMSprop
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import time
import numpy as np
import gym
import gym.wrappers
#from collections import deque
import tensorflow as tf
import os
import sys
#os.chdir("/home/dead/Documents/Master_Research/ddpg/")
from OrnsteinUhlenbeckProcess import OrnsteinUhlenbeckProcess

EPISODES = int(sys.argv[1])
BATCH = int(sys.argv[2])
lstm = sys.argv[3]
arg = int(sys.argv[4])
epc = int(sys.argv[5])
if lstm == 'y':
    lstm = True
else:
    lstm = False
DUMMY_ACTION, DUMMY_VALUE = np.zeros((1, 4)), np.zeros((1,1))
tf.device('cpu=0')

def actor_loss(advantage, old_prediction):
    def loss(y_true, y_pred):
        var = K.variable(1.0)
        pi = K.variable(3.1415926)
        denom = K.sqrt(2 * pi * var)
        prob_num = K.exp(- K.square(y_true - y_pred) / (2 * var))
        old_prob_num = K.exp(- K.square(y_true - old_prediction) / (2 * var))

        prob = prob_num/denom
        old_prob = old_prob_num/denom
        r = prob/(old_prob + 1e-10)

        return -K.mean(K.minimum(r * advantage, K.clip(r, min_value=1 - 1e-5, max_value=1 + 1e-5) * advantage))
    return loss


class AGENT:
    def __init__(self, nx, ny, s_link, sess, lstm):
        self.nx = nx[0]  #  Observation array length   nx = env.observation_space.shape[0]
        self.nx_lidar = 10
        self.nx_obs = 14
        self.ny = ny[0]  #   Action space length       ny = env.action_space.n
        self.lr_actor = 0.0001 # Actor_Learning rate
        self.lr_critic = 0.0002 # Critic_Learning rate
        self.gamma = 0.99
        self.alpha = 0.1
        self.s_link =s_link
        self.sess = sess
        self.lstm = lstm
        #self.deck = deque(maxlen=4000)

        self.e = 1.0
        self.e_= 0.01
        self.dc= 0.99
        self.tau = 0.001
        self.weight_decay = 0.001
        self.los = []
        self.layers=[256,128]

        # Network init_functions
        self.k_r = l2(self.weight_decay)
        self.initializer = "glorot_uniform"
        self.final_initializer = RandomUniform(minval = -0.003, maxval = 0.003)
        self.parameters={'lr': self.lr_actor, 'gamma':self.gamma,
                         'alpha':self.alpha, 'tau':self.tau,'dc':self.dc}

        if os.path.isfile('./' + self.s_link):
            self.message = ["\nLOAD existing keras model....", self.model.summary()]
            self.model = load_model(self.s_link)
        else:
            self.message = '\nBuilding New Model >_'
            ### Setting Actor #########################################################################
            self.actor_lidar_input, self.actor_state_input, self.actor_local = self.Actor()
            _,_, self.actor_target = self.Actor()
            # Take actor grafients################################
            self.actor_critic_grads = tf.placeholder(tf.float32, [None,self.ny])
            actor_local_weights = self.actor_local.trainable_weights
            self.actor_grads = tf.gradients(self.actor_local.output, actor_local_weights, -self.actor_critic_grads)
            grads = zip(self.actor_grads, actor_local_weights)
            self.optimize = tf.train.AdamOptimizer(self.lr_actor).apply_gradients(grads)
            ######################################################################
            ### Setting Critic #################################################
            self.critic_lidar_input, self.critic_state_input, self.critic_action_input, self.critic_local = self.Critic()
            _,_, _, self.critic_target = self.Critic()
            # Take critic gradients for actor training
            self.critic_grads = tf.gradients(self.critic_local.output,  self.critic_action_input)

            # Initialize variables
            self.sess.run(tf.global_variables_initializer())

        # Empty the lists
        self.ep_rewards, self.ep_obs, self.ep_act, self.ep_obs_new, self.ep_flags=[], [], [],[], []

    def choose_action(self,observation):
        state = observation[0][:14].reshape((1,14))
        lidar = observation[0][14:].reshape((1,10))
        # Use epsilon-greedy algorithm
        if np.random.rand() <= self.e :
            # epsilon Greedy             
            if self.e >= self.e_:
                self.e *= self.dc
                epsilon.append(self.e)
            action_=np.random.uniform(-1,1,4)
            return action_

        action = self.actor_local.predict([DUMMY_ACTION,DUMMY_VALUE,lidar,state])
        return action

    def storing(self, observation, action, reward, observation_new, flags ):
        # Storing for replay-expirience
        #self.deck.append((observation, action, reward, observation_new, flags ))
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
        old_act = Input(shape=(4,))
        advantage = Input(shape=(1,))
        #
        lidar_input = Input(shape=(self.nx_lidar,))
        lidar_conv = Dense(self.layers[0], activation='relu', kernel_regularizer=self.k_r)(lidar_input)
        #pool = MaxPooling1D(4)(lidar_conv)
        #flat = Flatten()(lidar_conv)
        lidar_conv = Dropout(0.05)(lidar_conv)
        #
        state_input = Input(shape=(self.nx_obs,))
        state_h1 = Dense(self.layers[0], activation='relu', kernel_regularizer=self.k_r)(state_input)
        #state_h1 = GaussianNoise(1.0)(state_h1)
        #gauss = Flatten()(gauss)
        #
        merged = Concatenate()([lidar_conv,state_h1])
        if self.lstm:
            merged = Reshape((self.layers[0]*2,1))(merged)
            merged_lstm = LSTM(self.layers[1],activation='relu',kernel_regularizer=self.k_r ,\
                                kernel_initializer=self.initializer)(merged)
        else:
            merged_lstm = Dense(self.layers[1],activation='relu',kernel_regularizer=self.k_r ,\
                                kernel_initializer=self.initializer)(merged)

        output = Dense(self.ny, activation='tanh', kernel_regularizer=self.k_r,\
                       kernel_initializer=self.final_initializer)(merged_lstm)

        ##############
        model = Model(input=[old_act,advantage,lidar_input,state_input], output=output)
        adam = Adam(lr=self.lr_actor, decay=0.5)
        model.compile(loss=actor_loss(advantage,old_act), optimizer=adam)
        return lidar_input,state_input, model

    def Critic(self):
        lidar_input = Input(shape=(self.nx_lidar,))
        lidar_conv = Dense(self.layers[0], activation='relu', kernel_regularizer=self.k_r,\
                           kernel_initializer=self.initializer)(lidar_input)
        #flat= Flatten()(lidar_conv)
        lidar_conv =Dropout(0.5)(lidar_conv)
        #
        #
        state_input = Input(shape=(self.nx_obs,))
        state_h1 = Dense(self.layers[0], activation='relu', kernel_regularizer=self.k_r,\
                         kernel_initializer=self.initializer)(state_input)
        #state_h1 = Flatten()(state_h1)
        state_h1 = Dropout(0.5)(state_h1)
        #
        merge1 = Concatenate()([lidar_conv,state_h1])
        #merged_dense = Dense(self.layers[0], activation='relu')(merge1)
        #
        action_input = Input(shape=(self.ny,))
        action_h1    = Dense(self.layers[0], activation='relu',kernel_regularizer=self.k_r,\
                             kernel_initializer=self.initializer)(action_input)
        action_h1 = Dropout(0.5)(action_h1)
        #
        merge2 = Concatenate()([merge1,action_h1])
        #merge2 = Reshape((self.layers[0]*3,1))(merge2)
        #
        if self.lstm:
            merge2 = Reshape((self.layers[0]*3,1))(merge2)
            merged_lstm = LSTM(self.layers[1],activation='relu',kernel_regularizer=self.k_r ,\
                               kernel_initializer=self.initializer)(merge2)
        else:
            merged_lstm = Dense(self.layers[1],activation='relu',kernel_regularizer=self.k_r ,\
                                kernel_initializer=self.initializer)(merge2)
        merged_lstm = Dropout(0.5)(merged_lstm)
        #
        output= Dense(1,activation='linear', kernel_regularizer=self.k_r,\
                      kernel_initializer=self.final_initializer)(merged_lstm)
        ##############
        model  = Model(input=[lidar_input,state_input,action_input], output=output)
        adam  = Adam(lr=self.lr_critic, decay=0.6)
        model.compile(loss="mse", optimizer=adam)
        return lidar_input,state_input, action_input, model

    def _train_critic(self, lidar, state, act, Q_target):
        for _ in range(epc):
            self.critic_local.train_on_batch([lidar,state,act], [Q_target])
        #self.critic_local.fit(x=[lidar,state,act],y=Q_target, verbose=0, epochs=10)

    def _train_actor(self, lidar,state,act,rew):
        current_reward = self.critic_target.predict([lidar,state,act])[0]
        advantage = rew - current_reward

        predicted_action = self.actor_local.predict([act,advantage,lidar,state])
        grads = self.sess.run(self.critic_grads, feed_dict = {
                self.critic_lidar_input : lidar,
                self.critic_state_input: state,
                self.critic_action_input: predicted_action})[0]

        for _ in range(epc):
            self.sess.run(self.optimize, feed_dict={
                    self.actor_lidar_input: lidar,
                    self.actor_state_input: state,
                    self.actor_critic_grads: grads})

    def update_target(self):
        """Soft update model parameters.
        ?_target = ?*?_local + (1 - ?)*?_target"""
        actor_local_weights  = self.actor_local.get_weights()
        actor_target_weights =self.actor_target.get_weights()
        #
        for i in range(len(actor_target_weights)):
            actor_target_weights[i] = self.tau*actor_local_weights[i] + (1-self.tau)*actor_target_weights[i]
        self.actor_target.set_weights(actor_target_weights)
        ##########################################################
        critic_local_weights  = self.critic_local.get_weights()
        critic_target_weights = self.critic_target.get_weights()
        #
        for i in range(len(critic_target_weights)):
            critic_target_weights[i] = self.tau*critic_local_weights[i] + (1-self.tau)*critic_target_weights[i]
        self.critic_target.set_weights(critic_target_weights)

    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        for i in range(len(rewards)-2,-1,-1):
            discounted_rewards[i] = rewards[i] + rewards[i+1]*self.gamma**i
        return discounted_rewards

    def create_batch(self,BATCH):
        state, lidar, state_new ,lidar_new, current_reward, advantage =[],[],[],[],[], []
        discounted_rewards = self.discount_rewards(self.ep_rewards[0:BATCH])
        traj = [self.ep_obs[0:BATCH], self.ep_act[0:BATCH], discounted_rewards,\
                       self.ep_obs_new[0:BATCH], self.ep_flags[0:BATCH]]
        rew = np.array(traj[2]).reshape(-1,1)
        act = np.array(traj[1])
        for observation, obs_new in zip(traj[0],traj[3]):
            state.append(observation[0][:14].reshape((14,)))
            lidar.append(observation[0][14:].reshape((10,)))
            state_new.append(obs_new[0][:14].reshape((14,)))
            lidar_new.append(obs_new[0][14:].reshape((10,)))

        state = np.array(state)
        lidar  = np.array(lidar)
        state_new  = np.array(state_new)
        lidar_new  = np.array(lidar_new)

        current_reward = self.critic_target.predict([lidar,state,act])
        advantage = rew - current_reward
        target_action = self.actor_target.predict([act,advantage,lidar_new,state_new])
        future_reward = self.critic_target.predict([lidar_new,state_new, target_action])
        Q_target =self.alpha*(rew + self.gamma * future_reward - current_reward)
        #print('cr: ' + str(current_reward.shape) + 'target: ' + str(target_action.shape) + 'fut: ' + str(future_reward.shape) + 'adv: ' + str(advantage.shape))
        return lidar, state, act, Q_target , advantage, rew

    def TRAIN(self, BATCH):
        lidar, state, act, Q_target , advantage, rew = self.create_batch(BATCH)
        start_train = time.time()
        # Train Critic
        #print('lidar:' + str(lidar.shape) + str(state.shape) + 'act: ' + str(act.shape) +\
        #      'Q_target: ' + str(Q_target.shape) + 'adv: ' + str(advantage.shape)+ 'rew: ' + str(rew.shape) )
        self._train_critic(lidar, state, act, Q_target) # TRain the network critic
        # Train Actor
        self._train_actor(lidar,state,act,rew)
        # Update Weights 
        self.update_target() # Update the netokr local and target weights for actor AND critic
        end = time.time() - start_train
        return end

    def Clear(self):

        self.ep_rewards, self.ep_obs, self.ep_act, self.ep_obs_new, self.ep_flags=[], [], [],[], []

if __name__ == '__main__':

    rendering = input("Visualize rendering ? [y/n]:  ")
    s_link = "BipedalWalker_model.h5"
    RENDER_REWARD_MIN = 5000
    RENDER_ENV = False
    if rendering == 'y': RENDER_ENV = True  #flag for rendering the environment       
    directory = 'Monitor'
    env = gym.make('BipedalWalker-v2')  #video_callable=lambda episode_id: episode_id%10==0
    #env = gym.wrappers.Monitor(env, directory+'lala3',  force=True)
    # Network Parameters
    nx = env.observation_space.shape # network-input  
    ny = env.action_space.shape # network-out
    sess = tf.Session()
    K.set_session(sess)
    agent = AGENT(nx,ny, s_link, sess, lstm)

    rewards_over_time = []
    error = []
    epsilon = []
    rew_var = []
    rew_mean = []
    mean_100 = []
    seed = np.random.seed(1)

    print('-----------------------------------')
    print(agent.message)
    print("Environment Observation_space: ", env.observation_space)
    print("Environment Action_space: ", env.action_space)
    print("Num of Episodes: %i | Batch: %i | Epochs %i" % (EPISODES,BATCH,epc))
    if agent.lstm: print("LSTM layer is ON!")
    print('-----------------------------------')
    print("\n:::::Algorithm_Parameters::::::")
    print(list(agent.parameters.items()))
    print('\n+++++++++++++++++++++++++++++++++')
    print('BipedalWalker-v2 Starts... Enjoy! >_')
    print('+++++++++++++++++++++++++++++++++')
    w = 0
    traj = 0

    # Start running the SIMULATION      
    for i in range(EPISODES):
        observation = env.reset()
        observation = observation.reshape(1,-1)
        start = time.time()
        counter = 0
        noise=OrnsteinUhlenbeckProcess(theta=.15, mu=0., sigma=.1,size=4)
        rew =0
        # Start EPISODE
        while True:
            if RENDER_ENV:
                env.render()
            counter +=1
            traj += 1
            action = agent.choose_action(observation)
            action = np.clip(action+noise.generate(counter), -1,1)
            action = action.reshape((4,))
            observation_new, reward, flag, inf = env.step(action)
            observation_new = observation_new.reshape((1,24))
            rew += reward
            # Store new information
            agent.storing(observation, action, reward, observation_new, flag)
            observation = observation_new

            # Measure the time
            end = time.time()
            # Set time constrain: 40secs stop episode
            time_space = end - start
            if time_space > 200:
                flag = True

            if flag==True:
                # Append rewards history
                rewards_over_time.append(rew)
                rew_mean.append(np.mean(agent.ep_rewards))
                rew_var.append(np.var(agent.ep_rewards))
                max_reward = np.max(rewards_over_time)
                episode_max = np.argmax(rewards_over_time)
                # Output results and terminate episode                
                # Winning Statement
                if rew >=300 :
                    w = w + 1
                    agent.save(s_link)
                # Print output per 100 episodes
                if i % arg == 0:
                    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                    print("Episode: %i | Steps: %i | Reward: %.3f | Time: %.2f" % (i,counter,rew,time_space))
                    print("Maximum Reward: %.2f   on Episode: %i | Times win: %i" % (max_reward, episode_max,w))

                if i % 100==0:
                    print("Mean reward of the past 100 episodes: ", str(np.mean(rewards_over_time[-100:])))
                    mean_100.append(np.mean(rewards_over_time[-100:]))
                    f = open('results.txt','a')
                    f.write('\n' + str(np.mean(rewards_over_time[-100:])))
                    f.close()

                if i % 100 <= 20 and rew >= -1.0:
                    print('Trainining Good Episode.. >_')
                    training_time = agent.TRAIN(counter)
                    print('Training Time %.2f' % training_time)

                if traj >= BATCH:
                    print('Training... >_')
                    training_time = agent.TRAIN(BATCH)
                    #agent.deck.clear()
                    traj = 0
                    agent.Clear()
                    print('Training Time %.2f' % training_time)
                break
            # END_IF
        # END_WHILE
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

                                                                                                                                                        
