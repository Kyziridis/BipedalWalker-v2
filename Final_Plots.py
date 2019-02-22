#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 16:47:30 2018

@author: dead
"""

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import os
import pandas as pd
os.chdir("final_results/")



e4 = np.load("mean100_44.npy")
e666 = np.load("666mean100.npy")
e66 = e666[:100]
ekalo = np.load("kalomean100.npy")


####DDPG##############
print("Max Average Reward |DDPG+PPO: %.3f | RDDPG: %.3f | DDPG: %.3f" %(max(ekalo),max(e4),max(e66)))
lala = np.arange(1,len(e4)+1)*100
plt.figure(figsize=(10,8))
plt.tick_params(size=5, labelsize=16.5)
plt.plot(lala,ekalo, label="DDPG+PPO")
plt.plot(lala,e4, label="RDDPG")
plt.plot(lala,e66, label="DDPG")
plt.xticks(np.arange(0,101,10)*100)
plt.xlabel("Episodes", fontsize = 17)
plt.ylabel("Mean Reward", fontsize = 17)
plt.title('Average Reward per 100 episodes', fontsize=17)
plt.legend(loc=0, fontsize=15)
plt.savefig("DDPG_PPO_Results.png")
######################
e44 = np.load("e44mean100.npy")#
lamean = np.load("lalamean100.npy")#
print("Max Average Reward |Big-DDPG: %.3f | Small-DDPG: %.3f " %(max(lamean),max(e44[0:100])))

lala = np.arange(1,len(lamean)+1)*100
plt.figure(figsize=(10,8))
plt.tick_params(size=5, labelsize=16.5)
plt.plot(lala,lamean, label="Big-DDPG")
plt.plot(lala,e44[0:100], label="Small-DDPG")
plt.xticks(np.arange(0,101,10)*100)
plt.xlabel("Episodes", fontsize = 17)
plt.ylabel("Mean Reward", fontsize = 17)
plt.title('Average Reward per 100 episodes', fontsize=17)
plt.legend(loc=0, fontsize=15)
plt.savefig("DDPG_Results.png")

# Only small ddpg
lala = np.arange(1,len(e44)+1)*100
plt.figure(figsize=(10,8))
plt.tick_params(size=5, labelsize=16.5)
plt.plot(lala,e44, label="Small-DDPG")
plt.xticks(np.arange(0,501,100)*100)
plt.xlabel("Episodes", fontsize = 17)
plt.ylabel("Mean Reward", fontsize = 17)
plt.title('Average Reward per 100 episodes', fontsize=17)
plt.legend(loc=0, fontsize=15)
plt.savefig("Small_ddpg.png")


#########################################################################
### PPO___#####################
# Average 100 Rewarss PPO

keras = np.load("mean100_k.npy")
tf = np.load("mean100_tf.npy")
print("Max Average Reward |keras: %.3f | tf: %.3f" %(max(keras),max(tf)))
lala = np.arange(1,len(keras)+1)*100
plt.figure(figsize=(10,8))
plt.tick_params(size=5, labelsize=16.5)
plt.plot(lala,tf[:-1], label="Boosted PPO")
plt.plot(lala,keras, label="Simple PPO")
#plt.plot(lala,lolo,label="DDPG")
plt.xticks(np.arange(0,101,10)*100)
plt.xlabel("Episodes", fontsize = 17)
plt.ylabel("Mean Reward", fontsize = 17)
plt.title('Average Reward per 100 episodes', fontsize=17)
plt.legend(loc=0, fontsize=15)
plt.savefig("PPO_avg.png")
######################################################
######################################################
 # Rewards
keras_rew = pd.read_csv("keras.csv")  
print('Maximum Total Reward: ', max(keras_rew.iloc[:,2]))

plt.figure(figsize=(9,7))
plt.tick_params(size=5, labelsize=15)
plt.plot(keras_rew.iloc[:,2])
plt.xticks(np.arange(0,1100,200), np.arange(0,1100,200)*10)
plt.xlabel("Episodes", fontsize = 14)
plt.ylabel("Episode Total Reward", fontsize = 14)
plt.title('PPO Episode Rewards', fontsize=15)
plt.savefig("PPO_Rewards.png")

tf_rew = pd.read_csv("tf.csv")
print('Maximum Total Reward: ', max(tf_rew.iloc[:,2]))

plt.figure(figsize=(9,7))
plt.tick_params(size=5, labelsize=15)
plt.plot(tf_rew.iloc[:,2])
plt.plot(np.repeat(300,1000), '--' )
plt.xticks(np.arange(0,1100,200), np.arange(0,1100,200)*10)
plt.xlabel("Episodes", fontsize = 14)
plt.ylabel("Episode Total Reward", fontsize = 14)
plt.title('Boosted PPO Episode Rewards', fontsize=15)
plt.savefig("Boost_PPO_Rewards.png")
#########################################
########################################













