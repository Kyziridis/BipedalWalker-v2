# BipedalWalker-v2

### [OpenAI](https://gym.openai.com/envs/BipedalWalkerHardcore-v2/) game.<br>
##### The repo is under development.
#### Methods

1. Deep-Q-Learning Networks based on Reinforcement-Learning<br>
2. Experiment with Recurrent Deterministic Policy Gradient Methods
3. General Research about state-of-the-art reinforcement learning algorithms
4. Actor-Critic Networks 
5. LSTM
<br>

**Instructions**<br>

1. In order to run both scripts you need to install openAI-gym lib.

2. For gym install follow the next steps:
```	
    $ pip install --upgrade pip
    $ git clone https://github.com/openai/gym.git
    $ cd gym
    $ pip install -e '.[box2d]'

    # test that it worked 
    $ python >>> import gym >>> gym.make('BipedalWalker-v2')

    # The above instructions are for linux systems. 
    # We do not use proprietary software so we do not have instructions for Windows or MacOS.  
```
You can visit [openAI gym](https://gym.openai.com/envs/#box2d) in order to inspect this 
kind of environments.<br>	

3. Other dependencies : Keras, matplotlib, numpy, random, tensorflow

4. Do not use any Python-IDE to run the code if you want to see 
the game-simulation (environment-render). 
Run the code in a console or a terminal by typping: `$ python script-name.py`
