import numpy as np
from Binary_Maze import *
from Interact import *
from Agent import *
import matplotlib.pyplot as plt
from scipy import stats



mazeName = '5level_binary_maze'
nb_levels = 5

reward_location = {(4,0): 1}

env_properties = {
    'init_state': 0,
    'episode_termination': 'environment termination states'
}

agent_properties = {
    'learning rate': 0.2,
    'value update': 'TD',
    'lambda': 0,
    'exploration policy': 'softmax',
    'learn model': False,
    'discount rate': 0.9
}


# Initialize new Maze object
FiveLevelMaze = Maze(mazeName,
                     nb_levels = nb_levels,
                     reward_location = reward_location)

# Connect environment object to Interact
Session = Interact(Map = FiveLevelMaze,
                   properties = env_properties)


nb_episodes = 500
nb_trials = 1
for trial in range(nb_trials):
    # Start trial
    TD_Agent = Agent(agent_properties) # init / reset agent
    print()
    for episode in range(nb_episodes):
        # Start episode
        obs = Session.init_episode()
        action = TD_Agent.step(obs)
        while obs[-1] == False:  # assert termination condition = False
            obs = Session.step(action)
            action = TD_Agent.step(obs)
        print('| TRIAL: ' + str(trial+1) +
              ' | Episode: '+str(episode+1) +
              ' | Reward = '+str(obs[-2]) + ' |')


