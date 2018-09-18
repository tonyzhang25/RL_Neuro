import numpy as np
from Binary_Maze import *
from Interact import *
from Agent import *
from Analysis import *
import matplotlib.pyplot as plt


mazeName = '5level_binary_maze'
nb_levels = 5 # total levels of maze: e.g. 5 levels = 4 levels of branching.

reward_location = {(nb_levels - 1, 1): 1} # can be multiple

env_properties = {
    'init_state': 0,
    'episode_termination': 'environment termination states'
}

agent_properties = {
    'learning rate': 0.2,
    'value update': 'TD',
    'lambda': 0,
    'exploration policy': 'e-greedy',
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


nb_episodes = 100
nb_trials = 1

for trial in range(nb_trials):
    # Start trial
    TD_Agent = Agent(agent_properties) # init / reset agent
    print()
    for episode in range(nb_episodes):
        # Start episode
        obs = Session.init_episode()
        action = TD_Agent.step(obs)
        termination = False
        while termination == False:  # assert termination condition = False
            obs = Session.step(action)
            action = TD_Agent.step(obs)
            termination = obs[-1]
        print('| TRIAL: ' + str(trial+1) +
              ' | Episode: '+str(episode+1) +
              ' | Reward = '+str(obs[-2]) + ' |')
        # End of episode processing
        Qvalues = TD_Agent.Qfunction  # obtain q values for analysis
        Session.add_value_to_record(Qvalues)
    # End of trial processing
    Session.process_trial()


## Analysis

Analyze = Analysis(FiveLevelMaze, Session)
Analyze.visualize()


