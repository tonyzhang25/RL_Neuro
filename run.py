from Experiment import *

# Indicate experiment name
experiment_name = 'Same_TD_Agent_Diff_Env'

# Agent spec variable
agents = [
    {
        'learning rate': 0.1,
        'value update': 'TD',
        'lambda': 0.7,
        'exploration policy': 'e-greedy',
        'learn model': False,
        'discount rate': 0.9
    },
    {
        'learning rate': 0.1,
        'value update': 'TD',
        'lambda': 0.7,
        'exploration policy': 'e-greedy',
        'learn model': False,
        'discount rate': 0.9
    },
    # To test additional environments, insert here
]

# Environment spec variable
environments = [
    {
        'maze name': '5level_binary_maze',
        'number of levels': 5, # last level is terminal
        'reward locations': {(4, 4): 1}, # index from 0
        'allow reversals': False
    },
    # To test additional environments, insert here (must match with number of agents)
    {
        'maze name': '5level_binary_maze',
        'number of levels': 5,  # last level is terminal
        'reward locations': {(4, 4): 1},  # index from 0
        'allow reversals': True
    }
]


# Run experiment
Experiment = Experiment(name = experiment_name,
                        environments = environments,
                        agents = agents,
                        nb_episodes = 100,
                        nb_trials = 5)