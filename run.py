from Experiment import *

# Indicate experiment name
experiment_name = 'exploration_bonus_greedy_e0_4level_noreversal'

# Agent spec variable
lr = 0.2
agents = [
    {
        'learning rate': lr,
        'value update': 'TD',
        'lambda': 0.1,
        'exploration policy': 'e-greedy',
        'epsilon': 0.1,
        'learn model': False,
        'discount rate': 0.9,
        'add exploration bonus': True,
        'reduction': 1
    },
    {
        'learning rate': lr,
        'value update': 'TD',
        'lambda': 0.1   ,
        'exploration policy': 'e-greedy',
        'epsilon': 0.5,
        'learn model': False,
        'discount rate': 0.9,
        'add exploration bonus': True,
        'reduction': 1
    },
    {
        'learning rate': lr,
        'value update': 'TD',
        'lambda': 0.1,
        'exploration policy': 'e-greedy',
        'epsilon': 0,
        'learn model': False,
        'discount rate': 0.9,
        'add exploration bonus': False
    },
    # To test additional agents, insert here
]

# Environment spec variable
environments = [
    {
        'maze name': '4level_binary_maze',
        'number of levels': 4, # last level is terminal
        'reward locations': {(3, 2): 100}, # index from 0
        'allow reversals': True
    },
    # # To test additional environments, insert here (must match with number of agents)
    # {
    #     'maze name': '5level_binary_maze',
    #     'number of levels': 5,  # last level is terminal
    #     'reward locations': {(4, 4): 1},  # index from 0
    #     'allow reversals': True
    # }
]


# Run experiment
Experiment = Experiment(name = experiment_name,
                        environments = environments,
                        agents = agents,
                        nb_episodes = 50,
                        nb_trials = 1000)