from Experiment import *

# Indicate experiment name
experiment_name = 'experiment'

# Agent spec variable
lr = 0.2
agents = [
    {
        'learning rate': lr,
        'value update': 'TD',
        'lambda': 0,
        'exploration policy': 'e-greedy',
        'epsilon': 0,
        'learn model': False,
        'discount rate': 0.9,
        'add exploration bonus': False,
        'reduction': 1
    },
    {
        'learning rate': lr,
        'value update': 'TD',
        'lambda': 0,
        'exploration policy': 'e-greedy',
        'epsilon': 0.3,
        'learn model': False,
        'discount rate': 0.9,
        'add exploration bonus': False,
        'reduction': 1
    },
    # To test additional agents, insert here
]

# Environment spec variable
environments = [
    {
        'maze name': '4level_binary_maze',
        'number of levels': 4, # last level is terminal
        # 'reward locations': {(3, 2): 100}, # index from 0
        'change reward location': True,
        'reward locations': {0: {(3,2): 100}, 100: {(3,3): 100}}, # change-episode, location, reward
        'allow reversals': False
    },
    # To test additional environments, insert here (must match with number of agents)
]


# Run experiment
Experiment = Experiment(name = experiment_name,
                        environments = environments,
                        agents = agents,
                        nb_episodes = 200,
                        nb_trials = 1000)