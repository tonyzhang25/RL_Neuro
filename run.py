from Experiment import *

# Indicate experiment name
experiment_name = 'novelty_agent3'

# Agent spec variable
lr = 0.5
agents = [
    {
        'learning rate': lr,
        'value update': 'TD',
        'lambda': 0,
        'exploration policy': 'e-greedy',
        'epsilon': 0,
        'learn model': False,
        'discount rate': 0.9,
        'add exploration bonus': True,
        'reduction': 0.5
    },
    # {
    #     'learning rate': lr,
    #     'value update': 'TD',
    #     'lambda': 0,
    #     'exploration policy': 'e-greedy',
    #     'epsilon': 0,
    #     'learn model': False,
    #     'discount rate': 0.9,
    #     'add exploration bonus': True,
    #     'reduction': 1
    # },
    # {
    #     'learning rate': lr,
    #     'value update': 'TD',
    #     'lambda': 0.5,
    #     'exploration policy': 'e-greedy',
    #     'epsilon': 0,
    #     'learn model': False,
    #     'discount rate': 0.9,
    #     'add exploration bonus': True,
    #     'reduction': 1
    # },
    # {
    #     'learning rate': lr,
    #     'value update': 'TD',
    #     'lambda': 0,
    #     'exploration policy': 'e-greedy',
    #     'epsilon': 0,
    #     'learn model': False,
    #     'discount rate': 0.9,
    #     'add exploration bonus': True,
    #     'reduction': 0.1
    # },
    # To test additional agents, insert here
]

# Environment spec variable
env = [
    {
        'maze type': 'Spatial',
        'maze name': 'MazeV4',
        'map': [
                'o-o-o   o-o-o   o-o-o   o-o-o',
                '  |       |       |       |  ',
                '  o-o-o-o-o       o-o-o-o-o  ',
                '  |   |   |       |   |   |  ',
                'o-o-o o o-o-o   o-o-o o o-o-o',
                '      |               |      ',
                '      o-o-o-o-o-o-o-o-o      ',
                '      |       |       |      ',
                'o-o-o o o-o-o o o-o-o o o-o-o',
                '  |   |   |   |   |   |   |  ',
                '  o-o-o-o-o   o   o-o-o-o-o  ',
                '  |       |   |   |       |  ',
                'o-o-o   o-o-o o o-o-o   o-o-o',
                '              |              ',
                'o-o-o-o-o-o-o-o              ',
                '              |              ',
                'o-o-o   o-o-o o o-o-o   o-o-o',
                '  |       |   |   |       |  ',
                '  o-o-o-o-o   o   o-o-o-o-o  ',
                '  |   |   |   |   |   |   |  ',
                'o-o-o o o-o-o o o-o-o o o-o-o',
                '      |       |       |      ',
                '      o-o-o-o-o-o-o-o-o      ',
                '      |               |      ',
                'o-o-o o o-o-o   o-o-o o o-o-o',
                '  |   |   |       |   |   |  ',
                '  o-o-o-o-o       o-o-o-o-o  ',
                '  |       |       |       |  ',
                'o-o-o   o-o-o   o-o-o   o-o-o',],
        'start position': (7, 0),
        # 'reward locations': {(8, 14): 1},  # one reward, no switch
        'reward locations': {},  # no rewards
        'change reward location': False,
        'allow reversals': True
    },
    # {
    #     'maze type': 'Binary',
    #     'maze name': '6level_binary_maze',
    #     'number of levels': 6,
    #     # 'reward locations': {(5, 17): 10}, # no reward switching
    #     'reward locations': {}, # no reward switching
    #     'change reward location': False,
    #     # 'reward locations': {0: {(3,2): 100}, 50: {(3,3): 100}}, # change-episode, location (index from 0), reward
    #     'allow reversals': True
    # },
    # To test additional environments, insert here (must match with number of agents)
]


# Run experiment
Experiment = Experiment(name = experiment_name,
                        environments = env,
                        agents = agents,
                        nb_episodes = 3000,
                        nb_trials = 50)