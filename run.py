from Experiment import *

# Indicate experiment name
experiment_name = 'TD_v_MC'

# Agent spec variable
agents = [
    {
        'learning rate': 0.1,
        'value update': 'TD',
        'lambda': 0,
        'exploration policy': 'e-greedy',
        'learn model': False,
        'discount rate': 0.9
    },
    {
        'learning rate': 0.1,
        'value update': 'TD',
        'lambda': 0.5,
        'exploration policy': 'e-greedy',
        'learn model': False,
        'discount rate': 0.9
    },
    {
        'learning rate': 0.1,
        'value update': 'TD',
        'lambda': 1,
        'exploration policy': 'e-greedy',
        'learn model': False,
        'discount rate': 0.9
    },
    {
        'learning rate': 0.1,
        'value update': 'MC',
        'exploration policy': 'e-greedy',
        'learn model': False,
        'discount rate': 0.9
    },
]

# Environment spec variable
environments = [
    {
        'maze name': '5level_binary_maze',
        'number of levels': 5, # last level is terminal
        'reward locations': {(4, 4): 1} # index from 0
    }
    # To test additional environments,
    # Insert here (must match with number of agents)
]


# Run experiment
Experiment = Experiment(name = experiment_name,
                        environments = environments,
                        agents = agents,
                        nb_episodes = 100,
                        nb_trials = 10)
Experiment.run_experiment()
