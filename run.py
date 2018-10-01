from Experiment import *

# Indicate experiment name
experiment_name = 'softmax_vs_egreedy_vs_planning'

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
        'lambda': 0,
        'exploration policy': 'softmax',
        'learn model': False,
        'discount rate': 0.9
    },
    {
        'learning rate': 0.1,
        'value update': 'TD',
        'lambda': 0,
        'exploration policy': 'softmax',
        'learn model': True,
        'planning steps': 5
    },
    {
        'learning rate': 0.1,
        'value update': 'TD',
        'lambda': 0,
        'exploration policy': 'e-greedy',
        'learn model': True,
        'planning steps': 5
    }
]

# Environment spec variable
environments = [
    {
        'maze name': '5level_binary_maze',
        'number of levels': 5,
        'reward locations': {(4, 4): 1}
    }
]


# Run experiment
Experiment = Experiment(name = experiment_name,
                        environments = environments,
                        agents = agents,
                        nb_episodes = 100,
                        nb_trials = 20)
Experiment.run_experiment()
