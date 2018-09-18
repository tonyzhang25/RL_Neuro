'''
Class for all visualizations. Pass in Interact class, which contains all history variables
related to agent's actions, state and reward observations.
This file is currently designed to work with Binary_Maze.py
'''

import numpy as np
import matplotlib.pyplot as plt
import os, sys, glob


class Analysis:

    def __init__(self, Maze, Interact):
        self.Map = Maze
        # History of [s_t, a_t, s_t+1] tupples
        self.state_action_history = Interact.state_act_history_trials
        # History of combined Interact output to Agent
        self.state_obs_history = Interact.state_obs_history_trials
        self.value_history = Interact.agent_qvalues_history_trials
        self.init_output_path('data/analysis/')

    def init_output_path(self, path):
        self.output_path = path
        if not os.path.exists(path):
            os.mkdir(path)

    def visualize(self):
        print('\nAnalyzing experiments..')
        self.visualize_reward_all_episodes()
        self.visualize_cumulative_reward()
        self.visualize_state_values()

    def visualize_reward_all_episodes(self):
        for trial_nb, trial_i in enumerate(self.state_obs_history):
            nb_episodes = len(trial_i)
            reward_record = []
            for episode_nb, episode_j in enumerate(trial_i):
                reward = episode_j[-1][-2]
                reward_record.append(reward)
            plt.figure(figsize = (10,0.8))
            x = np.arange(nb_episodes)
            plt.scatter(x + 1, reward_record,
                        s = 20,
                        facecolor = (0,0,0,0),
                        linewidths = 0.4,
                        edgecolor = 'C3')
            plt.xlabel('Episode')
            plt.xlim(0, nb_episodes+1)
            plt.ylim(-0.5,1.5)
            plt.ylabel('Reward')
            plt.savefig(self.output_path + self.Map.name + '_t' +
                        str(trial_nb) + '_reward_record.png',
                        dpi = 500, bbox_inches = 'tight')
            plt.close()
        print('Reward log visualized.')

    def visualize_cumulative_reward(self):
        for trial_nb, trial_i in enumerate(self.state_obs_history):
            nb_episodes = len(trial_i)
            reward_record = []
            for episode_nb, episode_j in enumerate(trial_i):
                reward = episode_j[-1][-2]
                if episode_nb > 0:
                    reward_record.append(reward_record[episode_nb-1] + reward)
                else:
                    reward_record.append(reward)
            plt.figure(figsize = (4,3))
            x = np.arange(nb_episodes)
            plt.plot(x+1, reward_record,
                        color = 'C2',
                        linewidth = 1.5)
            ax = plt.axes()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.xlabel('Episode')
            plt.xlim(1, nb_episodes+1)
            plt.ylim(0,)
            plt.ylabel('Cumulative Reward')
            plt.savefig(self.output_path + self.Map.name + '_t' +
                        str(trial_nb) + '_cumulative_reward.png',
                        dpi = 500, bbox_inches = 'tight')
            plt.close()
        print('Cumulative reward plot visualized.')

    def visualize_state_values(self):
        '''
        Visualize q value changes over episodes and trials.
        NOTE: this dictionary object may contain incomplete state access history,
        due to the nature of the exploration policy.
        '''
        for trial_nb, trial_i in enumerate(self.value_history):
            visited_stateactions = list(trial_i[-1])
            visited_stateactions.sort()
            nb_episodes = len(trial_i)
            value_matrix = np.zeros((len(visited_stateactions), nb_episodes))
            for episode_nb, episode_j in enumerate(trial_i):
                for row_nb, stateaction_k in enumerate(visited_stateactions):
                    if stateaction_k in episode_j:
                        value_matrix[row_nb, episode_nb] = episode_j[stateaction_k]
            # plot
            plt.figure(figsize = (6,4))
            plt.pcolor(value_matrix)
            plt.ylabel('Agent State-Action Value')
            plt.xlabel('Episode')
            plt.colorbar()
            plt.savefig(self.output_path + self.Map.name + '_t' + str(trial_nb) +
                        '_value_across_learning.png', dpi = 500, bbox_inches = 'tight')
            plt.close()
            self.value_matrix = value_matrix
        print('Value learning visualized.')




