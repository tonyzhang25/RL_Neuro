'''
Class for all visualizations. Pass in Interact class, which contains all history variables
related to agent's actions, state and reward observations.
This file is currently designed to work with Binary_Maze.py
'''

import numpy as np
import matplotlib.pyplot as plt
import os, sys, glob
import math

class Analysis:
    '''
    These are trial-level visualizations. They do not work for comparing experiment-level stats.
    Cross session comparisons are implemented under Experiment.py
    '''

    def __init__(self, exp_path, sess_id, Maze, Interact):
        self.exp_path = exp_path
        self.sess_id = sess_id
        self.Map = Maze
        # History of [s_t, a_t, s_t+1] tupples
        self.state_action_history = Interact.state_act_history_trials
        # History of combined Interact output to Agent
        self.state_obs_history = Interact.state_obs_history_trials
        self.value_history = Interact.agent_qvalues_history_trials
        self.novelty_history = Interact.agent_novelty_history_trials
        self.init_sub_session_path()
        self.cumulative_rewards = [] # for all trials
        self.all_timesteps_trial = [] # for the entire trial

    def init_sub_session_path(self):
        # Make session folder
        self.sess_output_path = self.exp_path + '/sess_' + str(self.sess_id) + '/'
        if not os.path.exists(self.sess_output_path):
            os.mkdir(self.sess_output_path)

    def visualize(self, dpi = 300):
        print('\nAnalyzing session data..')
        self.compare_total_steps_till_reward(dpi)
        self.visualize_reward_all_episodes(dpi)
        self.visualize_final_states(dpi)
        self.log_reward(dpi = dpi)
        self.visualize_state_values(dpi)
        self.visualize_state_novelty(dpi)
        self.visualize_timesteps_per_episode(dpi)

    def visualize_reward_all_episodes(self, dpi, plot = False):
        for trial_nb, trial_i in enumerate(self.state_obs_history):
            nb_episodes = len(trial_i)
            reward_record = []
            for episode_nb, episode_j in enumerate(trial_i):
                reward = episode_j[-1][-2]
                reward_record.append(reward)
            if plot:
                plt.figure(figsize = (10,1))
                x = np.arange(nb_episodes)
                plt.scatter(x + 1, reward_record,
                            s = 20,
                            facecolor = (0,0,0,0),
                            linewidths = 0.6,
                            edgecolor = 'C3')
                plt.xlabel('Episode')
                plt.xlim(0, nb_episodes+1)
                plt.ylim(-1, max(reward_record) + 1)
                plt.ylabel('Reward')
                plt.savefig(self.sess_output_path + self.Map.name + '_t' +
                            str(trial_nb) + '_reward_record.png',
                            dpi = dpi, bbox_inches = 'tight')
            plt.close()
        print('Reward log visualized.')

    def visualize_final_states(self, dpi, plot = False):
        list_of_final_states = self.Map.states_by_level[-1]
        min_state = list_of_final_states[0]
        nb_final_states = len(list_of_final_states)
        nb_episodes = len(self.state_obs_history[0])
        for trial_nb, trial_i in enumerate(self.state_obs_history):
            visual_matrix = np.zeros((nb_final_states, nb_episodes))
            # extract last states
            for episode_nb, episode_j in enumerate(trial_i):
                state = episode_j[-1][1]
                visual_matrix[state - min_state, episode_nb] = 1
            # plot
            if plot:
                plt.figure(figsize = (7,3))
                plt.pcolor(visual_matrix, cmap=plt.cm.Blues,
                           edgecolors='white', linewidths=0.5)
                # put the major ticks at the middle of each cell
                plt.yticks(np.arange(visual_matrix.shape[0]) + 0.5, list_of_final_states)
                plt.ylabel('Final State Reached')
                plt.xlabel('Episode')
                plt.savefig(self.sess_output_path + self.Map.name + '_t' + str(trial_nb) +
                            '_FinalState.png', dpi = dpi, bbox_inches = 'tight')
                plt.close()
        print('Final State Visited visualized.')

    def visualize_timesteps_per_episode(self, dpi, plot = False):
        for trial_nb, trial_i in enumerate(self.state_obs_history):
            episodes_length = []
            for episode_nb, episode_j in enumerate(trial_i):
                episode_length = len(episode_j)
                episodes_length.append(episode_length)
            nb_episodes = len(trial_i)
            # plot
            if plot:
                plt.figure(figsize = (5,3.5))
                x = np.arange(nb_episodes) + 1
                plt.plot(x, episodes_length, color = 'C1', linewidth = 2)
                ax = plt.axes()
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                plt.xlabel('Episode')
                plt.xlim(1, nb_episodes)
                plt.ylabel('Time Steps Until Termination)')
                plt.savefig(self.sess_output_path + self.Map.name + '_t' +
                            str(trial_nb) + '_timesteps_per_episode.png',
                            dpi = dpi, bbox_inches = 'tight')
                plt.close()
            self.all_timesteps_trial.append(episodes_length)
        print('Timesteps per episode plotted.')

    def compare_total_steps_till_reward(self, dpi):
        ''' compute how many total steps it takes until first reward is encountered
        state_obs_history: trials, episodes, timesteps within episode
        '''
        self.steps = []
        for trial_nb, trial_i in enumerate(self.state_obs_history):
            steps_trial_i = 0
            for episode_nb, episode_j in enumerate(trial_i):
                for obs_t in episode_j:
                    reward = obs_t[2]
                    if reward > 0:
                        break
                    else:
                        steps_trial_i += 1
                if reward > 0: break
            self.steps.append(steps_trial_i)

    def log_reward(self, plot = False, dpi = 300):
        for trial_nb, trial_i in enumerate(self.state_obs_history):
            nb_episodes = len(trial_i)
            reward_record = []
            for episode_nb, episode_j in enumerate(trial_i):
                reward = episode_j[-1][-2]
                if episode_nb > 0:
                    reward_record.append(reward)
                    # changed
                else:
                    reward_record.append(reward)
            if plot:
                self.plot_reward(dpi, nb_episodes, reward_record, trial_nb)
                print('Trial-averaged rewards across episodes plotted.')
            self.cumulative_rewards.append(reward_record)

    def plot_reward(self, dpi, nb_episodes, reward_record, trial_nb):
        plt.figure(figsize=(4, 3))
        x = np.arange(nb_episodes)
        plt.plot(x + 1, reward_record,
                 color='C2',
                 linewidth=1.5)
        ax = plt.axes()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.xlabel('Episode')
        plt.xlim(1, nb_episodes)
        plt.ylim(0, )
        plt.ylabel('Reward')
        plt.savefig(self.sess_output_path + self.Map.name + '_t' +
                    str(trial_nb) + '_episode_reward.png',
                    dpi=dpi, bbox_inches='tight')
        plt.close()

    def visualize_state_values(self, dpi, plot = False):
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
            if plot:
                plt.figure(figsize = (6,4))
                plt.pcolor(value_matrix,
                           vmin=math.floor(np.min(value_matrix)),
                           vmax=math.ceil(np.max(value_matrix)))
                plt.ylabel('Agent State-Action Value')
                plt.xlabel('Episode')
                plt.colorbar()
                plt.savefig(self.sess_output_path + self.Map.name + '_t' + str(trial_nb) +
                            '_value_across_learning.png', dpi = dpi, bbox_inches = 'tight')
                plt.close()
            self.value_matrix = value_matrix
            # USEFUL FOR DEBUGGING: print(np.max(value_matrix))
        print('Value learning visualized.')

    def visualize_state_novelty(self, dpi, plot = False):
        '''
        Visualize q value changes over episodes and trials.
        NOTE: this dictionary object may contain incomplete state access history,
        due to the nature of the exploration policy.
        '''
        for trial_nb, trial_i in enumerate(self.novelty_history):
            visited_stateactions = list(trial_i[-1])
            visited_stateactions.sort()
            nb_episodes = len(trial_i)
            novelty_matrix = np.zeros((len(visited_stateactions), nb_episodes))
            for episode_nb, episode_j in enumerate(trial_i):
                for row_nb, stateaction_k in enumerate(visited_stateactions):
                    if stateaction_k in episode_j:
                        novelty_matrix[row_nb, episode_nb] = episode_j[stateaction_k]
            # plot
            if plot:
                plt.figure(figsize = (6,4))
                plt.pcolor(novelty_matrix,
                           vmin=math.floor(np.min(novelty_matrix)),
                           vmax=math.ceil(np.max(novelty_matrix)))
                plt.ylabel('Agent State-Action Novelty')
                plt.xlabel('Episode')
                plt.colorbar()
                plt.savefig(self.sess_output_path + self.Map.name + '_t' + str(trial_nb) +
                            '_novelty_across_learning.png', dpi = dpi, bbox_inches = 'tight')
                plt.close()
            self.value_matrix = novelty_matrix
            # USEFUL FOR DEBUGGING: print(np.max(value_matrix))
        print('Novelty visualized.')




