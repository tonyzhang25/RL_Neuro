'''
Interact Class - modified from original under Maze.py.
This class works exclusively with Binary_Maze.py.
this class is used as intermediary between agent and environment.
Basically, Interact feeds observations to the agent based on the environment,
and receive actions from the agent, which it then processes to feed the next obs.

This class has a default set of properties that can be changed by feeding in the properties

Properties:
    'init_state':
        random: random start state, even distribution over state space.
        state: non random, always start from same spot.

    'episode_termination':
        reward completion
        max steps
        environment termination states

    max_steps (if set above):
        nb_steps: integer
'''
import numpy as np
from copy import deepcopy
from Binary_Maze import *
import pdb


class Interact:

    def __init__(self, init_properties, maze_properties):
        '''
        NOTE: agent should not have access to any variables in this section.
        These variables are for tracking progress for evaluation / debug purposes.
        '''
        self.maze_properties = maze_properties
        self.properties = init_properties
        ## instantiate history variables for evaluation / debugging
        self.state_act_history = [] # record of agent's state and picked actions
        self.state_obs_history = [] # record agent's observed state history (action spaces in current case)
        self.state_act_history_episodes = []
        self.state_obs_history_episodes = []
        self.state_act_history_trials = []
        self.state_obs_history_trials = []
        self.agent_qvalues_history_episodes = []
        self.agent_qvalues_history_trials = []
        self.agent_novelty_history_episodes = []
        self.agent_novelty_history_trials = []
        self.termination_condition = init_properties['episode_termination']
        self.episode_nb = 0
        if self.termination_condition == 'max_episode':
            self.max_steps = init_properties['max_steps'] # integer referring to max number of steps

    def init_episode(self, episode):
        self.episode_nb += 1
        self.reset_environment(episode)
        if self.properties['init_state'] == 'random':
            self.init_state = np.random.randint(self.Maze.nb_states)
        else: # specific state
            self.init_state = self.properties['init_state']
        self.current_state = self.init_state
        reward = self.check_reward()
        term = self.check_termination()
        if term:
            ### end episode ##
            print('* Termination condition satisfied.')
        output = self.return_observation(reward, term)
        return output

    def reset_environment(self, episode):
        '''
        this is called at the beginning of EACH episode.
        1. clear history 2. reset environment
        steps for the first episode, regardless of the actual events
        New: episode being passed in here in order to accommodate
        changing of reward location within a trial.
        '''
        if self.episode_nb > 0: # only apply AFTER first episode
            # self.Maze = deepcopy(self.Maze_original)
            self.create_maze(episode)
            self.state_act_history, self.state_obs_history = [], []

    def create_maze(self, episode):
        mazeName = self.maze_properties['maze name']
        nb_levels = self.maze_properties['number of levels']
        reward_location = self.maze_properties['reward locations']
        allow_reversals = self.maze_properties['allow reversals']
        if self.maze_properties['change reward location']:
            change_episodes = list(reward_location.keys())
            change_episodes.reverse()
            for ep_change in change_episodes:
                if episode >= ep_change:
                    reward_location = reward_location[ep_change]
                    break
        self.Maze = Maze(mazeName, nb_levels=nb_levels, reward_location=reward_location,
                         allow_reversals=allow_reversals)
        self.action_space = np.arange(self.Maze.action_space)  # Based on environment


    def update_logs(self):
        self.state_act_history_episodes.append(self.state_act_history)
        self.state_obs_history_episodes.append(self.state_obs_history)


    def step(self, action, verbose = False):
        # get new state from environment class
        new_state = int(self.Maze.state_trans_matrix[self.current_state, action])
        if verbose:
            print(f"| Action: {action} | New State: {new_state}")
        # save to history
        self.state_act_history.append([self.current_state, action, new_state])
        self.current_state = new_state # set new state to current state
        reward = self.check_reward()
        termination = self.check_termination()
        output = self.return_observation(reward, termination)
        return output

    def return_observation(self, reward, termination):
        '''
        THIS FUNCTION DECIDES WHAT OBSERVATION TO GIVE TO AGENT
        Since we are assuming full MDP, the agent is fed perfect information (e.g. state = 1)
        Note: state_obs_history is a history of what the agent observes.
        If the agent observe the underlying MDP, then it's the same as the environment
        MDP history, state_act_history.
        '''
        combined_output_to_agent = [self.action_space, self.current_state, reward, termination]
        self.state_obs_history.append(combined_output_to_agent)
        return combined_output_to_agent

    def process_trial(self):
        # Call this after the end of a trial before reinitializing agent to naive state.
        # populates trial data into one
        if len(self.state_obs_history_episodes) > 0:
            self.state_obs_history_trials.append(self.state_obs_history_episodes)
            self.state_act_history_trials.append(self.state_act_history_episodes)
            # reset episode history memory
            self.state_obs_history_episodes = []
            self.state_act_history_episodes = []
        if len(self.agent_qvalues_history_episodes) > 0:
            self.agent_qvalues_history_trials.append(self.agent_qvalues_history_episodes)
            self.agent_qvalues_history_episodes = []
        if len(self.agent_novelty_history_episodes) > 0:
            self.agent_novelty_history_trials.append(self.agent_novelty_history_episodes)
            self.agent_novelty_history_episodes = []

    def add_value_to_record(self, Qvalues):
        self.agent_qvalues_history_episodes.append(Qvalues.copy())
        # use .copy because of pointer issue with python dictionaries

    def add_novelty_to_record(self, novelty):
        self.agent_novelty_history_episodes.append(novelty.copy())
        # use .copy because of pointer issue with python dictionaries

    def check_reward(self):
        reward = self.Maze.state_reward_matrix[self.current_state]
        return reward

    def check_termination(self):
        if self.termination_condition == 'reward completion':
            sum_rewards = sum(self.Maze.reward_func.values())
            if sum_rewards == 0:
                return True
            else:
                return False
        elif self.termination_condition == 'max steps':
            if self.step_nb == self.max_steps:
                return True
            else:
                return False
        elif self.termination_condition == 'environment termination states':
            if self.current_state in self.Maze.termination_states:
                return True
            else:
                return False



