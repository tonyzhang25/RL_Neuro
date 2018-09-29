'''
This class wraps around all the other classes for ease of experiment running.
It can be used to organize multiple experiments in order to plot comparison plots.
Class inputs:
    environments: list of dictionary maze properties, or just one maze property
        {mazeName, nb_levels, reward_location}
    agents: list of agents, or just one agent.
    if both agents and environments are lists, the corresponding indexed agent will be
    ran on the corresponding environment
'''

import numpy as np
import os, sys, glob
from Binary_Maze import *
from Interact import *
from Agent import *
from Analysis import *


class Experiment:

    def __init__(self, name, environments, agents, nb_episodes, nb_trials):
        self.name = name
        self.environments = environments
        self.agents = agents
        self.nb_episodes = nb_episodes
        self.nb_trials = nb_trials
        self.env_settings = {
            'init_state': 0,
            'episode_termination': 'environment termination states'
        }
        self.exp_data = []
        self.init_output_path(root_path = 'data/analysis/')

    def init_output_path(self, root_path):
        # Make root path (data/analysis) if not present
        self.output_path = root_path
        if not os.path.exists(root_path):
            os.mkdir(root_path)
        # Make experiment folder if missing
        self.exp_output_path = root_path + str(self.name)
        if not os.path.exists(self.exp_output_path):
            os.mkdir(self.exp_output_path)
        else:
            append_idx = 1
            while os.path.exists(self.exp_output_path + '_' + str(append_idx)):
                append_idx += 1
            # make folder with append_idx
            self.exp_output_path += '_' + str(append_idx)
            os.mkdir(self.exp_output_path)

    def run_experiment(self):
        print('Staring experiment..')
        nb_environments = len(self.environments)
        nb_agents = len(self.agents)
        if nb_environments > 1 and nb_agents > 1:
            assert nb_environments == nb_agents
            self.multi_agent_multi_environment()
        elif nb_environments > 1:
            self.multi_environment()
        else: # nb_agents > 1
            self.multi_agent()

    def multi_agent(self):
        '''
        for experiments involving one environment, and multiple agents
        '''
        print('Mode: multiple agents, single environment')
        env = self.environments[0]
        for exp_id, agent_i in enumerate(self.agents):
            self.init_env_and_session(env)
            self.baseloop(agent_i, exp_id)

    def multi_environment(self):
        '''
        for experiments involving one agent, and multiple environments
        '''
        print('Mode: multiple environments, single agent')
        agent = self.agents[0]
        for exp_id, env_i in enumerate(self.environments):
            self.init_env_and_session(env_i)
            self.baseloop(agent, exp_id)

    def multi_agent_multi_environment(self):
        '''
        for experiments involving multiple environment and multiple agents
        '''
        print('Mode: multiple agents, multiple environments')
        for exp_id, (agent_i, env_i) in enumerate(zip(self.agents, self.environments)):
            self.init_env_and_session(env_i)
            self.baseloop(agent_i, exp_id)


    def init_env_and_session(self, env_i_properties):
        # self.Agent_current = Agent(agent_properties)
        mazeName = env_i_properties['maze name']
        nb_levels = env_i_properties['number of levels']
        reward_location = env_i_properties['reward locations']
        self.Maze_current = Maze(mazeName, nb_levels = nb_levels, reward_location = reward_location)
        self.Session_current = Interact(Map = self.Maze_current, properties = self.env_settings)


    def baseloop(self, agent_spec, exp_id, visualize_sessions_separately = True):
        '''
        This function is the basis for running all RL interactions. Inherit from this class when needed.
        '''
        for trial in range(self.nb_trials):
            # Init fresh incarnation of agent
            Agent_current = Agent(agent_spec)
            # Get session and environment objects
            Session_current, Maze_current = self.Session_current, self.Maze_current
            # Start trial
            print()
            for episode in range(self.nb_episodes):
                # Start episode
                obs = Session_current.init_episode()
                action = Agent_current.step(obs)
                termination = False
                while termination == False:  # assert termination condition = False
                    obs = Session_current.step(action)
                    action = Agent_current.step(obs)
                    termination = obs[-1]
                print('| TRIAL: ' + str(trial + 1) +
                      ' | Episode: ' + str(episode + 1) +
                      ' | Reward = ' + str(obs[-2]) + ' |')
                # End of episode processing
                Qvalues = Agent_current.Qfunction  # obtain q values for analysis
                Session_current.add_value_to_record(Qvalues)
            # End of trial processing
            Session_current.process_trial()

        ## Analysis
        if visualize_sessions_separately:
            Analyze = Analysis(self.exp_output_path, exp_id,
                               Maze_current, Session_current)
            Analyze.visualize(dpi = 300)
