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
        self.save_configs()
        self.run_experiment()

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
            self.exp_output_path += '_' + str(append_idx) + '/'
            os.mkdir(self.exp_output_path)

    def save_configs(self):
        filename = 'config.txt'
        with open(self.exp_output_path + '/' + filename, 'w') as data:
            data.write('agents = ' + str(self.agents)+'\n\n')
            data.write('environments = ' + str(self.environments)+'\n\n')
            data.write('nb_episodes = ' + str(self.nb_episodes)+'\n\n')
            data.write('nb_trials = ' + str(self.nb_trials))

    def run_experiment(self):
        print('Starting experiment..')
        nb_environments = len(self.environments)
        nb_agents = len(self.agents)
        if nb_environments > 1 and nb_agents > 1:
            assert nb_environments == nb_agents
            self.multi_agent_multi_environment()
        elif nb_environments > 1:
            self.multi_environment()
        else: # nb_agents > 1
            self.multi_agent()
        ## Cross session analysis
        self.plot_comparison_visuals()

    def plot_comparison_visuals(self):
        # insert correct label
        if self.exp_mode == 'multi-agent':
            exp_label = 'Agent: '
        elif self.exp_mode == 'multi-env':
            exp_label = 'Environment: '
        else: # multi-agent, multi-env
            exp_label = 'Agent/Env: '

        self.plot_comparison_cumulative_rewards(exp_label)
        self.plot_comparison_timesteps_til_terminate(exp_label)
        self.plot_comparison_steps_til_reward(exp_label)
        print('Experiment-level comparison plots visualized.')

    def plot_comparison_cumulative_rewards(self, exp_label):
        cumulative_rewards = self.all_cumulative_rewards
        plt.figure(figsize=(6, 4))
        x = np.arange(1, self.nb_episodes + 1)
        averages = np.average(cumulative_rewards, axis = 1)
        errors = np.std(cumulative_rewards, axis = 1)

        for nb, (avg_i, err_i) in enumerate(zip(averages, errors)):
            upper_conf, lower_conf = avg_i + err_i, avg_i - err_i
            plt.fill_between(x, lower_conf, upper_conf, alpha = 0.2)
            plt.plot(x, avg_i, linewidth=1.5, label = exp_label + str(nb + 1))
        ax = plt.axes()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.xlabel('Episode')
        plt.xlim(1, self.nb_episodes)
        plt.ylim(0, )
        plt.ylabel('Cumulative Reward')
        plt.legend(frameon = False, loc = 'upper left')
        plt.savefig(f"{self.exp_output_path}/Experiment_cumulative_reward_comparisons.png",
                    dpi=350, bbox_inches='tight')
        plt.close()
        np.save(self.exp_output_path+'/'+'cumulative_reward_data.npy', cumulative_rewards)

    def plot_comparison_steps_til_reward(self, exp_label):
        all_steps = self.timesteps_until_reward
        plt.figure(figsize = (4,4))
        plt.boxplot(np.transpose(all_steps), showmeans=True, whis = 'range')
        plt.xlabel('Agent')
        plt.ylabel('Steps until first reward encounter')
        plt.savefig(f"{self.exp_output_path}/Experiment_steps_till_reward.png",
                    dpi=350, bbox_inches='tight')

    def plot_unique_states_visited(self):
        '''Plot unique states visited until reward'''
        pass

    def plot_comparison_timesteps_til_terminate(self, exp_label):
        timeteps_per_episode = self.all_timeteps_perepisode
        plt.figure(figsize=(6, 4))
        x = np.arange(1, self.nb_episodes + 1)
        averages = np.average(timeteps_per_episode, axis = 1)
        errors = np.std(timeteps_per_episode, axis = 1)

        for nb, (avg_i, err_i) in enumerate(zip(averages, errors)):
            upper_conf, lower_conf = avg_i + err_i, avg_i - err_i
            plt.fill_between(x, lower_conf, upper_conf, alpha = 0.2)
            plt.plot(x, avg_i, linewidth=1.5, label = exp_label + str(nb + 1))
        ax = plt.axes()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.xlabel('Episode')
        plt.xlim(1, self.nb_episodes)
        plt.ylabel('Timesteps Until Termination')
        plt.legend(frameon = False, loc = 'upper left')
        plt.savefig(f"{self.exp_output_path}/Experiment_timesteps_over_episodes_plot.png",
                    dpi=350, bbox_inches='tight')
        plt.close()
        np.save(f"{self.exp_output_path}/timesteps_over_episodes_data.npy", timeteps_per_episode)

    def init_cross_session_data(self, nb_sessions):
        self.all_cumulative_rewards = np.zeros((nb_sessions,
                                                self.nb_trials,
                                                self.nb_episodes))
        self.all_timeteps_perepisode = np.zeros((nb_sessions,
                                                self.nb_trials,
                                                self.nb_episodes))
        self.timesteps_until_reward = np.zeros((nb_sessions,
                                                self.nb_trials))

    def multi_agent(self):
        '''
        for experiments involving one environment, and multiple agents
        '''
        print('Mode: multiple agents, single environment')
        self.exp_mode = 'multi-agent'
        env = self.environments[0]
        self.init_cross_session_data(len(self.agents))
        for exp_id, agent_i in enumerate(self.agents):
            self.init_env_and_session(env)
            self.baseloop(agent_i, exp_id)

    def multi_environment(self):
        '''
        for experiments involving one agent, and multiple environments
        '''
        print('Mode: multiple environments, single agent')
        self.exp_mode = 'multi-env'
        agent = self.agents[0]
        self.init_cross_session_data(len(self.environments))
        for exp_id, env_i in enumerate(self.environments):
            self.init_env_and_session(env_i)
            self.baseloop(agent, exp_id)

    def multi_agent_multi_environment(self):
        '''
        for experiments involving multiple environment and multiple agents
        '''
        self.exp_mode = 'multi-agent, multi-env'
        print('Mode: multiple agents, multiple environments')
        self.init_cross_session_data(len(self.agents))
        for exp_id, (agent_i, env_i) in enumerate(zip(self.agents, self.environments)):
            self.init_env_and_session(env_i)
            self.baseloop(agent_i, exp_id)


    def init_env_and_session(self, env_i_properties):
        # self.Agent_current = Agent(agent_properties)
        mazeName = env_i_properties['maze name']
        nb_levels = env_i_properties['number of levels']
        reward_location = env_i_properties['reward locations']
        allow_reversals = env_i_properties['allow reversals']
        self.Maze_current = Maze(mazeName, nb_levels = nb_levels, reward_location = reward_location,
                                 allow_reversals = allow_reversals)
        self.Session_current = Interact(Map = self.Maze_current, properties = self.env_settings)


    def baseloop(self, agent_spec, exp_id, visualize_sessions = True, verbose = True):
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
                # update logs
                Session_current.update_logs()
                if verbose:
                    print('| EXP: ' + str(exp_id+1) +
                          ' | Trial: ' + str(trial + 1) +
                          ' | Episode: ' + str(episode + 1) +
                          ' | Reward = ' + str(obs[-2]) + ' |')
                # End of episode processing
                Qvalues = Agent_current.Qfunction  # obtain q values for analysis
                Session_current.add_value_to_record(Qvalues)
                if agent_spec['add exploration bonus']:
                    Session_current.add_novelty_to_record(Agent_current.exploration_bonus)
            # End of trial processing
            Session_current.process_trial()

        ## Session analysis
        if visualize_sessions:
            Analyze = Analysis(self.exp_output_path, exp_id,
                               Maze_current, Session_current)
            Analyze.visualize(dpi = 300)
            ## Obtain session data for cross-session analysis
            self.all_cumulative_rewards[exp_id] = np.array(Analyze.cumulative_rewards)
            self.all_timeteps_perepisode[exp_id] = np.array(Analyze.all_timesteps_trial)
            self.timesteps_until_reward[exp_id] = np.array(Analyze.steps)
