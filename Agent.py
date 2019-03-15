'''
Agent. All agent interface occurs through 'step' function.
Given obs, which is in the list format [curr action space, state, reward, termination]

Agent that can learn / decide using multiple strategies
Inputs: state and reward
Output: action
Parameters:
    learning rate
    value update:
        TD
            pick one parameter below to specify:
            1. Lambda: 0 <= lambda < 1
            2. steps: >= 0
        MC
        To understand differences between implementations of TD(1), TD(0), and MC:
        http://www-anw.cs.umass.edu/~barto/courses/cs687/Chapter%207-printable.pdf
    exploration policy
        random (not dependent on values)
        e-greedy
        softmax
    Decay
        set parameter for decay per trial
        todo: value decay / forgetting across trials


'''
import numpy as np
import random
import pdb



class Agent:

    def __init__(self, parameters):
        self.parameters = parameters
        if 'learning rate' in parameters:
            self.learn_rate = parameters['learning rate']
        else:
            self.learn_rate = 0.1
            print('learn_rate = '+str(self.learn_rate))
        if 'value update' in parameters:
            self.learn_mode = parameters['value update']
            if self.learn_mode == 'TD':
                self.init_TD_variables()
            elif self.learn_mode == 'MC':
                self.init_MC_variables()
        else:
            raise Exception('Missing parameter: learning mode')
        if 'exploration policy' in parameters:
            self.exploration_policy = parameters['exploration policy']
        else:
            raise Exception('Missing parameter: exploration policy')
        if 'probabilistic agent state' in parameters:
            self.probabilistic = parameters['probabilistic agent state']
        else:
            self.probabilistic = False
        if 'discount rate' in parameters:
            self.discount_rate = parameters['discount rate']
        else:
            self.discount_rate = 0.8
            print('Discount rate set to default = ' + str(self.discount_rate))
        self.prev_state = None
        self.prev_action = None
        if parameters['learn model']:
            self.Model = {} ## initial format: key = (STATE, ACTION). map = (STATE, REWARD)
            if 'planning steps' in parameters:
                self.planning_steps = parameters['planning steps']
            else:
                raise Exception('Model-based agent instantiated under parameters. \n'
                                'Please indicate the number of planning steps (> 0).')
        else:
            self.Model = None
        if parameters['add exploration bonus']:
            self.exploration_bonus = {}  # format: key = (state, action), content = value between 0 and 1
        self.Qfunction = {}

    def init_memory(self):
        self.use_memory = True
        self.memory = []  # memory across episodes
        self.episode_memory = []  # memory for current episode only

    def init_MC_variables(self):
        self.init_memory()
        self.state_return_dict = {} # this keeps a list of returns for each visited state

    def init_TD_variables(self):
        if 'lambda' in self.parameters:
            self.TD_lambda = self.parameters['lambda']
        else:
            raise Exception('TD value updating mode instantiated under parameters.\n'
                            'Please indicate lambda parameters (0 to 1)')
        if self.TD_lambda > 0:
            self.init_memory()
        else:
            self.use_memory = False

    def step(self, env_obs):
        '''
        :param obs:
         env_obs[0] = action space (what actions agent could take from environment)
         env_obs[1] = state (tabular)
        '''
        agent_obs = self.determine_agent_state(env_obs)
        if self.learn_mode == 'TD':
            action = self.step_TD(agent_obs)
        elif self.learn_mode == 'MC':
            action = self.step_MC(agent_obs)
        # if self.parameters['add exploration bonus']:
        #     # import pdb; pdb.set_trace()
        #     if action is not None: # not terminal
        #         # reduce novelty for current state-action
        #         self.reduce_state_novelty(action)
        #         # increase novelty for all other pairs
        #         # self.increase_state_novelty(action)
        return action

    def reduce_state_novelty(self, action, min_novelty = 0, reduction = 1):
        '''reduce novelty associated for current state action pair'''
        bonus_current = self.exploration_bonus[self.curr_state, action]
        if bonus_current > min_novelty:
            bonus_reduced = bonus_current - reduction
            if bonus_reduced < 0: bonus_reduced = 0
            self.exploration_bonus[self.curr_state, action] = bonus_reduced

    def increase_state_novelty(self, action, min_novelty = 0, max_novelty = 1, rate = 0.2):
        # increase novelty for all state action pairs NOT selected
        curr_state_action = (self.curr_state, action)
        all_state_actions = list(self.exploration_bonus.keys())
        all_state_actions.remove(curr_state_action)
        for state_action in all_state_actions:
            bonus_current = self.exploration_bonus[state_action]
            diff = max_novelty - bonus_current
            bonus_new =+ rate * diff
            self.exploration_bonus[state_action] = bonus_new

    def update_novelty(self, action, reduction):
        # import pdb; pdb.set_trace()
        if action is not None: # not terminal
            # reduce novelty for current state-action
            self.reduce_state_novelty(action, reduction = reduction)
            # increase novelty for all other pairs
            # self.increase_state_novelty(action)

    def step_TD(self, obs):
        self.curr_state = obs[1]  # current set to integer state (0,1,2,...)
        self.curr_actionspace = obs[0]
        self.reward = obs[2]
        self.terminate = obs[-1]
        action = self.pick_action()
        # update novelty below. Must be done before value updates occurs
        if self.parameters['add exploration bonus']:
            reduction = self.parameters['reduction']
            self.update_novelty(action, reduction)
        if self.use_memory:
            # general purpose memory for use in eligibility trace
            self.add_memory(obs, action)
        # ONLY running if prev_state field is populated!
        if self.TD_lambda == 0:
            self.learn_TD0_value()
        else:  # TD_lambda > 0:
            self.learn_TDl_values()
            # note: TD(lambda) is updated online, every episode
            # in contrast with MC, which updates at the end of episodes
        if self.Model is not None:
            self.learn_model()
            self.plan()  # planning in the context of DynaQ. See Sutton & Barto.
        # preparation for next step
        self.prev_state = self.curr_state
        self.prev_action = action
        if self.terminate:
            # restart episode
            self.prev_state = None
            self.prev_action = None
        return action

    def determine_agent_state(self, env_obs, p_random = 0.3):
        '''
        Optional. Use when
        1) agent has probabilistic memory
        or
        2) agent has superstitious actions (actions outside of set of environment action space)
        '''
        if self.probabilistic:
            if random.random() > p_random:
                agent_obs = env_obs
            else:
                all_visited_states = self.find_states_visited()
                if all_visited_states is not None:
                    state = np.random.choice(all_visited_states)
                    env_obs[1] = state # change perceived current state
                agent_obs = env_obs
        else:
            agent_obs = env_obs
        return agent_obs

    def find_states_visited(self):
        '''find states visited from q function'''
        state_action_pairs = np.array(list(self.Qfunction.keys()))
        if len(state_action_pairs) > 0:
            states = list(set(state_action_pairs[:,0]))
        else:
            states = None
        return states


    def step_MC(self, obs):
        self.curr_actionspace = obs[0]
        self.curr_state = obs[1]  # current set to integer state (0,1,2,...)
        self.reward = obs[2]
        self.terminate = obs[-1]
        # ONLY running if prev_state field is populated!
        action = self.pick_action()
        # if self.Model is not None:
        #     self.learn_model()
        #     self.plan() # planning in the context of DynaQ. See Sutton & Barto.
        # preparation for next step
        self.prev_state = self.curr_state
        self.prev_action = action
        if self.use_memory:  # eligibility trace
            self.add_memory(obs, action)
        if self.terminate:
            # UPDATE VALUES (end of every episode)
            self.learn_MC_value()
            # restart episode
            self.prev_state = None
            self.prev_action = None
        return action

    def add_memory(self, obs, action):
        '''
        This function keeps track of visited states within an episode for MC / multi step TD methods.
        '''
        state = obs[1]
        reward = obs[-2]
        terminate = obs[-1]
        self.episode_memory.append([state, reward, action])
        if terminate:
            # Append memory of current episode to all memories
            self.memory.append(self.episode_memory)
            self.episode_memory = [] # clear episode memory

    def learn_TDl_values(self):
        if len(self.memory) > 0 or len(self.episode_memory) > 0:
            if len(self.episode_memory) > 0:
                memory_episode = self.episode_memory
            else:
                memory_episode = self.memory[-1]
            if len(memory_episode) >= 2: # only learn after there are two states visited
                # compute target at current state
                reward = memory_episode[-1][1]
                curr_state = memory_episode[-1][0]
                curr_action = memory_episode[-1][2]
                prev_state = memory_episode[-2][0]
                prev_action = memory_episode[-2][2]
                if curr_action is not None: # not yet at a terminal state
                    current_value = self.Qfunction[curr_state, curr_action]
                else: # before reaching terminal state
                    current_value = 0
                    '''
                    NOTE: this only works if the value of terminal state
                    is the same as the reward, aka no more reward after.
                    '''
                prev_value = self.Qfunction[prev_state, prev_action]
                if self.parameters['add exploration bonus']:
                    # if curr_action is not None: # todo: QA
                    # novelty_bonus = self.exploration_bonus[curr_state, curr_action]
                    novelty_bonus = self.exploration_bonus[prev_state, prev_action] # is this correct?
                    reward += novelty_bonus
                delta_target = reward + self.discount_rate * current_value - prev_value
                eligibility_t = 1
                # pdb.set_trace()
                for t in range(len(memory_episode) - 1, 0, -1): # iterate backwards
                    # iterate backwards, from most recent state
                    prev_state = memory_episode[t - 1][0]
                    prev_action = memory_episode[t - 1][2]
                    # Update Q function
                    self.Qfunction[prev_state, prev_action] += \
                        self.learn_rate * eligibility_t * delta_target
                    eligibility_t *= self.TD_lambda * self.discount_rate



    def learn_MC_value(self):
        memory_episode = self.memory[-1] # pull out last episode memory
        return_t = 0
        for t in range(len(memory_episode) - 1, 0, -1):
            # iterate backwards, from most recent state
            reward = memory_episode[t][1]
            prev_state = memory_episode[t-1][0]
            prev_action = memory_episode[t-1][2]
            return_t = return_t * self.discount_rate + reward
            if (prev_state, prev_action) not in self.state_return_dict:
                self.state_return_dict[prev_state, prev_action] = []
            self.state_return_dict[prev_state, prev_action].append(return_t)
            # update Q function through averaging
            self.Qfunction[prev_state, prev_action] = np.average(self.state_return_dict[prev_state, prev_action])

    def learn_TD0_value(self):
        '''
        This function can be used if learning is required after every step,
        such as in the case of TD(0).
        '''
        if self.prev_state is not None: # after 1st step only!
            actionspace = self.curr_actionspace
            max_value = -1 # arbitrary initial value less than 0
            for a in actionspace:
                if (self.curr_state, a) not in self.Qfunction.keys():
                    self.Qfunction[self.curr_state, a] = 0
                # Check if current Q(s,a) has the largest value
                if self.Qfunction[self.curr_state,a] > max_value:
                    max_action = a
                    max_value = self.Qfunction[self.curr_state,a]

            if (self.prev_state, self.prev_action) not in self.Qfunction.keys():
                self.Qfunction[self.prev_state, self.prev_action] = 0
            ### HERE THE Q VALUE IS UPDATED!
            self.Qfunction[self.prev_state, self.prev_action] += self.learn_rate * (self.reward + self.discount_rate * self.Qfunction[self.curr_state, max_action] - self.Qfunction[self.prev_state, self.prev_action])
            # if self.termination: pdb.set_trace()

    def learn_model(self):
        if self.prev_state is not None: # beyond 1st step only!
            ## epsilon greedy
            '''
            In this model, we're currently assuming perfect memory. It can rememember 
            all VISITED states and how they transition to each other (based on experience)
            Model is in coordinate space
            '''
            ## learn state transition matrix
            if (self.prev_state, self.prev_action) not in self.Model.keys():
                self.Model[self.prev_state, self.prev_action] = (self.reward, self.curr_state)

    def pick_action(self):
        if self.terminate:
            return None
        actionspace = self.curr_actionspace
        values = []
        if self.parameters['add exploration bonus']:
            self.init_novelty(actionspace)
        self.init_values(actionspace)
            ## softmax
        for a in actionspace:
            values.append(self.Qfunction[self.curr_state, a])
        ## Exploration policy based on values
        if self.exploration_policy == 'e-greedy':
            chosen_action = self.egreedy_choice(values, actionspace)
        elif self.exploration_policy == 'softmax':
            chosen_action = self.softmax_choice(values, actionspace)
        elif self.exploration_policy == 'random':
            chosen_action = self.random_choice(actionspace)
        return chosen_action

    def init_values(self, actionspace, lr = 0.2):
        '''
        This function ensures that all the Q(s,a) have initial values attached to them.
        '''
        for a in actionspace:
            if (self.curr_state, a) not in self.Qfunction.keys():
                if self.parameters['add exploration bonus']:
                    init_value = 0
                    novelty = self.exploration_bonus[self.curr_state, a]
                    diff = novelty - init_value
                    init_value = init_value + lr * diff
                    # init_value = self.exploration_bonus[self.curr_state, a] # todo: QA
                else:
                    init_value = 0
                self.Qfunction[self.curr_state, a] = init_value

    def init_novelty(self, actionspace, max_bonus = 1):
        '''
        This function ensures that all the e(s,a) have values attached to them. If not, assign random.
        '''
        for a in actionspace:
            if (self.curr_state, a) not in self.exploration_bonus.keys():
                self.exploration_bonus[self.curr_state,a] = max_bonus

        # now, update value function. Note that this process is different from initializing Q functions

    def egreedy_choice(self, values, actionspace):
        # probablity of exploration
        e = self.parameters['epsilon']
        rand = random.random()
        if rand > e and len(set(values)) > 1:
            # exploit
            argmax_V = np.argmax(values)
            chosen_action = actionspace[argmax_V]
        else:
            # explore
            chosen_action = np.random.choice(actionspace)
        return chosen_action

    def softmax_choice(self, values, actionspace):
        exp = np.exp(values)
        sum = np.sum(exp)
        prob = exp / sum
        chosen_action = np.random.choice(actionspace, p=prob)
        return chosen_action

    def random_choice(self, actionspace):
        chosen_action = np.random.choice(actionspace)
        return chosen_action

    def plan(self):
        '''
        planning function based on DynaQ. Not model-based planning literature without value function
        approximation. Note distinction. Refer to Sutton and Barto Dyna-Q.
        '''

        if self.prev_state is not None: # only start planning after 1st step
            for i in range(self.planning_steps):
                random_stateaction = random.choice(list(self.Model.keys()))
                next_rewardstate = self.Model[random_stateaction]
                prev_state = random_stateaction[0]
                prev_action = random_stateaction[1]
                reward = next_rewardstate[0]
                curr_state = next_rewardstate[1]
                #### learn!
                max_value = -1 # arbitrary value less than 0
                for a in [0,1,2,3]:
                    if (curr_state, a) in self.Qfunction.keys():
                        if self.Qfunction[curr_state, a] > max_value:
                            max_next_action = a
                            max_next_value = self.Qfunction[curr_state,a]
                ### HERE THE Q VALUE IS UPDATED!
                self.Qfunction[prev_state, prev_action] += self.learn_rate * (reward + self.discount_rate * self.Qfunction[curr_state, max_next_action] - self.Qfunction[prev_state, prev_action])












