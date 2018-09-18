'''
Agent. All agent interface occurs through 'step' function.
Given obs, which is in the list format [curr action space, state, reward, termination]

Agent that can learn / decide using multiple strategies
Inputs: observation (partially observed state) and reward
Output: action
Parameters:
    learning rate
    value update:
        TD
            Lambda: 0 <= lambda < 1
            todo: implement td lambda > 0
        MC
            todo: implement eligibility trace
            todo: soft-Q learning value estimation (right not only hardmax is implemented)
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
                if 'lambda' in parameters:
                    self.TD_lambda = parameters['lambda']
                else:
                    raise Exception('TD value updating mode instantiated under parameters.\n'
                                    'Please indicate lambda parameters (0 to 1)')

        else:
            raise Exception('Missing parameter: learning mode')
        if 'exploration policy' in parameters:
            self.exploration_policy = parameters['exploration policy']
        else:
            raise Exception('Missing parameter: exploration policy')
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
        self.Qfunction = {}

    def step(self, obs):
        self.curr_actionspace = obs[0]
        self.curr_state = obs[1] # current set to integer state (0,1,2,...)
        self.reward = obs[2]
        self.termination = obs[-1]
        # ONLY running if prev_state field is populated!
        self.learn_value()
        action = self.pick_action()
        if self.Model is not None:
            self.learn_model()
            self.plan() # planning in the context of DynaQ. See Sutton & Barto.
        # preparation for next step
        self.prev_state = self.curr_state
        self.prev_action = action
        if self.termination:
           # restart episode
           self.prev_state = None
           self.prev_action = None

        return action

    def learn_value(self):
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
        if self.termination:
            return None
        actionspace = self.curr_actionspace
        values = []
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

    def init_values(self, actionspace):
        '''
        This function ensures that all the Q(s,a) have values attached to them. If not, assign random.
        '''
        for a in actionspace:
            if (self.curr_state, a) not in self.Qfunction.keys():
                self.Qfunction[self.curr_state,a] = 0

    def egreedy_choice(self, values, actionspace):
        e = 0.1 # probablity of exploration
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












