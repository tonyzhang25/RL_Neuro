'''
Binary maze environment object for RL agent

Assumptions:
- episode terminates at the end branching point, regardless of rewarded or not.
- reward placed at one of the end nodes at the bottom of the tree.

Parameters:
    name: string description. Used to save map under data/maze.
    levels: must be > 1. 2 is equivalent to simple one juncture t maze.
    reward_location: given by dictionary object:
    {(level_1, pos_1): 1, (level_2, pos_2): 0.5,..}. Index by zero.

Variables:
    Actions: 2 (0 or 1)
    Transition matrix: State(t+1) = state_transition_matrix[State(t), action(t)]
    All variables are fixed. Transition matrix is instantiated based on parameter 'levels'

Each state ALWAYS has 2 or 3 actions (depending on if reversal is allowed.
For custom mazes with arbitrary action space in grid space, use Maze.py

Note: Nothing is stored / modified in this object during learning.

'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os, sys, glob


class BinaryMaze:

    def __init__(self, name, nb_levels, reward_location, allow_reversals = True, save = False):
        assert nb_levels > 1
        self.name = name
        self.nb_levels = nb_levels
        self.allow_reversals = allow_reversals
        if self.allow_reversals:
            self.action_space = 3
        else:
            self.action_space = 2
        self.nb_states = 2 ** (nb_levels) - 1
        self.assign_states_to_levels()
        self.init_state_transition_map()
        self.reward_locations = reward_location
        self.init_reward()
        self.start_state = 0
        if save:
            self.save_map(name)
        self.set_termination_states(reward_state = True, start_state= True)

    def init_save_path(self, path):
        self.output_path = path
        if not os.path.exists(path):
            os.mkdir(path)

    def init_state_transition_map(self):
        self.state_trans_matrix = np.zeros((self.nb_states, self.action_space))
        self.compute_trans_map()

    def assign_states_to_levels(self):
        self.states_by_level = []
        curr_state = 0
        for level_i in range(self.nb_levels):
            nb_states_in_level_i = 2 ** level_i
            states_in_level_i = np.arange(curr_state,curr_state+nb_states_in_level_i)
            self.states_by_level.append(states_in_level_i)
            curr_state = curr_state+nb_states_in_level_i

    def compute_trans_map(self):
        # for level_i in range(self.nb_levels):
        #     if level_i < self.nb_levels - 1: # before reaching final layer
        #         for pos, state_j in enumerate(self.states_by_level[level_i]):
        #             next_level = level_i + 1
        #             next_state_a0_position = pos*2
        #             next_state_a0 = self.states_by_level[next_level][next_state_a0_position]
        #             next_state_a1 = next_state_a0 + 1
        #             self.state_trans_matrix[state_j] = [next_state_a0, next_state_a1]
        # ### allow reversals
        # if self.allow_reversals:
        # todo: fix transition from leaf node for conditions allowing reversal
        for level_i in range(self.nb_levels):
            for pos, state_j in enumerate(self.states_by_level[level_i]):
                next_states = []
                if level_i < self.nb_levels - 1:  # before reaching final layer
                    next_level = level_i + 1
                    next_state_a0_position = pos * 2
                    next_state_a0 = self.states_by_level[next_level][next_state_a0_position]
                    next_state_a1 = next_state_a0 + 1
                    next_states.append(next_state_a0) # next level, action L
                    next_states.append(next_state_a1) # next level, action R
                else: # last level. no more next states, only previous
                    next_states.extend([state_j, state_j]) # todo: verify statistics with this
                if self.allow_reversals:
                    if level_i > 0:
                        if pos % 2 == 0:
                            prev_state_pos = pos // 2
                        else:
                            prev_state_pos = (pos - 1) // 2
                        prev_state = self.states_by_level[level_i - 1][prev_state_pos]
                    else:
                        prev_state = 0
                    next_states.append(prev_state)
                self.state_trans_matrix[state_j] = next_states

    def init_reward(self):
        self.state_reward_matrix = np.zeros(self.nb_states)
        # if type(self.reward_locations)
        for reward_loc in self.reward_locations:
            level, pos = reward_loc[0], reward_loc[1]
            state = self.states_by_level[level][pos]
            self.state_reward_matrix[state] = self.reward_locations[reward_loc]

    def set_termination_states(self, reward_state = True, leaf_nodes = False, start_state = False):
        # set last level states to termination
        self.termination_states = []
        if reward_state:
            # only set rewarding state as terminal
            self.termination_states.extend(np.where(self.state_reward_matrix != 0)[0])
        if leaf_nodes:
            # set all states at lowest level to be terminal
            self.termination_states.extend(self.states_by_level[-1])
        if start_state:
            start_state = 0
            self.termination_states = np.append(self.termination_states, start_state)

    def save_map(self, name):
        self.init_save_path('data/maps/')
        np.savez(self.output_path + name,
                 state_trans_matrix = self.state_trans_matrix,
                state_reward_matrix = self.state_reward_matrix)
        # print('** Maze successfully saved as .npz file: '+name)

    def compute_shortest_dist(self, point = None, visualize = True):
        '''
        *** Deprecated function from Maze.py
        compute shortest distance between each state and the point
        Do this by assigning distances for adjacent states and progressively
        move down.
        This function is akin to Dijkstra's Algorithm.
        Currently computing shortest distance to one reward location
        '''
        print('* Computing shortest distance to reward for each state..')
        unvisited_states = set(np.arange(self.nb_states))
        distances = np.full((self.nb_states), np.inf) # this is used to keep track of all distances.
        curr_state = list(self.reward_func.keys())[0]
        distances[curr_state] = 0 # reward state set to distance of 0
        neighbor_states = self.find_neighbors(curr_state)
        while len(unvisited_states) > 0:
            if len(neighbor_states) > 0:
                for neighborState in neighbor_states:
                    if neighborState in unvisited_states:
                        distances[neighborState] = distances[curr_state] + 1
            # when finished with updating neighbor state dists, remove curr state
            unvisited_states.discard(curr_state)
            print('States left: '+str(len(unvisited_states)))
            ## now pick next state: one with the smallest dist
            minDist = float("inf")
            for unvisitedState in unvisited_states:
                dist = distances[unvisitedState]
                if dist < minDist:
                    minDist = dist
                    minDistState = unvisitedState
            curr_state = minDistState
            neighbor_states = self.find_neighbors(curr_state)

        self.state_reward_distance = distances

        if visualize:
            plt.ioff()
            fig = plt.figure()
            ax = fig.add_subplot(111)
            cax = ax.matshow(np.reshape(distances, (17,17)), cmap = 'Reds')
            map_png_dir = '/media/tzhang/Tony_WD_4TB/Maze_Videos/Maze_Design.png'
            maze_map_png = mpimg.imread(map_png_dir)
            height = np.shape(maze_map_png)[0]
            maze_map_png = maze_map_png[:, 0:height]
            plt.imshow(maze_map_png, extent=[-0.5, 16.5, -0.5, 16.5], origin='lower')
            plt.gca().invert_yaxis()  # invert axis to align with video
            fig.colorbar(cax)
            plt.savefig('MazeDistanceMap.png', dpi = 200)
            plt.close()

    def find_neighbors(self, state):
        '''
        *** Deprecated function from Maze.py
        for deterministic state transition maps only.
        Returns all neighboring states, no corresponding actions
        '''
        stateTransProb = self.state_transition_prob[state]
        neighbors = []
        for actionProb in stateTransProb:
            nextState = np.where(actionProb)[0][0] # finding where prob is 1
            if nextState != state:
                neighbors.append(nextState)
        return neighbors








