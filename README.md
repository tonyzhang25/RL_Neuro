# Tabular RL Repo for Animal Behavior Modelling in binary maze. Implemented models include Q-learning, TD(lambda), Dyna-Q, Monte-Carlo, novelty-based exploration.

Requirements: Python 3.6, numpy

To run experiments, first clone the repo (master branch), then modify dictionary parameters under run.py and execute.
Maze maps and visualizations are saved under /data.

## Implemented Configurations
### Agent Parameters:
1. learning rate
2. value update
   - TD: pick one parameter below to specify:
      - Lambda: 0 <= lambda < 1
      - steps: >= 0
   - MC
   - To understand differences between implementations of TD(1), TD(0), and MC:
    http://www-anw.cs.umass.edu/~barto/courses/cs687/Chapter%207-printable.pdf
3. Exploration policy
   - Random (not dependent on values)
      - E-greedy
      - Softmax
4. Learn model
   - Specify model learning for Dyna-Q algorithm.
   - Need to specify number of planning steps in addition.
5. probabilistic agent state
   - set to True or False (False is Default)
   - Introduce uncertainty in agent state
   - Randomness currently parameterized through p_random under Agent.py

### Environments Parameters:
1. Name: string description. Used to save map under data/maze.
2. levels: must be > 1. 2 is equivalent to simple one juncture t maze.
3. reward_location: given by dictionary object: {(level_1, pos_1): 1, (level_2, pos_2): 0.5,..}. Index by zero.

## To-dos
To-do items are listed under section 'Projects'.
