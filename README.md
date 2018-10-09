# RL_Neuro

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
    To understand differences between implementations of TD(1), TD(0), and MC:
    http://www-anw.cs.umass.edu/~barto/courses/cs687/Chapter%207-printable.pdf
3. Exploration policy
  - random (not dependent on values)
  - e-greedy
  - softmax
4. learn model
  - Specify model learning for dyna-Q algorithms

### Environments Parameters:
1. Name: string description. Used to save map under data/maze.
2. levels: must be > 1. 2 is equivalent to simple one juncture t maze.
3. reward_location: given by dictionary object: {(level_1, pos_1): 1, (level_2, pos_2): 0.5,..}. Index by zero.

## To-dos
To-do items are listed under section 'Projects'.
