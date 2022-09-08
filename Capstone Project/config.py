import numpy as np

num_stacks = 3
max_blocks = 3
number_of_blocks = 6
blocks = list(range(1, number_of_blocks+1))
state_test = np.array([[0,0,0],[6,5,4],[3,2,1]], dtype=int)
state_target = np.array([[0,0,0],[1,2,3],[4,5,6]], dtype=int)


discount_factor = 0.95
replay_memory = 5000
reward = 1 # get reward only when terminal state is reached




















