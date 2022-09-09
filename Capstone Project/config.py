import numpy as np

# num_stacks = 3
# max_blocks = 3
# number_of_blocks = 6
# blocks = list(range(1, number_of_blocks+1))
# state_test = np.array([[0,0,0],[6,5,4],[3,2,1]], dtype=int)
# state_target = np.array([[0,0,0],[1,2,3],[4,5,6]], dtype=int)

num_stacks = 4
max_blocks = 3
number_of_blocks = 8
blocks = list(range(1, number_of_blocks+1))
state_test = np.array([[0,0,0,0],[8,7,6,5],[4,3,2,1]], dtype=int)
state_target = np.array([[0,0,0,0],[1,2,3,4],[5,6,7,8]], dtype=int)

device = "cpu"

MEMORY_SIZE = 5000
MAX_EPISODES = 50000
MAX_STEPS = 250
REWARD = 1

BATCH_SIZE = 256
GAMMA = 0.99
EPS_START = 0.95
EPS_END = 0.05
EPS_DECAY = 10000 ### 1000, 2500, 5000, 10000, 15000, 20000
EPS_DECAYS = [1000, 2500, 5000, 10000, 15000, 20000]
TARGET_UPDATE = 20
TRAIN_PERIOD = 5

FILTER_SIZE = 32 ### 256, 512, 1024, 128 ###
FILTER_SIZES = [32, 64, 128, 256, 512]
LEARNING_RATE = 0.0005 ###  0.001, 0.0001, 0.00005, 0.00001
LEARNING_RATES = [0.005, 0.001, 0.0005, 0.0001, 0.00005]
MOMENTUM = 0.9

TEST_PERIOD = 10
MAX_TEST_STEP = 250











