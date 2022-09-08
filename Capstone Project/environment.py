from copy import deepcopy
import config as cfg
import numpy as np


class Environment:
    def __init__(self):
        self.columns = cfg.num_stacks
        self.rows = cfg.max_blocks
        self.blocks = cfg.blocks
        self.action_size = self.columns*(self.columns-1)
        self.action_list = self.make_action_list()
        self.state = deepcopy(cfg.state_target)


    def make_action_list(self):
        # make action list
        action_list = []
        for i in range(self.columns):
            for j in range(self.columns):
                if i != j:
                    self.action_list.append([1, i, j])
        return action_list


    def step(self, action_num):
        # action -> [validity, out stack, in stack]
        # return state, action, next state, reward, done
        pre_state = deepcopy(self.state)
        action = self.action_list[action_num]
        self.execute_action(action)
        done, reward = self.check_terminal_state()
        return pre_state, action_num, self.state, reward


    def execute_action(self, in_action):
        out_stack = in_action[1]
        in_stack = in_action[2]

        for i in range(self.rows):
            if self.state[i, out_stack] != 0:
                target_block = self.state[i, out_stack]
                self.state[i, out_stack] = 0
                break

        for i in range(self.rows):
            if self.state[-i-1, in_stack] == 0:
                self.state[-i-1, in_stack] = target_block
                break


    def get_valid_action_list(self):
        # get possible action and impossible action
        action_list = deepcopy(self.action_list)
        stack_size = []
        for i in range(self.columns):
            tmp_stack_size = 0
            for j in range(self.rows):
                if self.state[j, i] != 0:
                    tmp_stack_size += 1
            stack_size.append(tmp_stack_size)

        for i in range(len(stack_size)):
            for j in range(len(action_list)):
                if stack_size[i] == 0 and action_list[j][1] == i:
                    action_list[j][0] = 0
                if stack_size[i] == self.rows and action_list[j][2] == i:
                    action_list[j][0] = 0

        return action_list


    def check_terminal_state(self):
        '''
        check if self.state is terminal state
        if its terminal state than return True and reward(1)
        '''
        reward = 0
        for i in range(self.columns):
            for j in range(self.rows - 1):
                if self.state[j, i] > self.state[j + 1, i]:
                    return False, reward
        reward = cfg.reward
        return True, reward


    def get_child_states(self):
        parent_state = deepcopy(self.state)
        valid_actions_list = self.get_valid_action_list()
        child_state_list = []

        for valid_action in valid_actions_list:
            if valid_action[0] == 1:
                self.execute_action(valid_action)
                child_state_list.append(deepcopy(self.state))
                self.state = deepcopy(parent_state)
            else:
                child_state_list.append(False)

        return child_state_list


    def reset(self):
        while True:
            self.state = self.make_random_state()
            done, reward = self.check_terminal_state()
            if done is False:
                break


    def make_random_state(self):
        tmp_stack = self.make_90degree_random_state()

        for i in range(self.columns):
            while not len(tmp_stack[i]) == self.rows:
                tmp_stack[i].append(0)

        random_state = np.rot90(tmp_stack)
        return random_state


    def make_90degree_random_state(self):
        tmp_stack = []
        for _ in range(self.columns):
            tmp_stack.append([])

        for _ in range(len(self.blocks)):
            left_blocks = len(self.blocks)
            block_idx = np.random.randint(0, left_blocks)
            block_num = self.blocks[block_idx]
            while True:
                chosen_stack_num = np.random.randint(0, self.columns)
                if len(tmp_stack[chosen_stack_num]) < self.rows:
                    tmp_stack[chosen_stack_num].append(block_num)
                    break
            del self.blocks[block_idx]

        return tmp_stack
