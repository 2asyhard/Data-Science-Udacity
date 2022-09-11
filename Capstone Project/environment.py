import numpy as np
import copy

class Environment:
    def __init__(self, num_stack=4, max_stack=4):
        self.num_stack = num_stack
        self.max_stack = max_stack
        self.actions = self.num_stack*(self.num_stack-1)
        self.s0 = np.array([[0, 0, 0, 0], [3, 3, 0, 2], [3, 2, 1, 1], [1, 2, 3, 2]], dtype=int)
        self.s = copy.deepcopy(self.s0)
        self.stopped = 0
        self._make_action_list()


    def _make_action_list(self):
        '''
        generate action list
        action [i, j] -> move block from stack i to stack j
        '''
        self.action_list = []
        for i in range(self.num_stack):
            for j in range(self.num_stack):
                if i != j:
                    self.action_list.append([i, j])


    def reset(self):
        '''
        Reset environment and initialize initial state
        '''
        self.s = copy.deepcopy(self.s0)
        return self.s


    def step(self, action):
        """
        Execute action and check if new state is terminal state
        if agent didn't reach terminal state, reward is 0
        """
        s = self.s
        s1 = self._move_plate(s, self.action_list[action])

        self.success = False
        d, r = self.is_rearrange_finish(s1)
        if d:
            self.success = True

        return s1, r, d,


    def is_rearrange_finish(self, state):
        """
        check if state is terminal state
        if terminal state, returns True and 100(reward)
        """
        reward = 0
        for i in range(self.num_stack):
            for j in range(self.max_stack-1):
                if state[j,i] > state[j+1,i]:
                    return False, reward
        reward = 100
        return True, reward


    def _move_plate(self, state, action):
        """
        This is actual function to move block
        """
        target_plate = 0
        out_zone_tier = -1
        in_zone_tier = -1
        new_state = copy.deepcopy(state)
        for i in range(self.max_stack):
            if state[i, action[0]] != 0:
                out_zone_tier = i
                break

        for i in range(1, self.max_stack + 1):
            if state[-i, action[1]] == 0:
                in_zone_tier = self.max_stack - i
                break

        if in_zone_tier == -1 or out_zone_tier == -1:
            #reward -= 1
            return new_state
        else:
            for i in range(self.max_stack):
                if new_state[i, action[0]] != 0:
                    target_plate = new_state[i, action[0]]
                    new_state[i, action[0]] = 0
                    break

            for i in range(1, self.max_stack+1):
                if new_state[-i, action[1]] == 0:
                    in_zone_tier = self.max_stack - i
                    new_state[-i, action[1]] = target_plate
                    break

            return new_state