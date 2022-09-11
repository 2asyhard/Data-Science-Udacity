import tensorflow as tf
import numpy as np
import random
from collections import deque
import environment
import tensorflow.contrib.slim as slim
from copy import deepcopy

num_stack = 4
max_stack = 4
env = environment.Environment(num_stack, max_stack)

input_size = num_stack * max_stack
output_size = num_stack*(num_stack-1)

dis = 0.95
REPLAY_MEMORY = 50000

class DQN:
    def __init__(self, session, input_size, output_size, name = "main"):
        """
        Class that calculates Q-values for input states
        """
        self.session = session
        self.input_size = input_size
        self.output_size = output_size
        self.net_name = name
        self._build_network()
        self.summary_writer = tf.summary.FileWriter("output/summary")

    def _build_network(self, h_size = 128):
        with tf.variable_scope(self.net_name):
            self._X = tf.placeholder(tf.float32, [None, self.input_size], name = "input_x")
            self.learning_rate = tf.placeholder(tf.float32, shape=[], name = 'learning_rate')

            self.imageIn = tf.reshape(self._X, shape=[-1, max_stack, num_stack, 1])
            self.conv1 = slim.conv2d( \
                inputs=self.imageIn, num_outputs=h_size, kernel_size=[1, 1], stride=[1, 1], padding='SAME',
                biases_initializer=None)
            self.conv1 = tf.layers.batch_normalization(inputs=self.conv1)
            self.conv1 = tf.nn.relu(self.conv1)

            self.conv2 = slim.conv2d( \
                inputs=self.conv1, num_outputs=h_size, kernel_size=[2, 2], stride=[1, 1], padding='SAME',
                biases_initializer=None)
            self.conv2 = tf.layers.batch_normalization(inputs=self.conv2)
            self.conv2 = tf.nn.relu(self.conv2)

            self.conv3 = slim.conv2d( \
                inputs=self.conv2, num_outputs=h_size, kernel_size=[2, 2], stride=[1, 1], padding='SAME',
                biases_initializer=None)
            self.conv3 = tf.layers.batch_normalization(inputs=self.conv3)
            self.conv3 = tf.nn.relu(self.conv3)

            self.flat_1 = slim.flatten(self.conv3)
            xavier_init = tf.contrib.layers.xavier_initializer()
            self.flat_1_W = tf.Variable(xavier_init([int(self.flat_1.shape[1]), h_size]))

            self.flat_2 = tf.matmul(self.flat_1, self.flat_1_W)
            self.flat_2_W = tf.Variable(xavier_init([h_size, env.actions]))

            self._Qpred = tf.matmul(self.flat_2, self.flat_2_W)

        self._Y = tf.placeholder(shape = [None, self.output_size], dtype = tf.float32)

        # Loss function
        self._loss = tf.reduce_mean(tf.square(self._Y - self._Qpred))

        # Training
        self._train = tf.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(self._loss)

    def predict(self, state):
        x = np.reshape(state, [1, self.input_size])
        return self.session.run(self._Qpred, feed_dict = {self._X:x})

    def update(self, x_stack, y_stack, learning_rate):
        return self.session.run([self._loss, self._train], feed_dict={self._X:x_stack, self._Y:y_stack, self.learning_rate:learning_rate})



def replay_train(mainDQN, targetDQN, train_batch, learning_rate):
    """
    :param mainDQN: main network that is used for predicting Q-value
    :param targetDQN: target network that used for calculating loss value
    :param train_batch: datasets for training
    :param learning_rate: learning rate for training
    :return: returns trained main network(mainDQN)
    """
    x_stack = np.empty(0).reshape(0, input_size)
    y_stack = np.empty(0).reshape(0, output_size)

    for state, action, reward, next_state, done in train_batch:
        Q = mainDQN.predict(state)

        if done:
            Q[0, action] = reward
        else:
            Q[0, action] = reward + dis * np.max(targetDQN.predict(next_state))

        y_stack = np.vstack([y_stack, Q])
        x_stack = np.vstack([x_stack, np.reshape(state, [1, input_size])])

    return mainDQN.update(x_stack, y_stack, learning_rate)


def get_copy_var_ops(*, dest_scope_name = "target", src_scope_name="main"):
    """
    This function will copy mainDQN to targetDQN
    """
    op_holder = []

    src_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=src_scope_name)
    dest_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=dest_scope_name)

    for src_var, dest_var in zip(src_vars, dest_vars):
        op_holder.append(dest_var.assign(src_var.value()))

    return op_holder


def main():
    """
    main function to train network using reinforcement learning
    """
    max_episodes = 500000
    maximum_step = 500
    successList = []
    step_list = []
    replay_buffer = deque()
    lr = 1e-5

    with tf.Session() as sess:
        # Initiailize networks
        mainDQN = DQN(sess, input_size, output_size, name="main")
        targetDQN = DQN(sess, input_size, output_size, name="target")
        tf.global_variables_initializer().run()

        #initial copy q_net -> target_net
        copy_ops = get_copy_var_ops(dest_scope_name="target", src_scope_name="main")
        sess.run(copy_ops)

        for episode in range(1, max_episodes+1):
            # start episode
            e = 1. / ((episode / 50000) + 1)
            step_count = 0
            state = deepcopy(env.reset())
            episode_data = []

            for step in range(1, maximum_step+1):
                if np.random.rand(1) < e:
                    action = np.random.randint(0, num_stack * (num_stack - 1))
                else:
                    q_list = mainDQN.predict(state)
                    action = np.argmax(q_list)
                next_state, reward, done = env.step(action)

                env.s = next_state
                step_count += 1
                episode_data.append((state, action, reward, next_state, done))

                if done:
                    for data in episode_data:
                        replay_buffer.append(data)
                        if len(replay_buffer) > REPLAY_MEMORY:
                            replay_buffer.popleft()

                    step_list.append(step)
                    successList.append(1)
                    break

                state = next_state
                step += 1

            if not done:
                """
                If it failed to reach terminal state
                add 0 to successList and add maximum step to step_list
                """
                successList.append(0)
                step_list.append(maximum_step)

            # train every 20 episodes
            if episode % 20 == 0:
                for _ in range(40):
                    minibatch = random.sample(replay_buffer, 64)
                    mainDQN = replay_train(mainDQN, targetDQN, minibatch, lr)

            # update target network every 60 episodes
            if episode % 60 == 0:
                sess.run(copy_ops)

            if episode % 500 == 0:
                print(f"Episode: {episode-499} ~ {episode}")
                print(f"Success rate: {sum(successList)/500}")
                print(f"Average steps: {np.mean(step_list)}")
                successList = []
                step_list = []


if __name__ == "__main__":
    main()





