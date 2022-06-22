import numpy as np
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.losses import MSE
import tensorflow as tf


class ActorNetwork(Model):
    def __init__(self, n_actions):
        super().__init__()

        self.hidden_layer = [512, 256, 128]
        self.sequential = Sequential()

        for hidden in self.hidden_layer:
            self.sequential.add(Dense(hidden, activation='relu'))

        self.u = Dense(n_actions, activation='tanh')

    def call(self, inputs):
        output = self.sequential(inputs)
        u = self.u(output)

        return u


class CriticNetwork(Model):
    def __init__(self):
        super().__init__()

        self.hidden_layer = [512, 256, 128]
        self.sequential = Sequential()

        for hidden in self.hidden_layer:
            self.sequential.add(Dense(hidden, activation='relu'))

        self.Q = Dense(1, activation=None)

    def call(self, inputs):
        output = self.sequential(inputs)
        q = self.Q(output)

        return q


class ReplayBuffer():
    def __init__(self, max_mem, input_shape, n_action):
        self.max_mem = max_mem
        self.input_shape = input_shape
        self.n_action = n_action
        self.state_memory = np.zeros((self.max_mem, self.input_shape))
        self.next_state_memory = np.zeros((self.max_mem, self.input_shape))
        self.action_memory = np.zeros((self.max_mem, self.n_action))
        self.reward_memory = np.zeros(self.max_mem)
        self.terminal_memory = np.zeros(self.max_mem)
        self.mem_counter = 0

    def store_transition(self, state, action, reward, next_state, done):
        index = self.mem_counter % self.max_mem

        self.action_memory[index] = action
        self.state_memory[index] = state
        self.reward_memory[index] = reward
        self.next_state_memory[index] = next_state
        self.terminal_memory[index] = done

        self.mem_counter += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_counter, self.max_mem)
        batch = np.random.choice(max_mem, batch_size)

        state = self.state_memory[batch]
        next_state = self.next_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        terminal = self.terminal_memory[batch]

        return state, actions, rewards, next_state, terminal


class TD3:
    def __init__(self, alpha, beta, max_size, tau, n_actions, input_shape, min_action_value, max_action_value):
        self.alpha = alpha  # use for actor network learning rate
        self.beta = beta  # use for critic network learning rate
        self.max_size = max_size  # replay buffer memory size
        self.tau = tau  # weight value
        self.noise_var = 0.1
        self.batch_size = 64
        self.gamma = 0.99  # discounted factor
        self.update_actor_iter = 2  # actor network updated every 10 times
        self.n_actions = n_actions
        self.input_shape = input_shape
        self.min_action_value = min_action_value
        self.max_action_value = max_action_value

        # build memory buffer
        self.replay_buffer = ReplayBuffer(self.max_size, self.input_shape, self.n_actions)

        # build network
        self.actor_network = ActorNetwork(n_actions=self.n_actions)
        self.actor_target_network = ActorNetwork(n_actions=self.n_actions)
        self.critic_network_1 = CriticNetwork()
        self.critic_target_network_1 = CriticNetwork()
        self.critic_network_2 = CriticNetwork()
        self.critic_target_network_2 = CriticNetwork()

        # compile network
        self.actor_network.compile(Adam(learning_rate=alpha))
        self.actor_target_network.compile(Adam(learning_rate=alpha))
        self.critic_network_1.compile(Adam(learning_rate=beta))
        self.critic_target_network_1.compile(Adam(learning_rate=beta))
        self.critic_network_2.compile(Adam(learning_rate=beta))
        self.critic_target_network_2.compile(Adam(learning_rate=beta))

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.store_transition(state, action, reward, next_state, done)

    def choose_action(self, state):
        state = tf.convert_to_tensor([state])
        action = self.actor_network(state)

        # add noise for exploration
        action += tf.random.normal(shape=action.shape, mean=0.0, stddev=self.noise_var)

        # clip action to avoid action values small than min_action_value
        # or big than max_action_value
        action *= self.max_action_value
        action = tf.clip_by_value(action, self.min_action_value, self.max_action_value)

        return action[0]

    def update_weight(self):
        # update actor
        new_actor_weight = []

        for new_weight, old_weight in zip(self.actor_network.get_weights(), self.actor_target_network.get_weights()):
            new_actor_weight.append(new_weight * self.tau + old_weight * (1 - self.tau))

        self.actor_target_network.set_weights(new_actor_weight)

        # update critic 1
        new_critic_1_weight = []

        for new_weight, old_weight in zip(self.critic_network_1.get_weights(),
                                          self.critic_target_network_1.get_weights()):
            new_critic_1_weight.append(new_weight * self.tau + old_weight * (1 - self.tau))

        self.critic_target_network_1.set_weights(new_critic_1_weight)

        # update critic 2
        new_critic_2_weight = []

        for new_weight, old_weight in zip(self.critic_network_2.get_weights(),
                                          self.critic_target_network_2.get_weights()):
            new_critic_2_weight.append(new_weight * self.tau + old_weight * (1 - self.tau))

        self.critic_target_network_2.set_weights(new_critic_1_weight)

    def learn(self):
        if self.replay_buffer.mem_counter < self.batch_size:
            return

        state, action, reward, next_state, done = self.replay_buffer.sample_buffer(batch_size=self.batch_size)

        state = tf.convert_to_tensor(state, dtype=tf.float32)
        action = tf.convert_to_tensor(action, dtype=tf.float32)
        reward = tf.convert_to_tensor(reward, dtype=tf.float32)
        next_state = tf.convert_to_tensor(next_state, dtype=tf.float32)

        # learn critic 1
        with tf.GradientTape(persistent=True) as tape:
            s_a = tf.concat([state, action], axis=1)

            n_a = self.actor_target_network(next_state)
            n_a += np.clip(np.random.normal(scale=0.2), -0.3, 0.3)

            n_s_a = tf.concat([next_state, n_a], axis=1)

            q_value_1 = tf.squeeze(self.critic_network_1(s_a), axis=1)
            q_value_2 = tf.squeeze(self.critic_network_2(s_a), axis=1)

            n_q_value_1 = tf.squeeze(self.critic_target_network_1(n_s_a), axis=1)
            n_q_value_2 = tf.squeeze(self.critic_target_network_2(n_s_a), axis=1)

            min_n_q_value = np.minimum(n_q_value_1, n_q_value_2)

            td_target = reward + self.gamma * min_n_q_value * (1 - done)

            critic_1_loss = MSE(td_target, q_value_1)
            critic_2_loss = MSE(td_target, q_value_2)

        critic_1_gradient = tape.gradient(critic_1_loss, self.critic_network_1.trainable_variables)
        critic_2_gradient = tape.gradient(critic_2_loss, self.critic_network_2.trainable_variables)

        self.critic_network_1.optimizer.apply_gradients(
            zip(critic_1_gradient, self.critic_network_1.trainable_variables))
        self.critic_network_2.optimizer.apply_gradients(
            zip(critic_2_gradient, self.critic_network_2.trainable_variables))

        if self.replay_buffer.mem_counter % self.update_actor_iter != 0:
            return

        # learn actor
        with tf.GradientTape() as tape:
            action = self.actor_network(state)
            s_a = tf.concat([state, action], axis=1)
            loss = -self.critic_network_1(s_a)
            loss = tf.math.reduce_mean(loss)

        gradient = tape.gradient(loss, self.actor_network.trainable_weights)
        self.actor_network.optimizer.apply_gradients(zip(gradient, self.actor_network.trainable_weights))

        self.update_weight()
