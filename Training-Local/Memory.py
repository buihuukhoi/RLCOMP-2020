import numpy as np
import random


class Memory:
    capacity = None

    def __init__(
            self,
            capacity,
            length=None,
            states=None,
            actions=None,
            rewards=None,
            new_states=None,
            dones=None,
    ):
        self.capacity = capacity
        self.length = 0
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.new_states = new_states
        self.dones = dones

    def push(self, state, action, reward, new_state, done):
        if self.states is None:
            self.states = state
            self.actions = action
            self.rewards = reward
            self.new_states = new_state
            self.dones = done
        else:
            self.states = np.vstack((self.states, state))
            self.actions = np.vstack((self.actions, action))
            self.rewards = np.vstack((self.rewards, reward))
            self.new_states = np.vstack((self.new_states, new_state))
            self.dones = np.vstack((self.dones, done))

        self.length = self.length + 1

        if self.length > self.capacity:
            self.states = np.delete(self.states, (0), axis=0)
            self.actions = np.delete(self.actions, (0), axis=0)
            self.rewards = np.delete(self.rewards, (0), axis=0)
            self.new_states = np.delete(self.new_states, (0), axis=0)
            self.dones = np.delete(self.dones, (0), axis=0)
            self.length = self.length - 1

    def sample(self, batch_size):
        if self.length >= batch_size:
            idx = random.sample(range(0, self.length), batch_size)
            state = self.states[idx, :]
            action = self.actions[idx, :]
            reward = self.rewards[idx, :]
            new_state = self.new_states[idx, :]
            done = self.dones[idx, :]
            """
            state = self.states[idx]
            action = self.actions[idx]
            reward = self.rewards[idx]
            new_state = self.new_states[idx]
            done = self.dones[idx]
            """
            return list([state, action, reward, new_state, done])
        else:
            return None

