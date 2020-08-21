import numpy as np
import random


class Memory:
    capacity = None

    def __init__(
            self,
            capacity,
            length=None,
            state_map=None,
            state_users=None,
            action=None,
            reward=None,
            new_state_map=None,
            new_state_users=None,
            done=None,
    ):
        self.capacity = capacity
        self.length = 0
        self.states_map = state_map
        self.states_users = state_users
        self.actions = action
        self.rewards = reward
        self.new_states_map = new_state_map
        self.new_states_users = new_state_users
        self.dones = done

    def push(self, state_map, state_users, action, reward, new_state_map, new_state_users, done):
        if self.length == 0:
            self.states_map = [state_map]
            self.states_users = [state_users]
            #print(f"self.states.shape = {self.states.shape}")
            self.actions = action
            self.rewards = reward
            self.new_states_map = [new_state_map]
            self.new_states_users = [new_state_users]
            self.dones = done
        else:
            self.states_map = np.vstack((self.states_map, [state_map]))
            self.states_users = np.vstack((self.states_users, [state_users]))
            #print(f"self.states.shape = {self.states.shape}")
            self.actions = np.vstack((self.actions, action))
            self.rewards = np.vstack((self.rewards, reward))
            self.new_states_map = np.vstack((self.new_states_map, [new_state_map]))
            self.new_states_users = np.vstack((self.new_states_users, [new_state_users]))
            self.dones = np.vstack((self.dones, done))

        self.length = self.length + 1

        if self.length > self.capacity:
            self.states_map = np.delete(self.states_map, (0), axis=0)
            self.states_users = np.delete(self.states_users, (0), axis=0)
            self.actions = np.delete(self.actions, (0), axis=0)
            self.rewards = np.delete(self.rewards, (0), axis=0)
            self.new_states_map = np.delete(self.new_states_map, (0), axis=0)
            self.new_states_users = np.delete(self.new_states_users, (0), axis=0)
            self.dones = np.delete(self.dones, (0), axis=0)
            self.length = self.length - 1

    def sample(self, batch_size):
        if self.length >= batch_size:
            idx = random.sample(range(0, self.length), batch_size)
            """
            state = self.states[idx, :]
            action = self.actions[idx, :]
            reward = self.rewards[idx, :]
            new_state = self.new_states[idx, :]
            done = self.dones[idx, :]

            #print(f"self.states.shape = {self.states.shape}")

            """
            states_map = self.states_map[idx]
            states_users = self.states_users[idx]
            actions = self.actions[idx]
            rewards = self.rewards[idx]
            new_states_map = self.new_states_map[idx]
            new_states_users = self.new_states_users[idx]
            dones = self.dones[idx]
            
            return list([states_map, states_users, actions, rewards, new_states_map, new_states_users, dones])
        else:
            return None

