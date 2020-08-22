import random
#from collections import deque


class Memory:
    def __init__(self, max_size):
        self.buffer = [None] * max_size
        self.max_size = max_size
        self.index = 0
        self.size = 0

    def append(self, state_map, state_users, action, reward, new_state_map, new_state_users, done):
        self.buffer[self.index] = (state_map, state_users, action, reward, new_state_map, new_state_users, done)
        self.size = min(self.size + 1, self.max_size)
        self.index = (self.index + 1) % self.max_size

    def sample(self, batch_size):
        indices = random.sample(range(self.size), batch_size)
        return [self.buffer[index] for index in indices]

"""
class Memory:
    def __init__(self, capacity):
        self.replay_memory = deque(maxlen=capacity)

    def update_replay_memory(self, state_map, state_users, action, reward, new_state_map, new_state_users, done):
        self.replay_memory.append((state_map, state_users, action, reward, new_state_map, new_state_users, done))

    def sample(self, batch_size):
        # convert deque to list can avoid time consuming when the deque becomes bigger
        minibatch = random.sample(list(self.replay_memory), batch_size)
        return minibatch
"""
