import random

__author__ = 'Aron'


class ReplayMemory:

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, transition):
        self.memory += [transition]

        if len(self.memory) > self.capacity:
            self.memory = self.memory[1:]

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
