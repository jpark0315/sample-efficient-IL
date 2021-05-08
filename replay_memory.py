import random
import numpy as np
from operator import itemgetter

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def push_batch(self, batch):
        if len(self.buffer) < self.capacity:
            append_len = min(self.capacity - len(self.buffer), len(batch))
            self.buffer.extend([None] * append_len)

        if self.position + len(batch) < self.capacity:
            self.buffer[self.position : self.position + len(batch)] = batch
            self.position += len(batch)
        else:
            self.buffer[self.position : len(self.buffer)] = batch[:len(self.buffer) - self.position]
            self.buffer[:len(batch) - len(self.buffer) + self.position] = batch[len(self.buffer) - self.position:]
            self.position = len(batch) - len(self.buffer) + self.position

    def sample(self, batch_size):
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        batch = random.sample(self.buffer, int(batch_size))
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def sample_all_batch(self, batch_size):
        idxes = np.random.randint(0, len(self.buffer), batch_size)
        batch = list(itemgetter(*idxes)(self.buffer))
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def return_all(self):   
        return self.buffer

    def __len__(self):
        return len(self.buffer)


    

class MyMemory:
    """
    Inherits Memory from other agent
    """
    def __init__(self, capacity, buffer, args):
        self.capacity = capacity
        self.buffer = buffer
        self.masked_buffer = []
        self.position = 0
        self.create_masks(args)

    def create_masks(self, args):
        batch = random.sample(self.buffer, len(self.buffer))
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        masks = np.where(reward>= args.reward_threshold)[0]

        masked_state, masked_action, masked_reward, masked_next_state, masked_done = (
            state[masks], action[masks], reward[masks], next_state[masks], done[masks]
            )
        for s,a,r,n,d in zip(masked_state, masked_action, masked_reward, masked_next_state, masked_done):
            self.masked_buffer.append((s,a,r,n,d))

        print('Number of masked pairs : ', len(self.masked_buffer))

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity


    def sample(self, batch_size):
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        batch = random.sample(self.buffer, int(batch_size))
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def sample_masked(self, batch_size):
        if batch_size > len(self.masked_buffer):
            batch_size = len(self.masked_buffer)
        batch = random.sample(self.masked_buffer, int(batch_size))
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def return_all(self):   
        return self.buffer

    def __len__(self):
        return len(self.buffer)


