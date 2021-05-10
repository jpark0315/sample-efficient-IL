import numpy as np
from copy import deepcopy
from collections import OrderedDict
import random
import torch
import torch.nn as nn
from tqdm import trange
from loss import critic_loss , policy_loss
from torchutils import softmax

class SmallD(nn.Module):
    def __init__(self, logger, s = 3, a = 1,lipschitz = 0.1, loss = 'linear'):
        super().__init__()
        self.loss = loss
        self.lipschitz = lipschitz
        self.net = nn.Sequential(nn.Linear(s+a, 64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64,1))
        self.optim = torch.optim.Adam(self.parameters(), lr = 3e-4)
        self.logger = logger

        for p in self.parameters():
            p.data.clamp_(-self.lipschitz, self.lipschitz)
    def forward(self,s,a):

        if not isinstance(s,torch.Tensor):
            s,a = torch.FloatTensor(s), torch.FloatTensor(a)
        sa = torch.cat([s,a],1)
        return self.net(sa)

    def loss_fn(self, lscore, escore):
        loss = self.loss
        #KL naive
        if loss == 'kl':
            diff = (lscore-escore).mean()
            with torch.no_grad():
                weights = softmax(diff)
            loss = torch.sum(weights* diff)

        elif loss == 'linear':
            loss = (lscore - escore).mean()

        return loss

    def step(self, state_batch,action_batch, e_state_batch, e_action_batch):
        state_batch, action_batch = torch.FloatTensor(state_batch), torch.FloatTensor(action_batch)
        e_state_batch, e_action_batch = torch.FloatTensor(e_state_batch), torch.FloatTensor(e_action_batch)

        lscore = self(state_batch, action_batch)
        escore = self(e_state_batch, e_action_batch)
        loss = self.loss_fn(lscore, escore)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        for p in self.parameters():
            p.data.clamp_(-self.lipschitz, self.lipschitz)
        self.logger.log('discrim loss', loss.item())
        self.logger.log('escore', escore.mean().detach().item())
        self.logger.log('lscore', lscore.mean().detach().item())

        return loss.item()


    def train_discrim(self, e_states, e_actions, l_states, l_actions,num_steps = 10,  bs = 256):

        for _ in range(num_steps):
            idx = np.random.permutation(l_states.shape[0])[:bs]
            e_idx = np.random.permutation(e_states.shape[0])[:bs]

            l_state_batch, l_action_batch = l_states[idx], l_actions[idx]
            e_state_batch, e_action_batch = e_states[e_idx], e_actions[e_idx]

            loss = self.step(l_state_batch, l_action_batch, e_state_batch, e_action_batch)


class SmallD_S(nn.Module):
    def __init__(self, logger, s = 3, lipschitz = 0.1, loss = 'linear'):
        super().__init__()
        self.loss = loss
        self.lipschitz = lipschitz
        self.net = nn.Sequential(nn.Linear(s, 64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64,1))
        self.optim = torch.optim.Adam(self.parameters(), lr = 3e-4)
        self.logger = logger

        for p in self.parameters():
            p.data.clamp_(-self.lipschitz, self.lipschitz)
    def forward(self,s):

        if not isinstance(s,torch.Tensor):
            s= torch.FloatTensor(s)
        #sa = torch.cat([s,a],1)
        return self.net(s)

    def loss_fn(self, lscore, escore, loss = 'linear'):
        loss = self.loss 
        #KL naive
        if loss == 'kl':
            diff = (lscore-escore).mean()
            with torch.no_grad():
                weights = softmax(diff)
            loss = torch.sum(weights* diff)

        elif loss == 'linear':
            loss = (lscore - escore).mean()

        return loss

    def step(self, state_batch, e_state_batch):
        state_batch=  torch.FloatTensor(state_batch)
        e_state_batch= torch.FloatTensor(e_state_batch)

        lscore = self(state_batch)
        escore = self(e_state_batch)
        loss = self.loss_fn(lscore, escore)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        for p in self.parameters():
            p.data.clamp_(-self.lipschitz, self.lipschitz)
        self.logger.log('discrim loss', loss.item())
        self.logger.log('escore', escore.mean().detach().item())
        self.logger.log('lscore', lscore.mean().detach().item())

        return loss.item()


    def train_discrim(self, e_states, l_states, num_steps = 10,  bs = 256):

        for _ in range(num_steps):
            idx = np.random.permutation(l_states.shape[0])[:bs]
            e_idx = np.random.permutation(e_states.shape[0])[:bs]

            l_state_batch = l_states[idx]
            e_state_batch = e_states[e_idx]

            loss = self.step(l_state_batch, e_state_batch)


class Discrim(nn.Module):

	activations = {
	'Tanh': nn.Tanh,
	'ReLU': nn.ReLU,
	}

	def __init__(self, args, logger = None):
		super().__init__()

		self.logger = logger
		self.args = args
		# activ = self.activations[args.critic_activ]
		# self.critic = nn.ModuleList([nn.Linear(args.state_dim + args.act_dim,
		# 		args.critic_hidden_size), activ()])
		# for _ in range(args.num_critic_hidden_layers):
		# 	self.critic.extend([nn.Linear(args.critic_hidden_size,args.critic_hidden_size),
		# 			activ()])
		# self.critic.extend([nn.Linear(args.critic_hidden_size, 1, bias = args.discrim_bias)])

		self.critic = nn.Sequential(
			nn.Linear(args.state_dim+ args.act_dim, args.critic_hidden_size),
			nn.ReLU(),
			nn.Linear(args.critic_hidden_size, args.critic_hidden_size),
			nn.ReLU(),
			nn.Linear(args.critic_hidden_size, 1, bias = False)
			)

		# if args.lipshitz_clamp:
		# 	for p in self.parameters():
		# 		p.data.clamp_(-args.lipschitz, args.lipschitz)

		if args.rms:
			self.optim =  torch.optim.RMSprop(self.parameters(), lr=args.critic_lr)
		else:
			self.optim = torch.optim.Adam(self.parameters(), lr = args.critic_lr)

		self.critic_loss_fn = critic_loss[args.critic_loss_fn]

	def forward(self, x):

		return self.critic(x)

	def gradient_step(self, expert_data, actor):

		"""
		Critic takes one gradient step

			args expert_data: tuple/list(states,actions,next_states, next_actions)

		"""
		expert_states, expert_actions, expert_next_states, expert_next_actions = expert_data
		expert_states, expert_actions, expert_next_states, expert_next_actions = (torch.FloatTensor(expert_states),
			torch.FloatTensor(expert_actions), torch.FloatTensor(expert_next_states), torch.FloatTensor(expert_next_actions))
		expert_initial_states = expert_states

		with torch.no_grad():
			_, policy_next_actions, next_log_probs = actor(expert_next_states)
			_, policy_initial_actions, _ = actor(expert_initial_states)

		expert_init_inputs = torch.cat((expert_initial_states, policy_initial_actions), 1)
		expert_inputs = torch.cat((expert_states, expert_actions), 1)
		expert_next_inputs = torch.cat((expert_next_states, policy_next_actions),1)
		expert_real_next_inputs = torch.cat((expert_next_states, expert_next_actions),1)

		expert_nu_0 = self(expert_init_inputs)
		expert_nu = self(expert_inputs)
		expert_nu_next = self(expert_next_inputs)
		expert_real_nu_next = self(expert_real_next_inputs)


		loss = self.critic_loss_fn(expert_nu_0, expert_nu, expert_nu_next,expert_real_nu_next, self.args)

		self.optim.zero_grad()
		loss.backward()
		self.optim.step()

		if self.args.lipshitz_clamp:
			for p in self.parameters():
				p.data.clamp_(-self.args.lipschitz, self.args.lipschitz)

		self.logger.log(self.args.critic_loss_fn +' c_loss', loss.detach().item())
