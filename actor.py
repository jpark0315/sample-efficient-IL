import numpy as np 
import random 
import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.distributions import MultivariateNormal

import gym 
from tqdm import trange

from torchutils import orthogonal_regularization, get_entropy, softmax 
from loss import policy_loss

LOG_STD_MIN = -5
LOG_STD_MAX = 2
EPS = np.finfo(np.float32).eps
class Actor(nn.Module):

	def __init__(self, args, logger):
		super().__init__()
		#state_dim, hidden_dim, action_dim = args.state_dim, args.policy_hidden_dim, args.act_dim
		state_dim, hidden_dim , action_dim = 11,256,3
		self.trunk = nn.Sequential(
			nn.Linear(state_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim), 
			nn.ReLU(),
			nn.Linear(hidden_dim, action_dim*2)
			)
		self.ad = action_dim
		#self.loss_fn = policy_loss[args.policy_loss_fn]
		self.args = args 
		self.logger = logger
		# self.optim = torch.optim.Adam(self.parameters(),
		# 		lr = args.policy_lr)
		self.optim = torch.optim.Adam(self.parameters(),
				lr = 3e-4)
		self.MSE, self.L1 = nn.MSELoss(), nn.L1Loss()

	def get_dist_and_mode(self, states):
		assert len(states.shape) == 2
		out = self.trunk(states)
		mu, log_std = out[:,:self.ad], out[:,self.ad:]
		mode = torch.tanh(mu)
		log_std = torch.tanh(log_std)
		log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)

		std = torch.exp(log_std)

		dist = torch.distributions.Normal(mu, std)

		assert LOG_STD_MAX > LOG_STD_MIN
		return dist, mode 

	def get_log_prob(self, states, actions):

		dist, _ = self.get_dist_and_mode(states)
		log_probs = dist.log_prob(actions)
		return log_probs 

	def forward(self, states):
		if not isinstance(states, torch.Tensor):
			states = torch.FloatTensor(states)
		dist, mode = self.get_dist_and_mode(states)
		samples = dist.rsample()
		log_probs = dist.log_prob(samples)

		return mode, samples, log_probs 

	def gradient_step(self, expert_data, critic):

		"""
		Actor takes one adversartial gradient step 
			args expert_data: tuple/list(states,actions,next_states, next_actions)
		
		"""
		expert_states, expert_actions, expert_next_states, expert_next_actions = expert_data 
		expert_states, expert_actions, expert_next_states, expert_next_actions = (torch.FloatTensor(expert_states),
			torch.FloatTensor(expert_actions), torch.FloatTensor(expert_next_states), torch.FloatTensor(expert_next_actions))
		expert_initial_states = expert_states 
	
		_, policy_next_actions, _ = self(expert_next_states)
		_, policy_initial_actions, _ = self(expert_initial_states)

		expert_init_inputs = torch.cat((expert_initial_states, policy_initial_actions), 1)
		expert_inputs = torch.cat((expert_states, expert_actions), 1)
		expert_next_inputs = torch.cat((expert_next_states, policy_next_actions),1)
		expert_real_next_inputs = torch.cat((expert_next_states, expert_next_actions),1)

		expert_nu_0 = critic(expert_init_inputs)
		expert_nu = critic(expert_inputs)
		expert_nu_next = critic(expert_next_inputs)
		expert_real_nu_next = critic(expert_real_next_inputs)


		loss = self.loss_fn(expert_nu_0, expert_nu, expert_nu_next,expert_real_nu_next, self.args)
		
		if self.args.policy_orthogonal_reg:
			reg = orthogonal_regularization(self, reg_coef = self.args.reg_coef)
			loss = loss + reg 

		self.optim.zero_grad()
		loss.backward()
		self.optim.step()

		with torch.no_grad():
			cur_expert_diff = expert_nu - expert_nu_0 
			expert_diff = expert_nu - expert_nu_next 
			weights = softmax(expert_diff)
			entropy = get_entropy(weights)

		self.logger.log('Weight Entropy', entropy.detach().item())
		self.logger.log('Average cur diff', cur_expert_diff.mean().detach().item())
		self.logger.log('Average diff', expert_diff.mean().detach().item())
		self.logger.log('Max diff', expert_diff.max().detach().item())
		self.logger.log('Min Diff', expert_diff.mean().detach().item())
		self.logger.log('nu_0', expert_nu_0.mean().detach().item())
		self.logger.log('Max nu_0', expert_nu_0.max().detach().item())
		self.logger.log('Expert_nu', expert_nu.mean().detach().item())
		self.logger.log('Max Expert_nu', expert_nu.max().detach().item())
		#self.logger.log('Policy Log Probs', next_log_probs.mean().detach().item())
		self.logger.log('Policy reg', reg.mean().detach().item())
		self.logger.log(self.args.policy_loss_fn+' p_loss', loss.detach().item())


	def bc_step(self, expert_data):
		"""
		Actor takes one behavioral cloning step
			args expert_data: tuple/list(states,actions,next_states, next_actions)
		"""
		# expert_states, expert_actions, expert_next_states, expert_next_actions = expert_data 
		# expert_states, expert_actions, expert_next_states, expert_next_actions = (torch.FloatTensor(expert_states),
		# 	torch.FloatTensor(expert_actions), torch.FloatTensor(expert_next_states), torch.FloatTensor(expert_next_actions))

		expert_states, expert_actions = expert_data
		expert_states, expert_actions= (torch.FloatTensor(expert_states),
			torch.FloatTensor(expert_actions))

		_, pred_actions, pred_logprobs = self(expert_states)
		#_, pred_next_actions, pred_next_logprobs = self(expert_next_states)

		loss = self.bc_loss(pred_actions, expert_actions, expert_states)

		self.optim.zero_grad()
		loss.backward()
		self.optim.step() 
		self.logger.log('BC loss', loss.item())
		print(loss.item())


	def bc_loss(self, pred, label, states ):

		# if self.args.bc_clamp:
		# 	pred = torch.clamp(pred, self.action_space.low[0],self.action_space.high[0])
		# 	label = torch.clamp(label.float(), self.action_space.low[0],self.action_space.high[0])

		# if self.args.bc_loss_fn == 'L1':
		# 	loss = self.L1(pred, label)
		# elif self.args.bc_loss_fn == 'L2':
		# 	loss = self.MSE(pred, label)
		# elif self.args.bc_loss_fn == 'Hybrid':
		# 	loss = 0.5*(self.MSE(pred, label) + self.L1(pred, label))
		# elif self.args.bc_loss_fn == 'MLE':
		# 	log_prob = self.get_log_prob(states,label)
		# 	loss = -log_prob.mean()
		# log_prob = self.get_log_prob(states,label)
		# loss = -log_prob.mean()
		#loss = 0.5*(self.MSE(pred, label) + self.L1(pred, label))
		loss = self.MSE(pred, label)
		return loss 

	def train_bc(self, expert_states, expert_actions, num_steps = 10000, batch_size = 256):
		from tqdm import trange
		for i in trange(num_steps):
			idx = np.random.permutation(expert_states.shape[0])[:batch_size]
			s_b, a_b = expert_states[idx], expert_actions[idx]

			self.bc_step((s_b, a_b))





"""
TOdo:
validations +holdout 
ensemble
regulariztion(lipshitz, )
scaling 
"""

LOG_STD_MIN = -5
LOG_STD_MAX = 2
EPS = np.finfo(np.float32).eps
class Solo_BC(nn.Module):
	"""
	BC class for non-ensemble agent 
	"""
	activations = {
	'Tanh': nn.Tanh,
	'ReLU': nn.ReLU,
	#'Swish': Swish
	}

	def __init__(self, state_dim,act_dim, action_space = 'Box'):
		super().__init__()

		#self.args = args 
		self.action_space = action_space
		self.ad = act_dim 
		# self.net = nn.Sequential(
		# 	nn.Linear(state_dim, args.critic_hidden_size),
		# 	nn.ReLU(),
		# 	nn.Linear(args.critic_hidden_size, args.critic_hidden_size),
		# 	nn.ReLU(),
		# 	nn.Linear(args.critic_hidden_size, act_dim)
		# 	)
		self.trunk = nn.Sequential(
			nn.Linear(state_dim, 256),
			nn.ReLU(),
			nn.Linear(256, 256),
			nn.ReLU(),
			nn.Linear(256, act_dim * 2)
			)

		self.optim = torch.optim.Adam(self.parameters(), lr = 3e-4) 
		self.optim_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optim, gamma = 0.99)

		self.MSE, self.L1 = nn.MSELoss(), nn.L1Loss()


	def get_dist_and_mode(self, states):
		assert len(states.shape) == 2
		out = self.trunk(states)
		mu, log_std = out[:,:self.ad], out[:,self.ad:]
		mode = torch.tanh(mu)
		log_std = torch.tanh(log_std)
		log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)

		std = torch.exp(log_std)

		dist = torch.distributions.Normal(mu, std)

		assert LOG_STD_MAX > LOG_STD_MIN
		return dist, mode 

	def get_log_prob(self, states, actions):

		dist, _ = self.get_dist_and_mode(states)
		log_probs = dist.log_prob(actions)
		return log_probs.sum(1)

	def forward(self, states):
		if not isinstance(states, torch.Tensor):
			states = torch.FloatTensor(states)
		dist, mode = self.get_dist_and_mode(states)
		samples = dist.rsample()
		log_probs = dist.log_prob(samples)

		return mode, samples, log_probs 


	def loss_fn(self, input, label):
		"""
		Label, pred always torch.Tensor
		"""
		#loss = self.MSE(pred,label)
		loss = self.get_log_prob(input, label) 

		return loss 

	def step(self, state, label, return_loss = False):
		"""
		Takes one gradient step with minibatch
		"""
		if len(state.shape) == 1:
			assert len(label.shape) == 1
			state, label = state.reshape(1,-1), label.reshape(1,-1)

		state, label = torch.FloatTensor(state), torch.FloatTensor(label)
		loss = -self.get_log_prob(state, label).mean()

		if return_loss:
			return loss.detach().item()

		self.optim.zero_grad()
		loss.backward()
		self.optim.step() 


		#assert len(state.shape) == len(label.shape) and len(pred.shape) == len(label.shape)


		return loss.item()

	def train(self,state,action, train_step = 1000, eva = False):
		losses, scores = [], []
		state, action = torch.FloatTensor(state), torch.FloatTensor(action)
		for i in trange(train_step):
			idxs = np.random.permutation(state.shape[0])[:256]
			state_,action_ = state[idxs], action[idxs]
			loss = self.step(state_, action_)

			if i % 100 == 0 and eva:
				score, time = evaluate(self, None, env)
				print(score)
				scores.append(score)

			losses.append(loss)

		return losses, scores

	def train_(self, memory, inputs = None, labels=None, without_replacement = False):
		"""
		Param input: np.ndarray (N x state_space)
		Param label: np.ndarray (N x action_space)
		"""
		bs, losses = 256, []
		from tqdm import trange 
		if inputs is None:
			for steps in trange(10000):
				inputs,labels,_,_,_ = memory.sample(bs)
				loss = self.step(inputs, labels)

				if steps % 100 == 0:
					print(loss)
				losses.append(loss)
			return losses 

		else: 
			timesteps = int(len(inputs)/bs)
			for steps in trange(self.args.num_policy_epochs * timesteps):
				idxs = np.random.permutation(inputs.shape[0])[:bs]
				input = inputs[idxs]
				label = labels[idxs]

				loss = self.step(input, label)
				losses.append(loss)

			return losses


	def act(self, state):
		if len(state.shape) == 1:
			state = state.reshape(1,-1)
		action = self(state).detach()

		return action 

	def save(self):
		import os 
		if not os.path.exists('bc/models/'):
			os.makedirs('bc/models/')        
		path = "bc/models/modelnew_"
		torch.save(self.state_dict(), path) 

	def load(self):
		"""
		Loads Model params and fits scaler mu/sigma since 
		calling this means scaler wasn't fit before 
		"""
		print('Loading Model...')

		path = "bc/models/modelnew_"
		self.load_state_dict(torch.load(path))
 


class Solo_BC_(nn.Module):
	"""
	BC class for non-ensemble agent 
	"""
	activations = {
	'Tanh': nn.Tanh,
	'ReLU': nn.ReLU,
	#'Swish': Swish
	}

	def __init__(self, args, action_space = 'Box'):
		super().__init__()

		self.args = args 
		self.action_space = action_space

		activ = self.activations[args.policy_activation]

		self.policy = nn.ModuleList([nn.Linear(args.state_dim, 
				args.policy_hidden_dim), activ()])
		for _ in range(args.policy_num_hidden):
			self.policy.extend([nn.Linear(args.policy_hidden_dim,args.policy_hidden_dim), 
					activ()])

		if isinstance(self.action_space, gym.spaces.Box):
			self.policy.extend([nn.Linear(args.policy_hidden_dim, action_space.shape[0])])

		elif isinstance(self.action_space, gym.spaces.discrete.Discrete):
			self.policy.extend([nn.Linear(args.policy_hidden_dim, action_space.n)])


		self.optim = torch.optim.Adam(self.parameters(), lr = args.policy_lr,  weight_decay=args.policy_weight_decay)
		self.optim_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optim, gamma = 0.99)

		self.MSE, self.L1 = nn.MSELoss(), nn.L1Loss()

	def forward(self, x):
		"""
		Always takes two dimensional matrices (n x state_space)
		"""
		assert len(x.shape) == 2
		if not isinstance(x, torch.Tensor):
			x = torch.FloatTensor(x)

		for net in self.policy:
			x = net(x)

		return x 

	def loss_fn(self, pred, label):
		"""
		Label, pred always torch.Tensor
		"""
		if isinstance(self.action_space, gym.spaces.Box):
			pred = torch.clamp(pred, self.action_space.low[0],self.action_space.high[0])
			label = torch.clamp(label.float(), self.action_space.low[0],self.action_space.high[0])

			if self.args.policy_loss_fn == 'L1':
				loss = self.L1(pred, label)
			elif self.args.policy_loss_fn == 'L2':
				loss = self.MSE(pred, label)
			elif self.args.policy_loss_fn == 'Hybrid':
				loss = 0.5*(self.MSE(pred, label) + self.L1(pred, label))

		elif isinstance(self.action_space, gym.spaces.discrete.Discrete):
			loss = F.cross_entropy(pred, label.flatten().long())

		return loss 

	def step(self, state, label, return_loss = False):
		"""
		Takes one gradient step with minibatch
		"""
		if len(state.shape) == 1:
			assert len(label.shape) == 1
			state, label = state.reshape(1,-1), label.reshape(1,-1)

		state, label = torch.FloatTensor(state), torch.FloatTensor(label)
		pred = self(state)
		loss = self.loss_fn(pred,label)

		if return_loss:
			return loss.detach().item()

		self.optim.zero_grad()
		loss.backward()
		self.optim.step() 

		assert len(state.shape) == len(label.shape) and len(pred.shape) == len(label.shape)
		if isinstance(self.action_space, gym.spaces.Box):
			assert pred.shape == label.shape 


		return loss.item()

	def train(self, inputs, labels, without_replacement = False):
		"""
		Param input: np.ndarray (N x state_space)
		Param label: np.ndarray (N x action_space)
		"""

		if without_replacement:
			bs, losses = self.args.BC_batch_size, []
			for epoch in range(self.args.num_policy_epochs):
				epoch_loss = []
				for start_pos in trange(0, inputs.shape[0], bs):
					input = inputs[start_pos:start_pos+bs]
					label = labels[start_pos:start_pos+bs]

					loss = self.step(input, label)
					epoch_loss.append(loss)

				#TODO validation

				losses.extend(epoch_loss)
				print('Epoch {} | Average Loss {}'.format(epoch, np.asarray(epoch_loss).mean()))

				if self.args.BC_lr_decay:
					self.optim_scheduler.step()

			return losses

		else:
			bs, losses = self.args.BC_batch_size, []
			timesteps = int(len(inputs)/bs)
			for steps in trange(self.args.num_policy_epochs * timesteps):
				idxs = np.random.permutation(inputs.shape[0])[:bs]
				input = inputs[idxs]
				label = labels[idxs]

				loss = self.step(input, label)
				losses.append(loss)

				if self.args.BC_lr_decay:
					self.optim_scheduler.step()

			return losses


	def act(self, state):
		if len(state.shape) == 1:
			state = state.reshape(1,-1)

		if isinstance(self.action_space, gym.spaces.Box):
			action = self(state).detach().numpy()
		elif isinstance(self.action_space, gym.spaces.discrete.Discrete):
			prob = nn.Softmax(1)(self(state))

			if state.shape[0] == 1:
				action = prob.argmax().detach().item()
			else: 
				action = prob.argmax(1, keepdims = True).detach().numpy()

		return action 

	def act_gradient(self, state):
		if len(state.shape) == 1:
			state = state.reshape(1,-1)

		if isinstance(self.action_space, gym.spaces.Box):
			action = self(state)
		elif isinstance(self.action_space, gym.spaces.discrete.Discrete):
			prob = nn.Softmax(1)(self(state))

			if state.shape[0] == 1:
				action = prob.argmax()
			else: 
				#action = prob.argmax(1, keepdims = True)
				action = prob.mean(1, keepdims = True)

		return action 

	def get_judged(self, critic, model, next_state_batch, state_batch, verbose = False, no_step = False):

		#Onestep Rollout Backprop
		learner_actions = self.act_gradient(state_batch)
		next_states_grad = model.predict_next_states_gradient(state_batch, learner_actions)
		n_learner_actions = self.act_gradient(next_states_grad)
		learner_sa = torch.cat((next_states_grad, n_learner_actions), 1)
		score_first = torch.mean(-critic(learner_sa))

		if no_step:
			return -score_first.detach().item() 

		self.optim.zero_grad()
		score_first.backward()
		self.optim.step()

		# #Next State backprop 
		# learner_actions = self.act_gradient(next_state_batch)
		# learner_sa = torch.cat((torch.FloatTensor(next_state_batch), learner_actions), 1)
		# score = torch.mean(-critic(learner_sa))


		# self.optim.zero_grad()
		# score.backward()
		# self.optim.step()

		# if verbose:
		# 	print('Judged Score {}'.format(-score.item()))

		return -score_first.item(), 1#-score.item()
