from actor import Actor
from discriminator import *
from model import Ensemble_Model
from utils import *
from main import Args

import numpy as np
import torch
from itertools import product
import pickle
import gym
from ppo import *

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

"""
Things to get right:
	dataloading(geometric, torch, subsample, random, )
	initialization
	ppo details(advantage computation, architecture, etc )
	model perfect

TODO:
	train discriminator on also the real states
	try increasing model logprobs as well

	try different model penalty
Realized:
	Geometric sampling works a LOT better for BC
	State only discriminator works a bit better for BC additional ppo
	logprob seems slightly better than mse on hopper
	Model error-rate is in decreasing order, env.reset-> expert -> random state sampling
		-which means it's the policy is indeed causing the model's uncertainty in the horizon
		-so one might think that I can somehow fix that

	I think generally state only is better? and
"""

class Algorithm:
	"""
	Solver class for my algorithm
	"""
	def __init__(self, args, logger, env):

		self.actor = Actor(args, logger)
		self.critic = Discrim(args, logger)
		self.args = args
		self.env = env
		self.logger = logger

		#self.obs, self.acts, self.n_obs, self.n_acts = get_expert(env_name='Hopper-v2')

		#self.normalizer = Normalizer(self.obs, self.n_obs)
		#self.obs, self.n_obs = self.normalizer.normalize(self.obs), self.normalizer.normalize(self.n_obs)

		self.model = Ensemble_Model(state_size = args.state_dim,
		        action_size = args.act_dim, logger = logger)

	def train_model(self, buffer = None, include_expert = True, env_name = "Hopper-v2"):
		if buffer is None:
			buffer =  get_data(env_name = env_name)

		if include_expert:
			e_states, e_actions, e_next_states, e_next_actions = get_expert(env_name=env_name)

			states, actions, _,next_states,_ = buffer.sample(len(buffer))
			states, actions = np.concatenate([states, e_states], 0), np.concatenate([actions, e_actions], 0)
			next_states = np.concatenate([next_states, e_next_states], 0)

			delta_state = next_states - states
			inputs = np.concatenate((states, actions), axis = -1)
			labels = delta_state
			print('Num total data', len(inputs))

		self.model.train(inputs, labels)

	@staticmethod
	def evaluate(actor, normalizer, env, num_episodes=10, stats = 'mode'):

		total_timesteps = 0
		total_returns = 0

		for _ in range(num_episodes):
			state = normalizer.normalize(env.reset())
			done = False
			while not done:
				with torch.no_grad():
					if stats == 'mode':
						action, _, _ = actor(np.array([state]))
					else:
						_, action, _ = actor(np.array([state]))
				action = action[0].numpy()
				next_state, reward, done, _ = env.step(action)

				total_returns += reward
				total_timesteps += 1
				state = normalizer.normalize(next_state)

		return total_returns / num_episodes, total_timesteps / num_episodes


	def geometric_index(self):
		"""
		Return indices such that the previous states are weighted
		heavier than the next
		"""
		while True:
			idxs = np.random.geometric(1- 0.99, self.args.BC_batch_size)
			if (idxs<=len(self.obs)-1).sum() == self.args.BC_batch_size:
				return idxs


	def train(self,  param_keys, param, exp_id ,train_model = True):
		"""
		Args
			exp_id : experiment id for parallel experiments
			param_keys, param are parameters that are used to train
			in this experiment
		"""
		if train_model:
			self.train_model()
		else:
			self.model.load(states, actions, next_states, test = False, model_inds = [6, 2, 5, 1, 0])

		for i in range(self.args.num_train_steps):
			batch_idxs = self.geometric_index()
			expert_data = (self.obs[batch_idxs],
							self.acts[batch_idxs], self.n_obs[batch_idxs], self.n_acts[batch_idxs])

			#Adversarial step
			# self.critic.gradient_step(expert_data, self.actor)
			# self.actor.gradient_step(expert_data, self.critic)

			#BC step
			self.actor.bc_step(expert_data)

			if i % 100 == 0:
				print("Iter ", i)
				#print("Params: ", list(zip(param_keys, param)))
				avg_rewards, time = evaluate(self.actor, self.normalizer, self.env, stats = 'mode')
				print('Average Rewards: {}'.format(avg_rewards))
				self.logger.log('Avg Rewards', avg_rewards)
				self.logger.say()
				print()

		self.logger.say(all = True)
		#string = str(list(zip(param_keys, param)))
		#self.logger.plot(num = 'exp_id ' + string )
		self.logger.plot('just bc')


def experiment(args):
	"""
	Generate N combinations of parameters
	where params is a list of combinations,
	and N[a] is a tuple in the order of the keys
	of the param_dict

	Param_dict should have keys with the same spelling
	as variables in args
	"""
	param_dict = {
	'a' : [1,2] #...
	}
	param_keys = list(param_dict.keys())
	param_list = list(product(*param_dict.values()))

	#For parallelizing
	num_params = len(param_list)
	#Change the two below
	param_idx = num_params//2
	exp_id = 1

	param_list = param_list[:param_idx]

	for params in param_list:

		#modify args
		for key, param in zip(param_keys, params):
			setattr(args,key, param)

def get_model_and_data(env_name = 'Hopper-v2'):
	obs,acts,n_obs, n_acts = get_expert(env_name = env_name)
	buffer = get_data(env_name = env_name)
	states, actions, _,next_states,_ = buffer.sample(len(buffer))
	states, actions = np.concatenate([states, obs], 0), np.concatenate([actions, acts], 0)
	next_states = np.concatenate([next_states, n_obs], 0)

	algo.model.load(states, actions,next_states,  [2, 1, 4, 6, 0])

	return algo.model, states[:-990],states[-990:], actions[:-990], actions[-990:]


env = gym.make('Hopper-v2')

logger = Logger()
args = Args()
algo = Algorithm(args, logger, env)
model,states, e_states, actions, e_actions = get_model_and_data()


#two no bcs
#sa lipschitz0.05, s lipshitz 0.05, s lipshitz 0.05 + hor5

#to run massive hyperparam parallels :starting states, stateonly, stateaction, MSE,
	#hyperparams: parallel, horizon, lipshitz, lr, bc_train_step
########

# logger = Logger()
# discrim = SmallD_S(logger, s = 11,lipschitz = 0.05)
# #discrim = SmallD(logger, s = 11, a = 3, lipschitz = 0.05)
# ppo  = PPO(logger, bc_loss = "logprob", parallel = 2000, horizon = 5, geometric = True)

# try:
# 	algo2(ppo, discrim, model, env, states, e_states,e_actions, logger, s_a = False, update_bc = False)
# 	logger.plot('lp_fake_Sonlydis_bc,geoFalse,lips0.05,hor5')
# except:
# 	logger.plot('lp_fake_Sonlydis_bc,geoFalse,lips0.05,hor5')

#experiment with discrim trainstep, bc lamda , penalty ladma, include_buffer/no include
lipschitz_ = [0.03, 0.05]
parallel_ = [5000]
horizon_ = [10]
start_state_ = ['bad']
#bc_train_step_ = [1, 3, 5]
d_loss = ['linear', 'kl']

bc_lamda_ = [2,3]
penalty_lamda_ = [0,1]
include_buffer_ = [False]

#loss_ = ['MSE', 'logprob']
#bclamda 2,3,4 d_loss linear kl, penalty_lamda 0,1, lipshitz 0.05 0.03
params = list(product(lipschitz_, parallel_, horizon_, start_state_, d_loss,
		bc_lamda_, penalty_lamda_, include_buffer_))

for i, param in enumerate(params[3:6]):
	lipschitz, parallel, horizon, start_state, loss, bc_lamda, penalty_lamda, include_buffer = param

	logger = Logger()
	discrim = SmallD_S(logger, s = 11,lipschitz = lipschitz, loss = loss)
	#discrim = SmallD(logger, s = 11, a = 3, lipschitz = 0.05)
	ppo  = PPO(logger, bc_loss = "logprob", parallel = parallel, horizon = horizon, geometric = True,
	bc_lamda = bc_lamda)
	string = 'lp_fake_Sonlydis_bc, lips{},d_loss{} parallel{}, horizon{},penlam{},incbuf{},bclam{}'.format(
	lipschitz,loss, parallel, horizon, penalty_lamda, include_buffer, bc_lamda)
	try:
		algo2(ppo, discrim, model, env, states, actions, e_states,e_actions, logger, s_a = False,
		update_bc = True, start_state = start_state, penalty_lamda = penalty_lamda, include_buffer = include_buffer)
		logger.plot('may12/'+string)
	except KeyboardInterrupt:
		logger.plot('may12/'+string)
