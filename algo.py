from actor import Actor
from discriminator import Discrim
from model import Ensemble_Model
from utils import *
from main import Args 

import numpy as np 
import torch 
from itertools import product 
import pickle 
import gym 

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

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

		self.obs, self.acts, self.n_obs, self.n_acts = get_expert(env_name='PBHopper')

		self.normalizer = Normalizer(self.obs, self.n_obs)
		self.obs, self.n_obs = self.normalizer.normalize(self.obs), self.normalizer.normalize(self.n_obs)

		self.model = Ensemble_Model(state_size = args.state_dim, 
		        action_size = args.act_dim, logger = logger)

	def train_model(self):

		states, actions, next_states, next_actions = get_expert(env_name='Hopper-v2')
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

class Normalizer:

	def __init__(self, states, next_states):

		shift = -np.mean(states, 0)
		scale = 1.0 / (np.std(states, 0) + 1e-3)

		self.shift = shift
		self.scale = scale 

	def normalize(self, observation):
		return (observation+ self.shift) * self.scale 

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

# obs,acts,n_obs, n_acts = get_expert()
try:
	import pybulletgym

	env = gym.make('HopperMuJoCoEnv-v0')
except:
	env = gym.make('Hopper-v2')

logger = Logger()
args = Args()
algo = Algorithm(args, logger, env)

try:
	algo.train(1,1,1)
except KeyboardInterrupt:
	logger.plot('4')




