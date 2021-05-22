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
		ppo architecture
		do bc architecture separately?
	model perfect
	discriminator perfect
		-model size
		-train reps per it
		-random style loss
		-grad pen
		-remember past samples?
	normalize state for bc? 


TODO:

	train discriminator on also the real states
	try increasing model logprobs as well

	try different model penalty
	try reward for control penalty 
	lipshitz policy 
	smaller policy lr 

	discriminator regularization:
	try NOT using first rollout states 
	try regularizing discriminator with bad samples on both sides
	try regularizing discriminator with random states on both sides 
	try increaseing score of behavior policy very slightly(small batch or weight) just like CQL
	try starting from random state
	try regularizing discriminator with both lipshitz and gradient penalty 
	try regularizing with ensemble discriminator penalty? 
		discriminator will be less certain on states that are new? 
		Maximise certainty of the discriminator means that it is sure the state is good
	try 32 units with higher lipshitz 
	try weighting the samples with the penalty term 
	try a replay buffer for the states 
	try small behavioral cloning with the behavior policy
	try sigmod loss like gail or tanh 

	try like Rmax, cut when penalty big
	try bounding discirminator 
Realized:
	Geometric sampling works a LOT better for BC
	State only discriminator works a bit better for BC additional ppo
	logprob seems slightly better than mse on hopper
	Model error-rate is in decreasing order, env.reset-> expert -> random state sampling
		-which means it's the policy is indeed causing the model's uncertainty in the horizon
		-so one might think that I can somehow fix that

	I think generally state only is better? and

	Also, with penlam1 (small penalty) is just enough pessimism, too much is poison
		and bc lamda2. but for KL, larger lipshitz of 0.05 with pen1 seems to be good,
		maybe because the discriminator is not as strong for KL unless it goes above 0.05 
		I think whats happening is 0.05 for KL is right because the ratio between reward vs pessimism
		is just to overwhelming for small lipshitz 

	so it seems like with high penality the agent overfits to the demonstration, and the
	test penalty is higher, even though the train penalty is lower, but not when you are starting
	from a random state, you need that penalty there 

"""

class Algorithm:
	"""
	Solver class for my algorithm
	"""
	def __init__(self, args, logger, env):

		#self.actor = Actor(args, logger)
		self.critic = Discrim(args, logger)
		self.args = args
		self.env = env
		self.logger = logger

		#self.obs, self.acts, self.n_obs, self.n_acts = get_expert(env_name='Hopper-v2')

		#self.normalizer = Normalizer(self.obs, self.n_obs)
		#self.obs, self.n_obs = self.normalizer.normalize(self.obs), self.normalizer.normalize(self.n_obs)

		self.model = Ensemble_Model(state_size = 17,
		        action_size =6, logger = logger)

	def train_model(self, buffer = None, include_expert = True, env_name = "Walker2d-v2"):
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

# def get_model_and_data(env_name = 'HalfCheetah-v2'):
# 	env = gym.make(env_name)

# 	obs,acts,n_obs, n_acts = get_expert(env_name = env_name)
# 	buffer = get_data(env_name = env_name)
# 	states, actions, _,next_states,_ = buffer.sample(len(buffer))
# 	states, actions = np.concatenate([states, obs], 0), np.concatenate([actions, acts], 0)
# 	next_states = np.concatenate([next_states, n_obs], 0)

# 	if env_name == 'Hopper-v2':
# 		ids =[2, 1, 4, 6, 0]
# 	elif env_name == 'HalfCheetah-v2':
# 		ids = [6, 2, 0, 4, 3]
# 	algo.model.load(states, actions,next_states,  ids)
# 	print(algo.model.validate(states[:1000], actions[:1000], next_states[:1000]))

# 	return env,algo.model, states[:-990],states[-990:], actions[:-990], actions[-990:]

def get_model_and_data(env_name = 'Walker2d-v2'):
	env = gym.make(env_name)

	obs,acts, n_obs,_ = get_expert(env_name = env_name)
	buffer = get_data(env_name = env_name)
	states, actions, _,next_states,_ = buffer.sample(len(buffer))
	states, actions = np.concatenate([states, obs], 0), np.concatenate([actions, acts], 0)
	next_states = np.concatenate([next_states, n_obs], 0)

	if env_name == 'Hopper-v2':
		ids =[2, 1, 4, 6, 0]
	elif env_name == 'HalfCheetah-v2':
		#ids = [6, 2, 0, 4, 3]
		ids = [2, 6, 4, 1, 5]
	elif env_name == 'Walker2d-v2':
		ids = [2, 4, 1, 0, 6]

	algo.model.load(states, actions,next_states,  ids)
	print(algo.model.validate(states[:1000], actions[:1000], next_states[:1000]))
	print(algo.model.validate(obs, acts, n_obs))
	return env, algo.model, states, obs, actions, acts

logger = Logger()
args = Args()
algo = Algorithm(args, logger, env = None)
# algo.train_model()
env, model,states, e_states, actions, e_actions = get_model_and_data()
print(env, len(e_states))

#algo.train_model()


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
# , gradpen/nograd-en, remember/noremember, numtrain10/5
#try with new model, determintic false, 2,5 horizon, penalty 0.1,0.3/0.5, 
#try with deterministic true, other/more expert data, bclamda 3, hybrids
#try different penalty, (logprob, )
lipschitz_ = [0.05]
units_ = [64]
parallel_ = [5000]
horizon_ = [10]
start_state_ = [ 'hybrid3', 'bad']

d_loss = [ 'cql', 'gail', 'linear']
grad_pen_ = [False, True]
num_steps_ = [10]
remember_ = [False] 
deterministic_ = [True]
orthogonal_reg_ =[True]

bc_lamda_ = [2]
penalty_lamda_ = [0.2,0.4]
include_buffer_ = [False]

not_use_first_state_ = [False]
bad_both_sides_ = [False]
random_both_sides_ = [False]
control_penalty_ = [0]
tanh_ = [True]
#loss_ = ['MSE', 'logprob']
#geometric_ = [True,False]
#bclamda 2,3,4 d_loss linear kl, penalty_lamda 0,1, lipshitz 0.05 0.03
params = list(product(lipschitz_, parallel_, horizon_, start_state_, d_loss, grad_pen_, num_steps_, remember_,
		bc_lamda_, penalty_lamda_, include_buffer_, units_, deterministic_,
		not_use_first_state_, bad_both_sides_, random_both_sides_, orthogonal_reg_, control_penalty_, tanh_))

for i, param in enumerate(params[10:12]):
	(lipschitz, parallel, horizon, start_state, loss, grad_pen, num_steps, remember,
		bc_lamda, penalty_lamda, include_buffer, units, deterministic,
		not_use_first_state, bad_both_sides, random_both_sides, orthogonal_reg, control_penalty, tanh) = param

	logger = Logger()
	discrim = SmallD_S(env, logger, s = env.observation_space.shape[0],lipschitz = lipschitz, loss = loss, grad_pen = grad_pen,
		remember = remember, num_steps = num_steps, units = units)

	
	ppo  = PPO(logger,state_dim =env.observation_space.shape[0], action_dim = env.action_space.shape[0],
	 bc_loss = 'logprob' , parallel = parallel, horizon = horizon, geometric = False,
	bc_lamda = bc_lamda, orthogonal_reg = orthogonal_reg, tanh = tanh )

	# string = 'loss{}parallel{},horizon{},remember{},bc_lamda{},penalty_lamda{},include_buffer{}det{}'.format(
	# loss,parallel, horizon, remember, bc_lamda, penalty_lamda, include_buffer, deterministic
	# )
	string = 'penboth,cql,lips0.5,start{}d_loss{},pen{},gradpen{}'.format(
		start_state,loss, penalty_lamda,grad_pen, 
		)
	print(string)
	try:
		algo2(ppo, discrim, model, env, states, actions, e_states,e_actions, logger, s_a = False,
		update_bc = True, start_state = start_state, 
		penalty_lamda = penalty_lamda, include_buffer = include_buffer, deterministic = deterministic,
		not_use_first_state = not_use_first_state, 
		bad_both_sides = bad_both_sides, 
		random_both_sides = random_both_sides,
		control_penalty = control_penalty)
		logger.plot('may22/'+string)
	except KeyboardInterrupt:
		logger.plot('may21/'+string)

