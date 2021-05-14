import pickle
import numpy as np
import torch

def evaluate(actor, env, num_episodes=10, stats = 'mode', normalizer = None):

	total_timesteps = 0
	total_returns = 0

	for _ in range(num_episodes):
		if normalizer:
			state = normalizer.normalize(env.reset())
		else:
			state = env.reset()

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

			if normalizer:
				state = normalizer.normalize(next_state)
			else:
				state = next_state

	return total_returns / num_episodes, total_timesteps / num_episodes
class Logger:
	def __init__(self):
		self.dict = {}

	def log(self, key, value):
		if key in self.dict.keys():
			self.dict[key].append(value)
		else:
			self.dict[key] = [value]

	def plot(self, num = None):
		import matplotlib
		matplotlib.use("Qt5Agg")
		import matplotlib.pyplot as plt
		if num is None:
			for k,v in self.dict.items():
				plt.figure()
				plt.title(k+"{}".format(np.asarray(v).mean()))
				plt.plot(v)
				#plt.show()
				plt.savefig('figs/'+k)
		else:
			import os
			if not os.path.exists('offfigs/'+num+'/'):
				os.makedirs('offfigs/'+num+'/')
			for k,v in self.dict.items():
				print(k,np.asarray(v).mean())
				plt.figure()
				try:
					plt.title(k+"{}".format(np.asarray(v).mean()))
				except:
					plt.title(k)
				plt.plot(v)
				plt.savefig('offfigs/'+num+'/'+k+'.png')

	def say(self, all = False):
		if not all:
			for k, v in self.dict.items():
				print(k, v[-1])
		else:
			for k, v in self.dict.items():
				print('Average', k, np.asarray(v).mean())

	def plot_nb(self):
		import matplotlib
		matplotlib.use("Qt5Agg")
		import matplotlib.pyplot as plt
		for k,v in self.dict.items():
			print(k, np.asarray(v).mean())
			plt.figure()
			plt.title(k)
			plt.plot(v)
			plt.show()

def get_data(env_name ='Hopper-v2'):

	if env_name == 'Hopper-v2':
		buffer = pickle.load(open('hopper.pkl', 'rb'))
	elif env_name == 'HalfCheetah-v2':
		buffer = pickle.load(open('hc.pkl', 'rb'))
	return buffer





def get_expert(env_name = 'PBHopper', return_next_states = True):
	import pickle
	if env_name == 'Pendulum-v0':
		try:
			trajs = pickle.load(open('data/expert_pendulum.pkl', 'rb'))
		except:
			import pickle5 as pickle
			trajs =  pickle.load(open('data/expert_pendulum.pkl', 'rb'))
		states, actions = [], []
		for traj in trajs:
			traj = np.asarray(traj)
			states.append(np.stack(traj[:,0]))
			actions.append(np.stack(traj[:,1]).reshape(-1,1))

	elif env_name == 'PBHopper':
		trajs = pickle.load(open('data/exp_hop.pkl', 'rb'))
		states, actions = [],[]
		for traj in trajs:
			traj = np.asarray(traj)
			states.append(np.stack(traj[:,0]).astype('float'))
			actions.append(np.stack(traj[:,1]).astype('float'))
	elif env_name == 'HalfCheetah-v2':
		trajs = pickle.load(open('data/hcexpert.pkl', 'rb'))
		states, actions = [],[]
		for traj in trajs:
			traj = np.asarray(traj)
			states.append(np.stack(traj[:,0]).astype('float'))
			actions.append(np.stack(traj[:,1]).astype('float'))

	elif env_name == 'Hopper-v2':
		trajs = pickle.load(open('data/hopper_expert.pkl', 'rb'))
		states, actions = [],[]
		for traj in trajs:
			states.append(np.stack(traj['observations']))
			actions.append(np.stack(traj['actions']).astype('float'))

	if return_next_states:

		traj_states, traj_actions = states, actions
		states_, actions_, next_states_, next_actions_ = [],[],[], []
		for states, actions in zip(traj_states, traj_actions):
			next_states = np.copy(states[1:])
			next_actions = np.copy(actions[1:])
			states = states[:-1]
			actions = actions[:-1]
			rand = np.random.randint(len(states))
			assert len(states) == len(next_states) == len(actions)

			states_.append(states)
			actions_.append(actions)
			next_states_.append(next_states)
			next_actions_.append(next_actions)

		states, actions, next_states, next_actions = (np.vstack(states_), np.vstack(actions_),
								 np.vstack(next_states_), np.vstack(next_actions_))


		return states[:990], actions[:990], next_states[:990], next_actions[:990]
