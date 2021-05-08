

"""
Experiments:
	loss functions (look at gans )
	incorporate different data for optimization (eg. conq with importance sampling/random states, etc )
	hyperparams
	architectures (tan for logstd, etc)
	regularization schemes (model+ policy + critic)
	training scripts/procedure 
	training algorithm(not too different )
	sampling schemes
	standard tricks in papers/online 
	
Reference:
vdice 
conq 
MOPO

"""



network_size = 7
elite_size = 5
state_size = 3
action_size = 1
model_lr = 1e-3
model_batch_size = 256

class Args:
	num_train_steps = 30000
	discount = 0.99 

	policy_hidden_dim = 256
	policy_activation = 'ReLU'
	policy_num_hidden =  3
	policy_lr = 1e-5
	bc_loss_fn = 'Hybrid'
	BC_batch_size = 256
	state_dim = 11
	act_dim  = 3


	lipschitz = 0.05
	num_critic_hidden_layers = 1
	critic_hidden_size = 256
	critic_lr = 1e-3
	critic_activ = 'ReLU'
	rms = False

	reg_coef = 1e-4
	max_weight = None
	critic_loss_fn = 'kl'
	policy_loss_fn ='kl'
	policy_orthogonal_reg = True
	bc_clamp = False 

	discrim_bias = False 
	lipshitz_clamp = True 

