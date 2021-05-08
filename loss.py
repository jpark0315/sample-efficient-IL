import torch 
import numpy as np  
from torchutils import softmax 

#Critic Loss Functions
def kl_loss_c(expert_nu_0, expert_nu, expert_nu_next,expert_real_nu_next, args):
	"""
	Double Conjugate representation of KL div 
	"""
	expert_diff = expert_nu - args.discount * expert_nu_next 
	linear_loss = torch.mean(expert_nu_0 * (1-args.discount))
	with torch.no_grad():
		weights = softmax(expert_diff)
		assert weights.shape == expert_diff.shape 
	non_linear_loss = torch.sum( weights * expert_diff)
	loss = non_linear_loss - linear_loss 

	return loss 

def lse_loss_c(expert_nu_0, expert_nu, expert_nu_next,expert_real_nu_next, args):
	"""
	Double Conjugate representation of KL div, but in logsumexp form 
	"""
	expert_diff = expert_nu - args.discount * expert_nu_next 
	linear_loss = torch.mean(expert_nu_0 * (1-args.discount)) 
	non_linear_loss = torch.Tensor([np.log(1/expert_states.shape[0])]) \
				+torch.logsumexp(expert_diff, 0)
	loss = non_linear_loss - linear_loss 

	return loss 

def linear_max_loss_c(expert_nu_0, expert_nu, expert_nu_next,expert_real_nu_next, args):
	expert_diff = expert_nu - args.discount * expert_nu_next 

	non_linear_loss = expert_real_nu_next - expert_nu_next
	linear_loss = expert_nu_0 - expert_nu  
	loss = (discount * non_linear_loss  - linear_loss ).mean()
	softmax = torch.Tensor([np.log(1/expert_states.shape[0])])+torch.logsumexp(expert_diff, 0)
	loss += max_weight * softmax[0]
	return loss 

def linear_loss_c(expert_nu_0, expert_nu, expert_nu_next, expert_real_nu_next, args):
	non_linear_loss = expert_real_nu_next - expert_nu_next
	linear_loss = expert_nu_0 - expert_nu  
	loss = (args.discount * non_linear_loss -  linear_loss ).mean()
	return loss 


def penmax_loss_c(expert_nu_0, expert_nu, expert_nu_next, expert_real_nu_next, args):
	"""
	Penalizes soft-max values of the difference 
	"""
	non_linear_loss = expert_real_nu_next - expert_nu_next
	linear_loss = expert_nu_0 - expert_nu  
	loss = (args.discount * non_linear_loss  - linear_loss ).mean()
	softmax = torch.Tensor([np.log(1/len(expert_nu))])\
			+torch.logsumexp(linear_loss, 0)
	loss += args.max_weight * softmax[0]

	return loss 



								

#Policy Loss Functions

def kl_loss_p(expert_nu_0, expert_nu, expert_nu_next,expert_real_nu_next, args):
	"""
	Double Conjugate representation of KL div 
	"""
	expert_diff = expert_nu - args.discount * expert_nu_next 
	linear_loss = torch.mean(expert_nu_0 * (1-args.discount))
	with torch.no_grad():
		weights = softmax(expert_diff)
		assert weights.shape == expert_diff.shape 
	non_linear_loss = torch.sum( weights * expert_diff)
	loss = non_linear_loss - linear_loss 

	return -loss 

def lse_loss_p(expert_nu_0, expert_nu, expert_nu_next,expert_real_nu_next, args):

	expert_diff = expert_nu - args.discount * expert_nu_next 
	linear_loss = torch.mean(expert_nu_0 * (1-args.discount)) 
	non_linear_loss = torch.Tensor([np.log(1/expert_states.shape[0])]) \
				+torch.logsumexp(expert_diff, 0)
	loss = non_linear_loss - linear_loss 

	return -loss 

def linear_max_loss_p(expert_nu_0, expert_nu, expert_nu_next,expert_real_nu_next, args):
	expert_diff = expert_nu - args.discount * expert_nu_next 

	non_linear_loss = expert_real_nu_next - expert_nu_next
	linear_loss = expert_nu_0 - expert_nu  
	loss = (discount * non_linear_loss  - linear_loss ).mean()
	softmax = torch.Tensor([np.log(1/expert_states.shape[0])])+torch.logsumexp(expert_diff, 0)
	loss += max_weight * softmax[0]
	return -loss 

def linear_loss_p(expert_nu_0, expert_nu, expert_nu_next, expert_real_nu_next, args):
	non_linear_loss = expert_real_nu_next - expert_nu_next
	linear_loss = expert_nu_0 - expert_nu  
	loss = (args.discount * non_linear_loss -  linear_loss ).mean()
	return -loss 


def penmax_loss_p(expert_nu_0, expert_nu, expert_nu_next, expert_real_nu_next, args):
	non_linear_loss = expert_real_nu_next - expert_nu_next
	linear_loss = expert_nu_0 - expert_nu  
	loss = (discount * non_linear_loss  - linear_loss ).mean()
	softmax = torch.Tensor([np.log(1/expert_states.shape[0])])+torch.logsumexp(linear_loss, 0)
	loss -= max_weight * softmax[0]

	return -loss 




critic_loss = {
	'kl':kl_loss_c,
	

}

policy_loss = {
	'kl':kl_loss_p, 
	
}





