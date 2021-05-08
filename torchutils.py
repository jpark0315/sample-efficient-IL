import torch 
import torch.nn as nn
import numpy as np 
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
import torch.nn.functional as F



def orthogonal_regularization(model, reg_coef = 1e-4):

	with torch.enable_grad():
		reg = torch.zeros(1)
		for name, param in model.named_parameters():
			if 'bias' not in name:
				param_flat = param.view(param.shape[0], -1)
				sym = torch.mm(param_flat, torch.t(param_flat))
				reg += torch.sum(torch.square(sym * (1- torch.eye(param_flat.shape[0]))))
	return reg * reg_coef

def gradient_penalty(model, real_data, fake_data):

	batch_size = real_data.size()[0]

	EPS = np.finfo(np.float32).eps
	alpha = torch.rand(batch_size, 1)
	interpolated = alpha * real_data + (1-alpha) * fake_data 
	interpolated = Variable(interpolated, requires_grad = True)
	interpolated_out = model(interpolated)
	gradients = torch_grad(outputs = interpolated_out,inputs = interpolated,
		grad_outputs=torch.ones(interpolated_out.size()), create_graph=True, 
		retain_graph=True)[0] + EPS 
	#gradients = gradients.reshape(batch_size, -1)
	gradient_penalty = torch.mean(torch.square(torch.norm(gradients, dim= -1,
			keepdim = True)-1))
	return gradient_penalty

def get_entropy(prob):
	return -prob.T @ torch.log(prob)

def softmax(x, axis = 0):
	x = x - x.max(axis)[0]
	return torch.exp(x)/torch.sum(torch.exp(x), axis = axis, keepdims = True)
