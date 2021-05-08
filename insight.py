import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import itertools 
from tqdm import trange 
import os 
from utils import get_expert 
import gym 
from actor import Solo_BC
from ppo import PPO
import pickle 
from utils import * 
from actor import Actor 
try:
    import pybulletgym

    env = gym.make('HopperMuJoCoEnv-v0')
except:
    env = gym.make('Hopper-v2')


class Normalizer:

    def __init__(self, shift, scale):
        self.shift = shift
        self.scale = scale 

    def normalize(self, observation):
        return (observation+ self.shift) * self.scale 

def evaluate(actor, normalizer, env, num_episodes=10, stats = None):

    total_timesteps = 0
    total_returns = 0

    for _ in range(num_episodes):
        #state = normalizer.normalize(env.reset())
        state = env.reset()
        done = False
        while not done:
            with torch.no_grad():
                # if stats == 'mode':
                #     action, _, _ = actor(np.array([state]))
                # else:
                #     _, action, _ = actor(np.array([state]))
                action, _, _ = actor(np.array([state]))
                #action = actor(np.array([state]))
            action = action[0].numpy()
            next_state, reward, done, _ = env.step(action)

            total_returns += reward
            total_timesteps += 1
            #state = normalizer.normalize(next_state)
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
        import matplotlib.pyplot as plt 
        if num is None:
            for k,v in self.dict.items():
                plt.figure()
                plt.title(k)
                plt.plot(v)
                plt.show()
                #plt.savefig('figs/'+k)
        else:
            import os 
            if not os.path.exists('offfigs/'+num+'/'):
                os.makedirs('offfigs/'+num+'/')        
            for k,v in self.dict.items():
                plt.figure()
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

class Scaler:
    """
    Helper class for automatically normalizing inputs into the network
    """
    def __init__(self, x_dim = None):

        self.mu = None 
        self.sigma = None 

    def fit(self, data, transform = False):
        """
        Args:
        data(np.ndarray): num_inputs x dimension
            usually state/action concatenated 
        """
        self.mu, self.sigma = data.mean(0, keepdims = True), data.std(0, keepdims = True)
        self.sigma[self.sigma < 1e-12] = 1
        self.fitted = True 
        assert self.mu.shape == self.sigma.shape and self.mu.shape[0] == 1 
        if transform:
            return self.transform(data)

    def transform(self, data):
        """
        Transforms input matrix data using mu,sigma 
        """
        assert self.fitted 
        assert len(data.shape) == len(self.mu.shape) 
        return (data - self.mu) / self.sigma 

    def inverse_transform(self, data):
        """
        Normalized inputs back to original 
        """
        assert len(data.shape) == len(self.mu.shape)
        return self.sigma * data + self.mu 


class FC(nn.Module):
    def __init__(self, input_dim, output_dim, activation = None, ensemble_size = 7):
        super().__init__()

        self.ensemble_size = ensemble_size

        self.register_parameter('weight', torch.nn.Parameter(torch.zeros(ensemble_size, input_dim, output_dim)))
        self.register_parameter('bias', torch.nn.Parameter(torch.zeros(ensemble_size, 1, output_dim)))
        #torch.nn.init.trunc_normal_(self.weight, std = 1/(2*np.sqrt(input_dim)))
        self.weight.data.normal_( 0.0, 1/(2*np.sqrt(input_dim)) )
        self.select = list(range(0, self.ensemble_size))

        
        self.activation = Swish()
        if activation is "None":
            self.activation = None  

    def forward(self, x):
        weight = self.weight[self.select]
        bias = self.bias[self.select]

        if len(x.shape)== 2:
            x = torch.einsum('ij,bjk->bik', x, weight)
        else:
            assert x.shape[0] == self.ensemble_size
            x = torch.einsum('bij,bjk->bik', x, weight)

        x = x + bias 
        return self.activation(x) if self.activation is not None else x 

    def set_selelct(self, indexes):
        assert len(indexes) <= self.ensemble_size and max(indexes) < self.ensemble_size
        self.select = indexes

class Model(nn.Module):
    def __init__(self, obs_dim, act_dim, ensemble_size =7, 
                     hidden_layers = 4, hidden_size = 256, learning_rate = 1e-3, rew_dim = 1,weight_decay=0.000075,
                     with_reward = False):
        super().__init__()
        self.output_dim = obs_dim + rew_dim if with_reward else obs_dim
        module_list = []
        for i in range(hidden_layers):
            if i==0:
                module_list.append(FC(obs_dim+act_dim, hidden_size, ensemble_size=  ensemble_size))
            else:
                module_list.append(FC(hidden_size, hidden_size, ensemble_size=  ensemble_size))
        module_list.append(FC(hidden_size, 2*(self.output_dim), activation = "None" , ensemble_size = ensemble_size))
        self.module_list = torch.nn.ModuleList(module_list)

        self.max_logvar = Variable(torch.ones((1, self.output_dim)).type(torch.FloatTensor) / 2, requires_grad=True)
        self.min_logvar = Variable(-torch.ones((1, self.output_dim)).type(torch.FloatTensor) * 10, requires_grad=True)

        self.optim = torch.optim.Adam(self.parameters(), lr = learning_rate, weight_decay= weight_decay)
        self.optim_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optim, gamma = 0.99)
    def forward(self, x, return_logvar = False):
        for layer in self.module_list:
            x = layer(x)

        mean = x[:,:,:self.output_dim]
        logvar = self.max_logvar - F.softplus(self.max_logvar - x[:,:,self.output_dim:])
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)
        if return_logvar:
            return mean, logvar 
        else:
            return mean, torch.exp(logvar)

    def loss(self, mean, logvar, labels, return_mean = False, inc_var_loss = True ):
        """ 
        Input logvar must be in log space
        Returns loss vector of shape (ensemble_size)
        """

        inv_var = torch.exp(-logvar)
        if inc_var_loss:
            mse_loss = torch.mean(torch.mean(torch.square(mean - labels)* inv_var, -1),-1)
            var_loss = torch.mean(torch.mean(logvar, -1),-1)
            total_loss = mse_loss + var_loss 
        else:
            raise NotImplementedError

        if return_mean:
            return total_loss.mean()

        return total_loss

    def grad_step(self, loss):

        loss += 0.01 * torch.sum(self.max_logvar) - 0.01 * torch.sum(self.min_logvar)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()



class Ensemble_Model:
    def __init__(self, 
        network_size = 7, 
        elite_size = 5,
        state_size = 0, 
        action_size = 0, 
        reward_size=0,
        hidden_size=256, 
        learning_rate = 1e-3,
        weight_decay = 0.000075,
        hidden_layers = 4, 
        batch_size = 128,
        decay_lr = True,
        logger = None):
        self.network_size = network_size
        self.elite_size = elite_size
        self.model_list = []
        self.state_size = state_size
        self.action_size = action_size
        self.reward_size = reward_size
        self.elite_model_idxes = []
        self.scaler = Scaler()
        self.models = Model(state_size, action_size,
                        hidden_layers = hidden_layers, 
                        hidden_size = hidden_size, 
                        learning_rate = learning_rate,
                        weight_decay=weight_decay)
        self.logger = logger 
        self.batch_size = batch_size
        self.decay_lr = decay_lr
        self._model_inds = None 

    def adversarial_step(self, inputs, labels, ad_inputs, ad_labels):
        """ 
        Increase Likelihood of expert samples, decrease likelihood of policy's samples 
        """
        inputs, labels = torch.FloatTensor(inputs), torch.FloatTensor(labels)
        ad_inputs, ad_labels = torch.FloatTensor(ad_inputs), torch.FloatTensor(ad_labels)

        inputs = self.scaler.fit(inputs, transform = True)
        ad_inputs = self.scaler.transform(ad_inputs)

        mean, logvar = self.models(inputs, return_logvar = True)
        ad_mean, ad_logvar = self.models(ad_inputs, return_logvar = True)
        l1, l2 = self.models.loss(mean, logvar, labels,return_mean = False), self.models.loss(ad_mean, ad_logvar, ad_labels,return_mean = False)

        loss = l1.mean() - 0.01 * l2.mean() 
        self.models.grad_step(loss)
        print('l1', l1.mean(), 'l2', l2.mean())
        return loss.item()

    def train(self, inputs, labels, holdout_ratio = 0.1, max_epochs = 10000,
            max_epochs_after_update = 5):
        """
        Trains on entire buffer, until holdout set is improved 
        """
        batch_size = self.batch_size 
        def shuffle_rows(arr):
            idxs = np.argsort(np.random.uniform(size=arr.shape), axis=-1)
            return arr[np.arange(arr.shape[0])[:, None], idxs]

        self._max_epochs_since_update = max_epochs_after_update 
        self._start_train()

        num_holdout = int(inputs.shape[0] * holdout_ratio)
        permutation = np.random.permutation(inputs.shape[0])
        inputs, holdout_inputs = inputs[permutation[num_holdout:]], inputs[permutation[:num_holdout]]
        labels, holdout_labels = labels[permutation[num_holdout:]], labels[permutation[:num_holdout]]
        inputs, holdout_inputs = torch.FloatTensor(inputs), torch.FloatTensor(holdout_inputs)
        labels, holdout_labels = torch.FloatTensor(labels), torch.FloatTensor(holdout_labels)

        #Normalize using only train data 
        inputs = self.scaler.fit(inputs, transform = True)
        holdout_inputs = self.scaler.transform(holdout_inputs)
        print('input/holdouts', inputs.mean(), holdout_inputs.mean(), inputs.std(), holdout_inputs.std())
        print('num_holdout, inputshape, holdoutshape', num_holdout, inputs.shape[0], holdout_inputs.shape[0])
        break_train, grad_updates = False, 0

        for epoch in itertools.count():
            idxs = np.random.randint(inputs.shape[0], size=[self.network_size, inputs.shape[0]])
            loss_dict = {i: [] for i in range(self.network_size)}
            inputs, labels = inputs.numpy(), labels.numpy() 

            for batch_num in trange(int(np.ceil(idxs.shape[-1] / batch_size))):
                #different minibatch for each model!!! 
                batch_idxs = idxs[:, batch_num * batch_size:(batch_num + 1) * batch_size]
                input = torch.FloatTensor(inputs[batch_idxs])
                label = torch.FloatTensor(labels[batch_idxs])

                mean, logvar = self.models(input, return_logvar = True)
                loss = self.models.loss(mean, logvar, label, return_mean = False)
                self.models.grad_step(loss.mean())
                grad_updates += 1 
                assert len(input.shape) == len(label.shape) and len(input.shape) == 3
                assert input.shape[0] == self.network_size 
                if batch_num < int(np.ceil(idxs.shape[-1] / batch_size)) - 1:
                    assert input.shape[1] == batch_size 
                
                #record live training loss 
                for i,l in enumerate(loss):
                    loss_dict[i].append(l.detach().item())
                    #self.logger.log('Model Training Loss', l.detach().item())

            #validation
            with torch.no_grad():
                #Holdout samples
                mean, logvar = self.models(holdout_inputs,return_logvar = True)
                holdout_loss = self.models.loss(mean, logvar, holdout_labels)
                holdout_losses = [l.item() for l in holdout_loss]

                #Train samples
                inputs, labels = torch.FloatTensor(inputs), torch.FloatTensor(labels)
                # mean, logvar = self.models(inputs,return_logvar = True)
                # train_loss = self.models.loss(mean, logvar, labels)
                # train_losses = [l.item() for l in train_loss]

            loss_dict = [np.array(loss_dict[i]).mean() for i in range(self.network_size)]
            print('epoch{} Train sample Loss, Val sample Loss, Live Train Loss: '.format(epoch),
                         holdout_losses, loss_dict)

           # self.logger.log('Model Holdout Loss', np.array(holdout_losses).mean())
            break_train = self.break_train(epoch, holdout_losses)
            if break_train:
                print('Breaking Training, total num grad updates:', grad_updates)
                break 
                #self.logger.log('Model Grad updates', grad_updates)
               # self.logger.log('Model Epoch', epoch)
            if epoch == max_epochs:
                break 

            self.decay()
        self._end_train(holdout_losses)

       # sorted_loss_idx = np.argsort(losses)
       # self.elite_model_idxes = sorted_loss_idx[:self.elite_size].tolist()

    def decay(self):
        if self.decay_lr:
            self.models.optim_scheduler.step()

    def _start_train(self):
        self._state = {}
        self._snapshots = {i: (None, 1e10) for i in range(self.network_size)}
        self._epochs_since_update = 0

    def _end_train(self, holdout_losses):
        self.num_elites = 5
        sorted_inds = np.argsort(holdout_losses)
        self._model_inds = sorted_inds[:self.num_elites].tolist()
        print('Using {} / {} models: {}'.format(self.num_elites, self.network_size, self._model_inds))

    def random_inds(self, batch_size):
        inds = np.random.choice(self._model_inds, size=batch_size)
        return inds

    def break_train(self, epoch, holdout_losses):
        updated = False 
        for i in range(len(holdout_losses)):
            current = holdout_losses[i]
            _, best = self._snapshots[i]
            improvement = (best - current) / abs(best)
            if improvement > 0.01:
                self._snapshots[i] = (epoch, current)
                updated = True 

        if updated:
            self._epochs_since_update = 0
        else:
            self._epochs_since_update += 1
        print('Updated, epochs_since_update', updated, self._epochs_since_update,
            self._snapshots)
        if self._epochs_since_update >= self._max_epochs_since_update:
            return True 
        else:
            return False 


    def predict(self, inputs, gradient = False):
        """
        Returns ensemble_mean and ensemble variance(not log)
        """
        if not isinstance(inputs, torch.Tensor):
             inputs = torch.FloatTensor(inputs)
        inputs = self.scaler.transform(inputs)

       # self.logger.log('predicting scaler mean_dif', (inputs.mean(0) - self.scaler.mu).mean().item() )
        #self.logger.log('predicting scaler std_dif', (inputs.std(0) - self.scaler.sigma).mean().item() )
        ensemble_mean, ensemble_var = self.models(inputs ,return_logvar = False)
        if gradient:
            return ensemble_mean, ensemble_var 

        return ensemble_mean.detach().numpy(), ensemble_var.detach().numpy()

    def predict_next_states_gradient(self, obs, act):
        assert len(obs.shape) > 1

        obs,act = torch.FloatTensor(obs), torch.FloatTensor(act)
        inputs = torch.cat((obs, act), axis=-1)
        ensemble_model_means, ensemble_model_vars = self.predict(inputs, gradient = True)

        ensemble_model_means[:,:,:] += obs
        ensemble_model_stds = torch.sqrt(ensemble_model_vars)

        ensemble_samples = ensemble_model_means + torch.randn(ensemble_model_means.shape)*ensemble_model_stds
        num_models, batch_size, _ = ensemble_model_means.shape
        model_idxes = self.random_inds(batch_size)
        batch_idxes = torch.arange(0, batch_size)

        next_obs = ensemble_samples[model_idxes, batch_idxes]
        return next_obs 

    def predict_next_states(self, obs, act, deterministic=False):
        if len(obs.shape) == 1:
            obs = obs[None]
            act = act[None]
            return_single = True
        else:
            return_single = False

        inputs = np.concatenate((obs, act), axis=-1)
        ensemble_model_means, ensemble_model_vars = self.predict(inputs)

        ensemble_model_means[:,:,:] += obs
        ensemble_model_stds = np.sqrt(ensemble_model_vars)

        if deterministic:
            ensemble_samples = ensemble_model_means
        else:
            ensemble_samples = ensemble_model_means + np.random.normal(size=ensemble_model_means.shape) * ensemble_model_stds

        num_models, batch_size, _ = ensemble_model_means.shape
        model_idxes = self.random_inds(batch_size)
        batch_idxes = np.arange(0, batch_size)

        samples = ensemble_samples[model_idxes, batch_idxes]
        model_means = ensemble_model_means[model_idxes, batch_idxes]
        model_stds = ensemble_model_stds[model_idxes, batch_idxes]

        log_prob, dev = self._get_logprob(samples, ensemble_model_means, ensemble_model_vars)

        next_obs = samples

        batch_size = model_means.shape[0]
       
        if return_single:
            next_obs = next_obs[0]

        info = {'log_prob': log_prob, 'dev': dev,
        'vars': ensemble_model_vars
        }
        return next_obs, info

    def validate(self,state, action, next_state,verbose = True, inputs = None, labels = None):
        """
        Returns loss of test samples
        """
        if inputs is None:
            delta_state = next_state - state 
            inputs = torch.FloatTensor(np.concatenate((state, action), axis = -1))
            labels = torch.FloatTensor(delta_state)
        losses = []
        inputs = self.scaler.transform(inputs)

        with torch.no_grad():

            mean, logvar = self.models(inputs,return_logvar = True)
            train_loss = self.models.loss(mean, logvar, labels)
            losses = [l.item() for l in train_loss]
        if verbose:
            print('Test Loss', losses)
        else:
            return losses 


    def _get_logprob(self, x, means, variances):

        k = x.shape[-1]

        ## [ num_networks, batch_size ]
        log_prob = -1/2 * (k * np.log(2*np.pi) + np.log(variances).sum(-1) + (np.power(x-means, 2)/variances).sum(-1))

        ## [ batch_size ]
        prob = np.exp(log_prob).sum(0)

        ## [ batch_size ]
        log_prob = np.log(prob)

        stds = np.std(means,0).mean(-1)

        return log_prob, stds
    def save(self):
        if not os.path.exists('data/models/'):
            os.makedirs('data/models/')        
        path = "data/models/modelnew_"
        torch.save(self.models.state_dict(), path) 

    def load(self, states, actions,next_states, model_inds, paths = None, test = True):
        """
        Loads Model params and fits scaler mu/sigma since 
        calling this means scaler wasn't fit before 
        """
        print('Loading Model...')
        self._model_inds = model_inds 
        inputs = torch.FloatTensor(np.concatenate((states, actions), axis=-1))
        self.scaler.fit(inputs)

        if paths is None:
            path = "data/models/modelnew_"
        self.models.load_state_dict(torch.load(path))
 
        if test:
            print('Testing Params...')
            self.validate(states, actions, next_states)

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        x = x * F.sigmoid(x)
        return x



def load_model(states, actions, next_states):
    model = Ensemble_Model(state_size = 11,action_size = 3)

    memory = pickle.load(open('data/hopperpb.pkl','rb'))
    s,a,r,n,d = memory.sample(len(memory))
    states, actions, next_states = (np.concatenate([states, s], 0), np.concatenate([actions, a], 0),
         np.concatenate([next_states, n], 0))

    model.load(states, actions,next_states, [4, 3, 5, 2, 0], test = False )
    rand_idx = np.random.permutation(states.shape[0])[:1000]
    model.validate(states[rand_idx], actions[rand_idx], next_states[rand_idx])

    return model, states, actions, next_states 

#states, actions, next_states, next_actions = get_expert(env_name = 'Pendulum-v0', return_next_states = True)
e_states, e_actions, e_next_states, e_next_actions = get_expert(env_name = 'PBHopper', return_next_states = True)
#states, actions, next_states, next_actions = get_expert(env_name = 'PBHopper', return_next_states = True)

# states=  torch.normal(0.3, 0.3, size = (1000,3)).numpy()
#actions = torch.FloatTensor(1000,1).uniform_(-1,1).numpy() 
# actions = np.random.normal(states, states>0).max(1).reshape(-1,1)
#next_states = states[1:]
#states,actions = states[:-1], actions[:-1]
#discrim = SmallD(s = 11, a = 3)
#bc = SmallBC(s=11, a=3)
model,states,actions, next_states = load_model(e_states, e_actions, e_next_states)

shift = -np.mean(states, 0)
scale = 1.0 / (np.std(states, 0) + 1e-3)
# states = (states + shift) * scale
# next_states = (next_states + shift) * scale
# e_states = (e_states + shift) * scale
normalizer = Normalizer(shift, scale)


logger = Logger()
#agent = Actor(None,logger )

















