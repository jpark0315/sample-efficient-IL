import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
import numpy as np
from utils import *


################################## set device ##################################

print("============================================================================================")


# set device to cpu or cuda
device = torch.device('cpu')

# if(torch.cuda.is_available()):
#     device = torch.device('cuda:0')
#     torch.cuda.empty_cache()
#     print("Device set to : " + str(torch.cuda.get_device_name(device)))
# else:
#     print("Device set to : cpu")

print("============================================================================================")




################################## PPO Policy ##################################
def geometric_index(size, bs):
    """
    Return indices such that the previous states are weighted
    heavier than the next
    """
    while True:
        idxs = np.random.geometric(1- 0.99, bs)
        if (idxs<=size-1).sum() == bs:
            return idxs

class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []


    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

class ParallelRolloutBuffer:
    def __init__(self, act_dim, state_dim, parallel, horizon):
        self.actions = np.zeros((parallel, horizon,  act_dim))
        self.states =  np.zeros((parallel,horizon, state_dim))
        self.logprobs =  np.zeros((parallel, horizon))
        self.rewards =  np.zeros((parallel, horizon))
        self.is_terminals = np.zeros((parallel, horizon))
        self.parallel, self.horizon, self.act_dim, self.state_dim= parallel, horizon , act_dim, state_dim
    def clear(self):
        self.actions = np.zeros((self.parallel, self.horizon,  self.act_dim))
        self.states =  np.zeros((self.parallel,self.horizon, self.state_dim))
        self.logprobs =  np.zeros((self.parallel, self.horizon))
        self.rewards =  np.zeros((self.parallel, self.horizon))
        self.is_terminals = np.zeros((self.parallel, self.horizon))

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init):
        super(ActorCritic, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)

        # actor
        if has_continuous_action_space :
            self.actor = nn.Sequential(
                            nn.Linear(state_dim, 512),
                            nn.ReLU(),
                            nn.Linear(512, 512),
                            nn.ReLU(),
                            nn.Linear(512, action_dim),
                            nn.Tanh()
                        )
        else:
            self.actor = nn.Sequential(
                            nn.Linear(state_dim, 64),
                            nn.Tanh(),
                            nn.Linear(64, 64),
                            nn.Tanh(),
                            nn.Linear(64, action_dim),
                            nn.Softmax(dim=-1)
                        )


        # critic
        self.critic = nn.Sequential(
                        nn.Linear(state_dim, 64),
                        nn.Tanh(),
                        nn.Linear(64, 64),
                        nn.Tanh(),
                        nn.Linear(64, 1)
                    )

    def set_action_std(self, new_action_std):

        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")


    def get_dist_and_mode(self, states):
        assert len(states.shape) == 2
        action_mean = self.actor(states)
        cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
        dist = MultivariateNormal(action_mean, cov_mat)
        mode = action_mean

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

    def act(self, state):

        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)

        return action.detach(), action_logprob.detach()


    def evaluate(self, state, action):

        if self.has_continuous_action_space:
            action_mean = self.actor(state)

            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(device)
            dist = MultivariateNormal(action_mean, cov_mat)

            # For Single Action Environments.
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)

        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy


class PPO:
    def __init__(self, logger,
        state_dim = 11,
        action_dim = 3,
        lr_actor = 3e-4,
        lr_critic = 1e-3,
        gamma = 0.99,
        K_epochs = 5,
        eps_clip = 0.2,
        has_continuous_action_space  = True,
        action_std_init=0.6,
        parallel = 1000,
        horizon = 10,
        single = False,
        bc_batch_size = 256,
        geometric = False,
        bc_loss = "logprob",
        bc_ppo_train_step = 1
        ):

        self.logger = logger
        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_std = action_std_init

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        if parallel == 1:
            self.buffer = RolloutBuffer()
        else:
            self.buffer = ParallelRolloutBuffer(action_dim, state_dim, parallel, horizon)

        self.policy = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                    ])

        self.policy_old = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

        self.parallel = True if parallel > 1 else False
        self.horizon = horizon
        self.state_dim, self.action_dim = state_dim, action_dim
        self.bc_batch_size = bc_batch_size
        self.geometric = geometric
        self.bc_loss = bc_loss
        self.bc_ppo_train_step = bc_ppo_train_step
    def set_action_std(self, new_action_std):

        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)


    def decay_action_std(self, action_std_decay_rate, min_action_std):

        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if (self.action_std <= min_action_std):
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)




    def select_action(self, state, horizon = None):

        if self.has_continuous_action_space:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                actions, action_logprob = self.policy_old.act(state)

            if len(state.shape) == 2:
                # state, action, action_logprob = list(state), list(actions), list(action_logprob)
                # self.buffer.states.extend(state)
                # self.buffer.actions.extend(action)
                # self.buffer.logprobs.extend(action_logprob)
                self.buffer.states[:,horizon] = state.numpy()
                self.buffer.actions[:,horizon] = actions.numpy()
                self.buffer.logprobs[:,horizon] = action_logprob.numpy()
            else:
                self.buffer.states.append(state)
                self.buffer.actions.append(actions.flatten())
                self.buffer.logprobs.append(action_logprob)

            return actions.numpy()



    def update(self, e_states = None, e_actions = None, bc_step = False, clear_buffer = True):

        # Monte Carlo estimate of returns
        if not self.parallel:
            rewards = []
            discounted_reward = 0
            for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
                if is_terminal:
                    discounted_reward = 0
                discounted_reward = reward + (self.gamma * discounted_reward)
                rewards.insert(0, discounted_reward)

            old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
            old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
            old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)

        else:
            rewards = []
            discounted_reward = 0
            buffer_rewards, buffer_is_terminals = self.buffer.rewards.reshape(-1).tolist(),self.buffer.is_terminals.reshape(-1).tolist()
            #print(buffer_rewards, buffer_is_terminals)
            for reward, is_terminal in zip(reversed(buffer_rewards), reversed(buffer_is_terminals)):
                if is_terminal:
                    discounted_reward = 0
                discounted_reward = reward + (self.gamma * discounted_reward)
                rewards.insert(0, discounted_reward)
            #print(rewards)

            old_states = torch.FloatTensor(self.buffer.states.reshape(-1, self.state_dim))
            old_actions  = torch.FloatTensor(self.buffer.actions.reshape(-1, self.action_dim))
            old_logprobs = torch.FloatTensor(self.buffer.logprobs.reshape(-1))


        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor


        # Optimize policy for K epochs
        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            mseloss = 0.5*self.MseLoss(state_values, rewards)
            loss = -torch.min(surr1, surr2) + mseloss - 0.01*dist_entropy
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()


            self.logger.log('ppo grad', self.grad_norm(self.policy))
            self.logger.log('advantages', advantages.mean().detach().item())
            self.logger.log('advantages max', advantages.max().detach().item())
            self.logger.log('mseloss', mseloss.mean().detach().item())
            self.logger.log('training logprobs', logprobs.mean().detach().item())
            self.logger.log('state_values', state_values.mean().detach().item())
            self.logger.log('policy ratio', ratios.mean().detach().item())

            if e_states is not None:
                self.train_bc(e_states, e_actions, train_step = self.bc_ppo_train_step)
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        if clear_buffer:
            self.buffer.clear()


    def bc_step(self, state_batch,action_batch):
        state_batch, action_batch = torch.FloatTensor(state_batch), torch.FloatTensor(action_batch)
        pred = self.policy(state_batch)
        loss = nn.MSELoss()(pred, action_batch)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


        return loss.item()

    def grad_norm(self, model):
        with torch.no_grad():
            total_norm = 0
            for p in model.parameters():
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
        return total_norm

    def train_bc(self,state,action, train_step = 1500,
     eva = False, geometric = None, progress = False ):

        batch_size = self.bc_batch_size
        geometric = self.geometric if geometric is None else geometric


        losses, scores = [], []
        state, action = torch.FloatTensor(state), torch.FloatTensor(action)
        from tqdm import trange
        bar = trange if progress else range
        for i in bar(train_step):
            if geometric:
                idxs = geometric_index(state.shape[0],batch_size)
            else:
                idxs = np.random.permutation(state.shape[0])[:batch_size]
            state_,action_ = state[idxs], action[idxs]

            if self.bc_loss == "MSE":
                mode, samples, log_probs  = self.policy(state_)
                loss = nn.MSELoss()(samples, action_)
            elif self.bc_loss == "logprob":
                loss = -self.policy.get_log_prob(state_, action_).mean()

            if i % 100 == 0 and eva:
                score, time = evaluate(self, None, env)
                print(score)
                scores.append(score)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses.append(loss.item())
            self.logger.log('BC loss', loss.item())
            self.logger.log('bc grad', self.grad_norm(self.policy.actor))

        self.policy_old.load_state_dict(self.policy.state_dict())
        return losses, scores

    def train_bc_ppo(self,state,action, weight = 3, train_step = 3,
     eva = False, batch_size = 256, geometric = False):
        losses, scores = [], []
        state, action = torch.FloatTensor(state), torch.FloatTensor(action)
        from tqdm import trange
        for i in range(train_step):
            if geometric:
                idxs = geometric_index(state.shape[0],batch_size)
            else:
                idxs = np.random.permutation(state.shape[0])[:batch_size]
            state_,action_ = state[idxs], action[idxs]
            mode, samples, log_probs  = self.policy(state_)
            loss = weight * nn.MSELoss()(samples, action_)

            if i % 100 == 0 and eva:
                score, time = evaluate(self, None, env)
                print(score)
                scores.append(score)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            clear_buffer = True if i == train_step - 1 else False
            self.update(clear_buffer = clear_buffer)

            losses.append(loss.item())
            self.logger.log('BC loss', loss.item())
            self.logger.log('bc grad', self.grad_norm(self.policy.actor))

            self.policy_old.load_state_dict(self.policy.state_dict())
        return losses, scores

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)


    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))

def algo1(agent, discrim, env, e_states, e_actions, update_bc = True, s_a = False):

    for i in range(1):
        rollout_real_ppo(agent, discrim, env, agent.logger, batch_size = 20000, s_a = s_a)
        if s_a:
            discrim.train_discrim(e_states,e_actions, torch.stack(agent.buffer.states, dim=0),
                             torch.stack(agent.buffer.actions, dim=0))
        else:
            discrim.train_discrim(e_states, torch.stack(agent.buffer.states, dim=0))


        if update_bc:
            agent.update(e_states, e_actions)
        else:
            agent.update()
        print(i)
        agent.logger.say()
        print()

def rollout_real_ppo(agent, discrim, env,logger, batch_size = 50000, s_a = False):

    state = env.reset()
    rewards, fake_rewards = 0,0
    eps_rewards, eps_fake_reward = [], []
    for i in range(batch_size):
        action = agent.select_action(state)

        next_state, reward, done, _ = env.step(action.flatten())
        if s_a:
            fake_reward = discrim(state.reshape(1,-1),action.reshape(1,-1)).detach().item()
        else:
            fake_reward = discrim(state.reshape(1,-1)).detach().item()

        #agent.buffer.rewards.append(reward)
        agent.buffer.rewards.append(fake_reward)
        agent.buffer.is_terminals.append(done)
        rewards += reward
        fake_rewards += fake_reward
        if done:
            eps_rewards.append(rewards)
            eps_fake_reward.append(fake_rewards)
            rewards, fake_rewards = 0, 0
            state = env.reset()
            continue

        state = next_state

    logger.log('mean real reward', np.asarray(eps_rewards).mean())
    logger.log('mean fake reward', np.asarray(fake_rewards).mean())
    print('mean real reward', np.asarray(eps_rewards).mean())
    print('mean fake reward', np.asarray(fake_rewards).mean())




def algo2(agent, discrim,model, env, states, e_states, e_actions, logger,
    update_bc = True,
    s_a = False,
    start_state = 'bad'):

    for i in range(2000):
        rollout_single_ppo(agent, model, discrim, e_states,states, logger,env,
        s_a = s_a, start_state = start_state)

        if s_a:
            discrim.train_discrim(e_states,e_actions, torch.FloatTensor(agent.buffer.states.reshape(-1, e_states.shape[1])).numpy(),
                             torch.FloatTensor(agent.buffer.actions.reshape(-1, e_actions.shape[1])).numpy())
        else:
            discrim.train_discrim(e_states, torch.FloatTensor(agent.buffer.states.reshape(-1, e_states.shape[1])).numpy())


        if update_bc:
            agent.update(e_states, e_actions)
        else:
            agent.update()

        print(i)
        rew, _ = evaluate(agent.policy, env)
        logger.log('real reward', rew)
        agent.logger.say()
        print()

def rollout_single_ppo(agent, model, discrim, states, bad_states, logger,env,
                       s_a = True, start_state = 'bad'):
    total_rewards = []
    parallel, rollout_length = agent.buffer.states.shape[0], agent.buffer.states.shape[1]
    #state, rollout_rewards = states[np.random.randint(states.shape[0])], []
    #state, rollout_rewards = states[np.random.geometric(0.01)], []
    # if np.random.uniform() < 0.1:
    #     #state = bad_states[np.random.randint(0, len(bad_states), size =parallel)]
    #     state = bad_states[np.random.geometric(0.01, size = parallel)]
    # else:
    #     #state = states[np.random.randint(0, len(states), size =parallel)]
    #     try:
    #         state = states[np.random.geometric(0.01, size = parallel)]
    #     except:
    #         state = states[np.random.geometric(0.01, size = parallel)]
    if start_state == 'bad':
        state = bad_states[np.random.permutation(bad_states.shape[0])[:parallel]]
    elif start_state == 'good':
        state = states[geometric_index(states.shape[0],parallel )]
    elif start_state == 'random':
        state = np.asarray([env.reset() for _ in range(parallel)])

    for horizon in range(rollout_length):

        agent_action = agent.select_action(state, horizon)
        model_n_obs, info = model.predict_next_states(state, agent_action, deterministic = True)
        if s_a:
            reward = discrim(state,agent_action).detach()
        else:
            reward = discrim(state).detach()

        model_loss = model.validate(state, agent_action,
                                    model_n_obs, verbose = False)

        terminals = [False if horizon < rollout_length-1 else True for _ in range(len(reward))]
        #terminals = [True for _ in range(len(reward))]
        agent.buffer.rewards[:,horizon] = np.array(reward).reshape(-1)
        agent.buffer.is_terminals[:,horizon] = np.array(terminals).reshape(-1)


        logger.log('model logprobs', info['log_prob'].mean())
        logger.log('model loss', np.asarray(model_loss).mean())
        logger.log('model loss std', np.asarray(model_loss).std())
        logger.log('rollout {} rew'.format(horizon), (reward.mean().item(), state.mean(), state.std() ))
        logger.log('state mean',state.mean() )
        logger.log('state std',state.std())
        total_rewards.append(np.array(reward).reshape(-1))

        state = model_n_obs

    print('Discrim rewards', np.stack(total_rewards).mean(1), np.stack(total_rewards).sum(0).mean())
    logger.log('avg total rewards', np.stack(total_rewards).sum(0).mean())
