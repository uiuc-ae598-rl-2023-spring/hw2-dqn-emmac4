# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 13:42:07 2023

@author: clemm
"""
import torch 
import torch.nn as nn 
import numpy as np 
import random
from collections import namedtuple, deque
import matplotlib.pyplot as plt 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                ('state', 'action', 'next_state', 'reward', 'dones'))
class Policy(nn.Module): 
    def __init__(self, input_dim, output_dim):
        super(Policy, self).__init__()
        self.layers = nn.Sequential(nn.Linear(input_dim, 64),
                                    nn.Tanh(),
                                    nn.Linear(64,64),
                                    nn.Tanh(),
                                    nn.Linear(64, output_dim)) 

        return 
    
    def forward(self, x):
        out = self.layers(x)
        return out
    
class Replay(): 
    def __init__(self, size): 
        self.size = size 
        self.memory = deque([], maxlen=size)
        
        return 
    
    def store(self, episode): 
        self.memory.append(episode)
        return 
    
    def sample(self, batch_size): 
        sample = random.sample(self.memory, batch_size)
        return sample
    
    
class DQN(): 
    def __init__(self, env, gamma, lr, eps, batch, iters, use_memory, update_every): 
        #discount factor
        self.gamma = gamma
        #learn rate
        self.learn_rate = lr
        #for epsilon greedy action selection and epsilon decay 
        self.eps_start = eps 
        self.eps = self.eps_start
        self.min_eps = 0.0
        
        #training hyperparameters
        self.iters = iters
        self.batch_size = batch 
        self.use_memory = use_memory 
        self.update = update_every 
        
        #environment information 
        self.env = env
        self.act_space = env.num_actions 
        self.obs_space = env.num_states
        
        #initialize policy and target networks 
        self.policy = Policy(self.obs_space, self.act_space)
        self.target = Policy(self.obs_space, self.act_space)
        #copy parameters of policy to the target network 
        self.target.load_state_dict(self.policy.state_dict())
        self.target.eval()
        
        #initialize replay buffer 
        if self.use_memory == True: 
            self.memory = Replay(10000)
        else: 
            #if buffer not in use, set max size = to batch size
            self.memory = Replay(self.batch_size)
        
        #optimizer and loss function 
        self.opt = torch.optim.Adam(self.policy.parameters(), lr = self.learn_rate)
        self.loss_fun = nn.MSELoss()
        
        return 
    
    def get_action(self, state): 
        #get action from policy network 
        with torch.no_grad(): 
            action = self.policy(state).argmax(1)
        return action.item()
    
    def greedy_action(self, s): 
        #epsilon greedy action selection 
        with torch.no_grad():
            p = np.random.random()
            
            if p < self.eps: 
                a = random.randrange(self.act_space)
                
            else: 
                a = self.get_action(s)

        return a
    
    
    def update_model(self):
        #function to update the network model 
        #sample batch from replay buffer
        batch = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*batch))
        
        states = torch.cat([s.to(device) for s in batch.state])
        new_states = torch.cat([ns.to(device) for ns in batch.next_state])
        actions = torch.cat([a.to(device) for a in batch.action])
        rewards = torch.cat([r.to(device) for r in batch.reward])
        dones = torch.tensor(batch.dones).float()
        
        #calculate state-action values of current state
        Q = self.policy(states).gather(1, actions.unsqueeze(1))
        
        #calculate expected values of next states using target network 
        next_values = (1-dones)*self.target(new_states).max(1)[0].detach()
        Q_tar = rewards + self.gamma*(next_values)
        
        loss = self.loss_fun(Q, Q_tar.unsqueeze(1))
        self.opt.zero_grad()    
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy.parameters(), 1)
        self.opt.step()
        
        

    def train(self):
        train_return = []

        step = 0
        for i in range(self.iters):  
            s = self.env.reset()
            episode = deque([])
            s = torch.tensor(s).float().unsqueeze(0)
            ep_step = 0
            train_rew = 0
            for j in range(self.env.max_num_steps): 
                #step through environment 
                a = self.greedy_action(s)
                s_new, r, done = self.env.step(a)
                s_new = torch.tensor(s_new).float().unsqueeze(0)
                r  = torch.tensor([r]).float()
                train_rew += r.item()
                a = torch.tensor([a])
                
                #store transition in replay buffer 
                t = Transition(s, a, s_new, r, done)
                self.memory.store(t)
                s = s_new
                
                
                if len(self.memory.memory) < self.batch_size: 
                    #if there are not enough samples in memory, pass 
                    pass
                else: 
                    self.update_model()
                                
                if done: 
                    break
                
            train_return.append(train_rew)
            
            
            #linear epsilon decay 
            self.eps =np.max([self.min_eps, ((self.min_eps - self.eps_start)/(self.iters*.95))*i + self.eps_start])
        
            if i % self.update == 0: 
                #copy policy parameters to target network 
                self.target.load_state_dict(self.policy.state_dict())
        '''        
        fig, ax = plt.subplots()
        it = np.linspace(0, self.iters, self.iters)
        ax.plot(it, train_return)
        ax.set_ylim([-10, 100])
        '''
        return self.policy,  train_return
        

    
   