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

Transition = namedtuple('Transition',
                ('state', 'action', 'next_state', 'reward', 'dones'))
class Policy(nn.Module): 
    def __init__(self, input_dim, output_dim):
        super(Policy, self).__init__()
        self.layers = nn.Sequential(nn.Linear(input_dim, 64),
                                    nn.Tanh(),
                                    nn.Linear(64,64),
                                    nn.Tanh(),
                                    nn.Linear(64,64), 
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
        self.gamma = gamma
        self.learn_rate = lr
        
        self.eps_start = eps 
        self.min_eps = 0.1
        
        self.iters = iters
        self.batch_size = batch 
        self.use_memory = use_memory 
        self.update = update_every 
        
        self.env = env
        
        self.act_space = env.num_actions 
        self.obs_space = env.num_states
        
        
        return 
    
    def train_plots(self, loss, step, eval_rew, eval_step): 
        steps = np.linspace(0, step, step)
        fig, ax = plt.subplots()
        ax.plot(eval_step, eval_rew, color = 'orange')
        ax.set_ylim([-25, 25])
        ax.set_ylabel('return')
        ax.set_xlabel('steps')
        
        return 
    
    def greedy_action(self, s): 
        with torch.no_grad():
            p = np.random.random()
            
            if p < self.eps: 
                a = random.randrange(self.act_space)
                
            else: 
                d = self.policy(torch.tensor(s).float())
                a = torch.argmax(d).numpy().item()
                
        return a
    
    def rollout(self, num): 
        with torch.no_grad():
            rollouts = deque([], maxlen=num)
            for i in range(num): 
                s = self.env.reset()
                done = False
                episode = deque([])
                while not done: 
                    a = self.greedy_action(s)
                    s_new, r, done = self.env.step(a)
                    episode = Transition(s, a, s_new, r, done)
                    self.memory.store(episode)
                        
                    s = s_new
                    
                    rollouts.append(episode)
        return rollouts
    
    def evaluation(self): 
        with torch.no_grad():
            s = self.env.reset()
            done = False
            returns = 0
            while not done: 
                a = torch.argmax(self.policy(torch.tensor(s).float()))
                s_new, r, done = self.env.step(a)
                returns += r
                s = s_new
            
        return returns
    
    def train(self):
        self.policy = Policy(self.obs_space, self.act_space).float()
        self.target = Policy(self.obs_space, self.act_space).float()
        
        if self.use_memory == True: 
            self.memory = Replay(10000)
        else: 
            self.memory = Replay(self.batch_size)
            
        self.eps = self.eps_start 
        
        opt = torch.optim.AdamW(self.policy.parameters(), lr = self.learn_rate, amsgrad= True)
        #opt = torch.optim.RMSprop(self.policy.parameters(), lr = self.learn_rate)
        loss_fun = nn.MSELoss()
        step = 0
        eval_return = []
        
        for i in range(self.iters):
            #get rollouts
            rollout = self.rollout(self.batch_size) 
             
            
            batch = self.memory.sample(self.batch_size)
            batch = Transition(*zip(*batch))
            
            states = torch.tensor(batch.state).float()
            new_states = torch.tensor(batch.next_state).float()
            rewards = torch.tensor(batch.reward).float()
            dones = torch.tensor(batch.dones)
            actions = torch.tensor(batch.action)
            
            q = self.policy(states).gather(1, actions.reshape(actions.shape[0], 1))
            
            est_value = self.gamma*(self.target(new_states)).max(1)[0]
            
            #for terminal states, replace \gamma*Q_tar w/ 0 
            with torch.no_grad():
                for k in range(len(dones)): 
                    if dones[k] == True :
                        est_value[k] = 0 
            
            q_tar = rewards + est_value
            
            opt.zero_grad()
            loss = loss_fun( q.reshape(q.shape[0], 1), q_tar.reshape(q_tar.shape[0], 1))
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.policy.parameters(), 100)
            opt.step()
            step += 1 
            
            if (i+1) % self.update == 0: 
                self.target.load_state_dict(self.policy.state_dict())
            
            eval_rew = self.evaluation()
            eval_return.append(np.sum(eval_rew))
            
            if (i+1) % 5 == 0: 
                self.eps = np.max([self.eps*.99, self.min_eps])
                print("training iteration: " + str(i))
                print('epsilon: ' + str(self.eps))
            
        return self.policy,  eval_return, step
        
    def train_avg(self): 
        _,  er1, s1 = self.train()
        _,  er2, s2 = self.train()
        _, er3, s3 = self.train()
     
        er = np.stack([er1, er2, er3]) 
        er_avg = np.mean(er, axis = 0)
        er_std = np.std(er, axis = 0)
        er = np.stack([er_avg, er_std])
        
        step = s3
        return  er, step 
    
    def train_plot_avg(self): 
        er, es = self.train_avg() 
        
        s = np.linspace(0, es, es)
        fig, ax = plt.subplots()
        ax.plot(s, er[0])
        ax.fill_between(s, (er[0]-er[1]), (er[0]+er[1]), alpha = .25)
        ax.set_ylabel('return')
        ax.set_xlabel('iteration')
        ax.set_ylim([-200, 200])
        
        fig.savefig('./figures/learning_curve.png')
        
    