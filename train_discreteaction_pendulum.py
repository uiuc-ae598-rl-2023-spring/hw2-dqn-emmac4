# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 13:43:45 2023

@author: clemm
"""
from Algorithm import DQN 
import matplotlib.pyplot as plt
import numpy as np
import discreteaction_pendulum
import torch 


def plot_traj(env, policy): 
    # Initialize simulation
    s = env.reset()

    # Create dict to store data from simulation
    data = {
        't': [0],
        's': [s],
        'a': [],
        'r': [],
    }

    # Simulate until episode is done
    done = False
    rew =  0 
    while not done:
        a = torch.argmax(policy(torch.tensor(s).float()))
        (s, r, done) = env.step(a)
        rew += r
        data['t'].append(data['t'][-1] + 1)
        data['s'].append(s)
        data['a'].append(a)
        data['r'].append(rew)

    # Parse data from simulation
    data['s'] = np.array(data['s'])
    theta = data['s'][:, 0]
    thetadot = data['s'][:, 1]
    tau = [env._a_to_u(a) for a in data['a']]

    # Plot data and save to png file
    fig, ax = plt.subplots(3, 1, figsize=(10, 10))
    ax[0].plot(data['t'], theta, label='theta')
    ax[0].plot(data['t'], thetadot, label='thetadot')
    ax[0].legend()
    ax[1].plot(data['t'][:-1], tau, label='tau')
    ax[1].legend()
    ax[2].plot(data['t'][:-1], data['r'], label='return')
    ax[2].legend()
    ax[2].set_xlabel('time step')
    plt.tight_layout()
    fig.savefig('./figures/traj.png')
    
def plot_policy(env, policy):
    theta = np.linspace(-np.pi, np.pi, 100)
    thetadot = np.linspace(-15, 15, 100)
    action = np.zeros([len(theta), len(thetadot)])
    
    for i in range(len(theta)): 
        for j in range(len(thetadot)): 
            s = torch.tensor([theta[i], thetadot[j]]).float()
            a = torch.argmax(policy(s)).detach()
            action[i,j] = env._a_to_u(a)
    
    fig2, ax2 = plt.subplots()
    c = ax2.contourf(theta, thetadot, action, alpha = .75)
    ax2.set_xlabel(r'$\theta$')
    ax2.set_ylabel(r'$\dot{\theta}$')
    cbar = fig2.colorbar(c)
    cbar.ax.set_ylabel(r'$\tau$')
    fig2.savefig('./figures/policy.png')
    
def plot_value(env, policy):
    theta = np.linspace(-np.pi, np.pi, 100)
    thetadot = np.linspace(-15, 15, 100)
    value = np.zeros([len(theta), len(thetadot)])
    
    for i in range(len(theta)): 
        for j in range(len(thetadot)): 
                s = torch.tensor([theta[i], thetadot[j]]).float()
                v = torch.max(policy(s)).detach()
                value[i,j] = v
    
    fig3, ax3 = plt.subplots()
    c = ax3.contourf(theta, thetadot, value, alpha = .75)
    ax3.set_xlabel(r'$\theta$')
    ax3.set_ylabel(r'$\dot{\theta}$')
    cbar = fig3.colorbar(c)
    cbar.ax.set_ylabel('value')
    fig3.savefig('./figures/value.png')
    
def ablation(env): 
    rt = DQN(env = env, gamma = 0.95, lr = 0.01, eps = .9, batch = 32, iters = 200, use_memory = True, update_every = 20)
    nrt = DQN(env = env, gamma = 0.95, lr = 0.01, eps = .9, batch = 32, iters = 200, use_memory = False, update_every = 20)
    rnt = DQN(env = env, gamma = 0.95, lr = 0.01, eps = .9, batch = 32, iters = 200, use_memory = True, update_every = 1)
    nrnt = DQN(env = env, gamma = 0.95, lr = 0.01, eps = .9, batch = 32, iters = 200, use_memory = False, update_every = 1)
    
    er1, step1 = rt.train_avg()
    er2, step2 = nrt.train_avg()
    er3, step3 = rnt.train_avg()
    er4, step4 = nrnt.train_avg()

    s1 = np.linspace(0, step1, step1)
    s2 = np.linspace(0, step2, step2)
    s3 = np.linspace(0, step3, step3)
    s4 = np.linspace(0, step4, step4)

    fig, ax = plt.subplots()
    ax.plot(s1, er1[0], label = 'replay & target')
    ax.fill_between(s1, (er1[0]-er1[1]), (er1[0]+er1[1]), alpha = .25)
    
    ax.plot(s2, er2[0], label = 'target')
    ax.fill_between(s2, (er2[0]-er2[1]), (er2[0]+er2[1]), alpha = .25)
    
    ax.plot(s3, er3[0], label = 'replay')
    ax.fill_between(s3, (er3[0]-er3[1]), (er3[0]+er3[1]), alpha = .25)
    
    ax.plot(s4, er4[0], label = 'none')
    ax.fill_between(s4, (er4[0]-er4[1]), (er4[0]+er4[1]), alpha = .25)
    
    ax.set_ylabel('return')
    ax.set_xlabel('iteration')
    ax.legend()
    
    fig.savefig('./figures/ablation.png')
    
def main():
    env = discreteaction_pendulum.Pendulum()
    dqn = DQN(env = env, gamma = 0.95, lr = 0.025, eps = .9, batch = 64, iters = 200, use_memory = True, update_every = 20)
    dqn.train_plot_avg()
    policy, _, _ = dqn.train()

    plot_traj(env, policy)
    env.video(policy, filename='figures/test_discreteaction_pendulum.gif')
    plot_policy(env, policy)
    #ablation(env)
    plot_value(env, policy)
    return
    
    
    
if __name__ == '__main__':
    main()
