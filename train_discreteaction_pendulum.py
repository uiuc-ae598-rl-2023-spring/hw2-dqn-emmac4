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

        data['t'].append(data['t'][-1] + 1)
        data['s'].append(s)
        data['a'].append(a)
        data['r'].append(r)

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
    #plot state-action policy 
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
    #plot state-value function 
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
    
def get_avg_train(dqn, iters): 
    #trains model 5 separate times 
    #   - gets average and standard deviation of the training returns 
    #   - choose policy with the highest return over the second half of trianing and returns this policy 
    
    p1, r1 = dqn.train()
    p2, r2 = dqn.train()
    p3, r3 = dqn.train()
    p4, r4 = dqn.train()
    p5, r5 = dqn.train()


    policies = np.stack([p1, p2, p3, p4, p5])
    reward = np.stack([r1, r2, r3, r4, r5])
    r_avg = np.mean(reward, axis = 0)
    r_std = np.std(reward, axis = 0)
    
    rew = reward[:, int((iters/2)):]
    x = np.sum(rew, axis = 1)
    p_max = np.argmax(x)
    
    return policies[p_max], r_avg, r_std 

def plot_avg(dqn, iters): 
    #creates a plot of returns over training averaged over several training iterations 
    max_policy, r_avg, r_std = get_avg_train(dqn, iters)
    
    i = np.linspace(0, iters, iters)
    fig1, ax1 = plt.subplots()
    #plot avg return
    ax1.plot(i, r_avg)
    #plot avg return +- standard deviation 
    ax1.fill_between(i, (r_avg + r_std), (r_avg - r_std), alpha = .25)
    ax1.set_ylim([-10, 100])
    ax1.setxlabel('iterations')
    ax1.set_ylabel('return')
    fig1.savefig('./figures/learning_curve.png')
    
    return max_policy

def ablation(env, iters): 
    #ablation study for DQN w/ replay&target, replay only, target only, and neither 
    dqn_rt = DQN(env = env, gamma = 0.95, lr = 0.001, eps = 1.0, batch = 128, iters = iters, use_memory = True, update_every = 10)
    dqn_t = DQN(env = env, gamma = 0.95, lr = 0.001, eps = 1.0, batch = 128, iters = iters, use_memory = False, update_every = 10)
    dqn_r = DQN( env = env, gamma = 0.95, lr = 0.001, eps = 1.0, batch = 128, iters = iters, use_memory = True, update_every = 1)
    dqn_none = DQN( env = env, gamma = 0.95, lr = 0.001, eps = 1.0, batch = 128, iters = iters, use_memory = False, update_every = 1)
    p_rt, avg_rt, std_rt = get_avg_train(dqn_rt, iters)
    p_t, avg_t, std_t = get_avg_train(dqn_t, iters)
    p_r, avg_r, std_r = get_avg_train(dqn_r, iters)
    p_n, avg_n, std_n = get_avg_train(dqn_none, iters)
    
    print(std_rt)
    
    i = np.linspace(0, iters, iters)
    fig3, ax3 = plt.subplots()
    ax3.plot(i, avg_rt, label = 'replay & target')
    ax3.fill_between(iters, (avg_rt + std_rt), (avg_rt - std_rt), alpha = .25)

    ax3.plot(i, avg_t, label = 'target')
    ax3.fill_between(iters, (avg_t + std_t), (avg_t - std_t), alpha = .25)
    
    ax3.plot(i, avg_r, label = 'replay')
    ax3.fill_between(iters, (avg_r + std_r), (avg_r - std_r), alpha = .25)
    
    ax3.plot(i, avg_n, label = 'none')
    ax3.fill_between(iters, (avg_n + std_n), (avg_n - std_n), alpha = .25)
    
    ax3.legend()
    ax3.set_xlabel('iterations')
    ax3.set_ylabel('return')
    ax3.set_ylim([-10,100])
    fig3.savefig('./figures/ablation.png')
    
def main():
    env = discreteaction_pendulum.Pendulum()
    iters = 100
    dqn = DQN(env = env, gamma = 0.95, lr = 0.001, eps = 1.0, batch = 128, iters = iters, use_memory = True, update_every = 10)
    
    #plot training curve
    policy = plot_avg(dqn, iters)
    
    #save policy 
    #torch.save(policy.state_dict(), './policy')
    
    #plot trajectory
    plot_traj(env, policy)
    
    #get video of an episode w/ policy 
    env.video(policy, filename='figures/test_discreteaction_pendulum.gif')
    
    plot_policy(env, policy)
    plot_value(env, policy)
    
    #ablation study 
    ablation(env, iters)

    return
    
    
    
if __name__ == '__main__':
    main()
