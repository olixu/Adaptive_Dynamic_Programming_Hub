# -*- encoding: utf-8 -*-
'''
@File    :   adp.py
@Time    :   2021/06/12 09:22:21
@Author  :   olixu 
@Version :   1.0
@Contact :   273601727@qq.com
@WebSite    :   https://blog.oliverxu.cn
'''

# Import the required libs
from logging import exception, info
from os import device_encoding
import sys
import random
import time
import pdb
import collections
import numpy as np
import matplotlib.pyplot as plt
import gym
from gym import spaces
import torch
from torch._C import FloatStorageBase, dtype
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from torchviz import *
import simulation_envs

device = 'cuda' if torch.cuda.is_available() else 'cpu'
buffer_limit = 2000
batch_size = 100
tau = 0.005


class adp_refactor():
    """Using the Gym-like api to design the adp class.

    The env is initialized instaed of implemented in the main class

    This class is just the algorithm implementation module.
    """
    def __init__(self, learning_num=10, epislon=1.4, learning_rate=1e-4, modeltype='linear'):
        self.env = simulation_envs.DoubleIntegratorEnv()
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.V_model = Critic(self.state_dim, self.action_dim).to(device)
        self.A_model = Actor(self.state_dim, self.action_dim).to(device)
        self.learning_num = learning_num
        self.epislon = epislon
        self.learning_rate = learning_rate
        self.criterion = torch.nn.MSELoss(reduction='sum') # 平方误差损失
        self.actor_optimizer = torch.optim.Adam(self.A_model.parameters(), lr=self.learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.V_model.parameters(), lr=self.learning_rate)
        self.trajectory = []
        if modeltype == "linear":
            self.A = torch.tensor(self.env.A, dtype=torch.float32).to(device)
            self.B = torch.tensor(self.env.B, dtype=torch.float32).to(device)
            self.Q = torch.tensor(self.env.Q, dtype=torch.float32).to(device)
            self.R = torch.tensor(self.env.R, dtype=torch.float32).to(device)
            self.gamma = 0.99
            
    def learning(self):
        """
        The training process of the ADP algorithm.
        """
        for epoch in range(1):
            print("\033[1;32m 训练第 {} 个epoch \033[0m ".format(epoch))
            state = torch.from_numpy(self.env.reset()).float().to(device)
            for i in range(self.learning_num):
                delta_a = 1.0
                delta_c = 1.0
                print("\033[1;35m Training the {} --th episode \033[0m ".format(i))
                ###########################################################################################################################
                # STEP：1. 使用best_action与actor网络输出的action构造loss function并更新Actor
                ###########################################################################################################################
                while (delta_a > 1e-4):
                    J = self.V_model(state)
                    # 对于连续情况下的特殊情形:采用《discrete-time nonlinear HJB solution using approximate dynamic programming》中表达式17来进行计算
                    best_action = - (torch.inverse(self.R + (self.B.T * J).mm(self.B)).mm(self.B.T) * J).mm(self.A).mm(state.unsqueeze(0).T).squeeze(1)
                    actor_action = self.A_model(state)
                    actor_loss = self.criterion(best_action, actor_action)
                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    self.actor_optimizer.step()
                    delta_a = actor_loss.item()

                ###########################################################################################################################
                # STEP：2. 使用TD的方式更新critic网络
                ###########################################################################################################################
                # critic网络每次迭代后面误差会越来越大
                next_state, reward, done, info = self.env.step(best_action.detach().cpu().numpy())
                while (delta_c > 1e-4):
                    predict_value_k = self.V_model(state)
                    predict_value_next_k = self.V_model(torch.from_numpy(next_state).float().to(device))
                    target_value_k = reward + self.gamma * predict_value_next_k
                    critic_loss = self.criterion(predict_value_k, target_value_k)
                    self.critic_optimizer.zero_grad()
                    critic_loss.backward()
                    self.critic_optimizer.step()
                    delta_c = critic_loss.item()
                
                # 系统状态跳转到下一个状态
                state = torch.from_numpy(next_state).float().to(device)
                print("状态是：", state)

# 还有900多行代码涉及未发表论文的仿真，暂时就不开源了。

def test_unsafe_adp():
    """测试unsafe adp算法
    """
    adpsolver = adp_refactor_for_nonlinearenv(learning_num=100, learning_rate=0.1, modeltype='linear')
    adpsolver.learning()

if __name__=='__main__':
    test_unsafe_adp()