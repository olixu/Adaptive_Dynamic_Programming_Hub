# -*- encoding: utf-8 -*-
'''
@File    :   simulation_envs.py
@Time    :   2021/06/25 17:49:03
@Author  :   olixu 
@Version :   1.0
@Contact :   273601727@qq.com
@WebSite    :   https://blog.oliverxu.cn
'''

"""
This module is the environments for usage of the paper.

环境接口参考资料：https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
主要注意点：

1. 模块所有的接口数据都是np.array格式的数据，而不是torch.tensor(考虑到环境解耦，在env文件中，只引入numpy包)，但是由于网络的数据都是torch.tensor，所以这样会不会导致网络程序运行过程中的效率问题
2. 很多RL的实现，都是在repaly buffer的sample方法的输出中，将np.array转换成tensor
"""

# here put the import lib
import math
from typing import Any, Dict, Tuple
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import pdb

from numpy.lib.utils import info
from torch._C import dtype

class DoubleIntegratorEnv(gym.Env):
    """
    Description:
        Double Integrator Environment with state of 4 dimension, the first two elements of the state
        can be seen as a mass point moving on a plain. 
        The equilibrium is set as the original point, the control objective of this environment is 
        to control the mass point to the equulibrium point.

    Source:
        From the paper 《》

    Observation:
        Type: Box(4)
        Num Observation Min Max
        0   x0  -Inf    Inf
        1   x1  -Inf    Inf
        2   x2  -Inf    Inf
        3   x3  -Inf    Inf

    Actions:
        Type: Box(2)
        Num Action
        0   a_1
        1   a_2

    Reward:
        Reward is defined as the quadratic performance of the system like the form: X^TQX + u^TRU

    Episode Termination:
        For this simple control system, we consider that at the time instant 30, the iteration is terminated.


    """
    def __init__(self, state=np.array([3.0, 0.0, 0, 0]), Q=1/50.0*np.eye(4), R=1/72.0*np.eye(2)) -> None:
        super().__init__()
        self.A = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]])
        self.B = np.array([[1/2, 0], [0, 1/2], [1, 0], [0, 1]])
        self.Q = Q
        self.R = R
        self.state = state
        [self.state_dim, self.action_dim] = self.B.shape
        state_high = np.array([np.inf, np.inf, np.inf, np.inf])
        action_high = np.array([np.inf, np.inf])
        self.observation_space = spaces.Box(-state_high, state_high, dtype=np.float)
        self.action_space = spaces.Box(-action_high, action_high, dtype=np.float)

    def step(self, action: np.array) -> Tuple[np.array, float, bool, Dict[str, Any]]:
        next_state = np.matmul(self.A, self.state) + np.matmul(self.B, action)
        reward = np.matmul(np.matmul(self.state.reshape(-1, 1).T, self.Q), self.state.reshape(-1, 1)).item() + \
                 np.matmul(np.matmul(action.reshape(-1, 1).T, self.R), action.reshape(-1, 1)).item()
        done = False
        info = {}
        self.state = next_state
        return next_state, reward, done, info

    def reset(self, state=np.array([3.0, 0.0, 0, 0])):
        self.state = state
        return self.state

    def render(self):
        raise NotImplementedError("Error: render is not implemented for this environment.")
