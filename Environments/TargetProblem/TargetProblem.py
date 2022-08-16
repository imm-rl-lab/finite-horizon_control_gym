import numpy as np
import torch
from numpy.linalg import norm


class TargetProblem:
    def __init__(self, action_radius=np.array([1, 1]),
                 initial_state=np.array([0, 0, 0, 0, 0, 0, 0]),
                 terminal_time=10, dt=1, inner_step_n=10, target_point=[2, 2], penalty_coef=0.001):

        self.state_dim = 7
        self.action_dim = 2
        self.action_min = - action_radius
        self.action_max = + action_radius
        self.initial_state = initial_state
        self.terminal_time = terminal_time
        self.dt = dt
        self.inner_step_n = inner_step_n
        self.inner_dt = self.dt / self.inner_step_n
        self.penalty_coef = penalty_coef
        self.target_point = target_point
        #inner params
        self.k = 1
        self.m = 1
        self.g = 1
        return None

    def reset(self):
        self.state = self.initial_state       
        return self.state

    def step(self, action):
        self.state, reward, done, info = self.virtual_step(self.state, action)
        return self.state, reward, done, info

    def virtual_step(self, state, action):
        action = np.clip(action, self.action_min, self.action_max)
        
        for _ in range(self.inner_step_n):
            k1 = self.dynamics(state, action)
            k2 = self.dynamics(state + k1 * self.inner_dt / 2, action)
            k3 = self.dynamics(state + k2 * self.inner_dt / 2, action)
            k4 = self.dynamics(state + k3 * self.inner_dt, action)
            state = state + (k1 + 2 * k2 + 2 * k3 + k4) * self.inner_dt / 6

        reward = - self.penalty_coef * (norm(action) ** 2) * self.dt
        done = False
        if state[0] >= self.terminal_time - self.dt / 2:
            reward = - ((state[1] ** 2) + (state[2] ** 2) 
                        + ((state[3] - self.target_point[0]) ** 2) + ((state[4] - self.target_point[1]) ** 2))
            done = True

        return state, reward, done, {}

    def dynamics(self, state, action):
        return np.array([1, action[0], action[1], state[5], state[6],
                        -(self.k / self.m) * (state[3] - state[1]),
                        -(self.k / self.m) * (state[4] - state[2]) - self.g
                        ])
    