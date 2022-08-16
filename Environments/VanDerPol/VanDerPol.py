import numpy as np
import torch


class VanDerPol:
    def __init__(self, initial_state=np.array([0,1,0]), action_min=np.array([-1]), action_max=np.array([1]), 
                 terminal_time=11, dt=0.1, inner_step_n=10, penalty_coef=0.05):
        self.state_dim = 3
        self.action_dim = 1
        self.action_min = action_min
        self.action_max = action_max
        self.terminal_time = terminal_time
        self.dt = dt
        self.inner_step_n = inner_step_n
        self.inner_dt = self.dt / self.inner_step_n
        self.initial_state = initial_state
        self.penalty_coef = penalty_coef
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
            dynamics = np.array([1, state[2], (1 - state[1] ** 2) * state[2] - state[1] + action[0]])
            state = state + dynamics * self.inner_dt
        
        done = False
        reward = - self.penalty_coef * action[0] ** 2 * self.dt
        if state[0] >= self.terminal_time - self.dt / 2:
            done = True
            reward = - state[1] ** 2 - state[2] ** 2
        
        return state, reward, done, _

