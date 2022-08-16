import numpy as np
import torch


class Pendulum:
    def __init__(self, initial_state=np.array([0, np.pi, 0]), dt=0.2, terminal_time=5, inner_step_n=2,
                 action_min=np.array([-2]), action_max=np.array([2]), penalty_coef=0.05):
        self.state_dim = 3
        self.action_dim = 1
        self.action_min = action_min
        self.action_max = action_max
        self.terminal_time = terminal_time
        self.dt = dt
        self.initial_state = initial_state
        self.inner_step_n = inner_step_n
        self.inner_dt = dt / inner_step_n
        self.penalty_coef = penalty_coef

        #physic params
        self.gravity = 9.8
        self.m = 1.
        self.l = 1.
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
            dynamics = np.array([1, state[2], 
                                 -3 * self.gravity / (2 * self.l) * np.sin(state[1] + np.pi) 
                                 + 3. / (self.m * self.l ** 2) * action[0]])
            state = state + dynamics * self.inner_dt

        reward = - self.penalty_coef * (action[0] ** 2) * self.dt
        done = False
        if state[0] >= self.terminal_time - self.dt / 2:
            reward = - np.abs(state[1]) - 0.1 * np.abs(state[2])
            done = True

        return state, reward, done, None        

