import numpy as np
from gym import spaces


class DubinsCar:
    def __init__(self, initial_state=np.array([0, 0, 0, 0]), dt=0.1, terminal_time=2 * np.pi, inner_step_n=10,
                 action_min=np.array([-0.5]), action_max=np.array([1]), penalty_coef=0.01):
        self.state_dim = 4
        self.action_dim = 1
        self.action_min = action_min
        self.action_max = action_max
        self.terminal_time = terminal_time
        self.dt = dt
        self.initial_state = initial_state
        self.inner_step_n = inner_step_n
        self.inner_dt = dt / inner_step_n
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
            state = state + np.array([1, np.cos(state[3]), np.sin(state[3]), action[0]]) * self.inner_dt

        reward = - self.penalty_coef * (action[0] ** 2) * self.dt
        done = False
        if state[0] >= self.terminal_time  - self.dt / 2:
            reward -= np.abs(state[1] - 4) + np.abs(state[2]) + np.abs(state[3] - 0.75 * np.pi)
            done = True

        return state, reward, done, {}