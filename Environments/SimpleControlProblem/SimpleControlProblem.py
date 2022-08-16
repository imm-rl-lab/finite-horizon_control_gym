import numpy as np
from gym import spaces


class SimpleControlProblem:
    def __init__(self, dt=0.05, terminal_time=2, initial_state=np.array([0, 1]), inner_step_n=10,
                 action_min=np.array([-1]), action_max=np.array([1]), penalty_coef=0.5):
        self.state_dim = 2
        self.action_dim = 1
        self.dt = dt
        self.terminal_time = terminal_time
        self.initial_state = initial_state
        self.action_min = action_min
        self.action_max = action_max
        self.penalty_coef = penalty_coef
        self.inner_step_n = inner_step_n
        self.inner_dt = dt / inner_step_n
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
            state = state + np.array([1, action[0]]) * self.inner_dt

        reward = - self.penalty_coef * action[0] ** 2 * self.dt
        done = False
        if state[0] >= self.terminal_time - self.dt / 2:
            reward -= state[1] ** 2
            done = True
        
        return state, reward, done, {}
    