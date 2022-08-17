import numpy as np


class EarthOrbitalMotion():
    def __init__(self, action_radius=np.array([0.5, 0.5]), initial_state=np.array([0, 6900, 0, 0, 0.001109]), 
                 normalized_vector=np.array([1/7000, 1/7000, 100, 1, 100]), terminal_time=9000, 
                 dt=200, inner_step_n=1, required_orbit=7100, penalty_coef=0.001):
        self.state_dim = 5
        self.action_dim = 2
        self.action_radius = action_radius
        self.action_min = - self.action_radius
        self.action_max = + self.action_radius
        self.terminal_time = terminal_time
        self.dt = dt
        self.inner_step_n = inner_step_n
        self.inner_dt = self.dt / self.inner_step_n
        self.normalized_vector = normalized_vector # state coordinate scaling vector
        self.penalty_coef = penalty_coef
        #inner params
        self.M = 5.9726e24 # mass of the Earth, kg
        self.m = 50 # satellite mass, kg
        self.R = 6371 # Earth radius, km
        self.G_const = 6.67448478e-20 # gravitational constant, km^3/(kg*sec^2)
        #set initial state and required_orbit 
        self.initial_state = initial_state
        self.initial_state[4] = self.get_rotation_speed(initial_state[1]) # refinement of the orbital speed of rotation around the Earth
        self.required_orbit = np.array([required_orbit, self.get_rotation_speed(required_orbit)])
        return None
        
    def dynamics(self, state, action):
        return np.array([1, state[2], state[1]*state[4]**2 - self.G_const*self.M/(state[1]**2) + action[0]/(1000*self.m), 
                         state[4], -2*state[4]*state[2]/state[1] + action[1]/(state[1]*self.m*1000)])

    def reset(self):
        self.state = self.initial_state * self.normalized_vector
        return self.state

    def step(self, action):
        self.state = self.state / self.normalized_vector
        for _ in range(self.inner_step_n):
            k1 = self.dynamics(self.state, action)
            k2 = self.dynamics(self.state + k1 * self.inner_dt / 2, action)
            k3 = self.dynamics(self.state + k2 * self.inner_dt / 2, action)
            k4 = self.dynamics(self.state + k3 * self.inner_dt, action)
            self.state = self.state + (k1 + 2 * k2 + 2 * k3 + k4) * self.inner_dt / 6
            
        reward = - self.penalty_coef * np.linalg.norm(action) * self.dt
        done = False
        if self.state[0] >= self.terminal_time - self.dt / 2:
            reward = -np.linalg.norm([self.required_orbit[0]-self.state[1], 100000*(self.required_orbit[1] - self.state[4])])
            done = True

        self.state = self.state * self.normalized_vector
        return self.state, reward, done, None
    
    def get_rotation_speed(self, orbit):
        return np.sqrt(self.G_const * self.M / (orbit ** 3))
