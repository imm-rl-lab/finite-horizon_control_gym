import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output


class EarthOrbitalMotionVisualizer:
    def __init__(self, waiting_for_show=10):
        self.total_rewards = []
        self.noise_thresholds = []
        self.mean_reward = [0]
        self.waiting_for_show = waiting_for_show
    
    def show_fig(self, env, agent, sessions, episode):
        
        states = np.array([np.mean([session['states'][j] for session in sessions], axis=0) 
                           for j in range(len(sessions[0]['states']))])
        actions = np.array([np.mean([session['actions'][j] for session in sessions], axis=0) 
                              for j in range(len(sessions[0]['actions']))])
        
        total_rewards = self.total_rewards[-1000:]
        mean_rewards = self.mean_reward[-1000:]
        plt.figure(figsize=[18, 9])
        
        plt.subplot(2,3,1)
        plt.plot(states[-1][1]*np.cos(states[-1][3]),states[-1][1]*np.sin(states[-1][3]),'go', label='terminal state')
        plt.plot([state[1]*np.cos(state[3]) for state in states],[state[1]*np.sin(state[3]) for state in states],'g--', 
                 label='trajectory')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.grid()

        plt.subplot(2,3,2)
        label = f'total_rewards: \n current={self.total_rewards[-1]:.2f} \n min={min(self.total_rewards):.2f} \n max={max(self.total_rewards):.2f}'
        plt.plot(self.total_rewards, 'g', label=label)
        #plt.plot(mean_rewards, 'b', label='mean_reward')
        plt.legend()
        plt.grid()
        
        plt.subplot(2,3,3)
        plt.plot(self.noise_thresholds, 'b', label='agent_noise_threshold')
        plt.legend()
        plt.grid()
        
        plt.subplot(2,2,3)
        plt.step(np.arange(len(actions)) * env.dt, [action[0] for action in actions],'g', label='action[0]')
        plt.legend()
        plt.grid()
        
        plt.subplot(2,2,4)
        plt.step(np.arange(len(actions)) * env.dt, [action[1] for action in actions],'g', label='action[1]')
        plt.legend()
        plt.grid()

        clear_output(True)
        
        states[-1] = states[-1]/env.normalized_vector
        
        print(f"\t episode={episode:.0f}, t={states[-1][0]:.1f},\t total reward={self.total_rewards[-1]:.3f}")
        print(f"\t final r = {states[-1][1]:.3f},\t final psi_dot = {states[-1][4]:.6f}")
        print(f"\t required r = {env.required_orbit[0]:.3f},\t required psi_dot = {env.required_orbit[1]:.6f}")
        plt.show()
            
    def show(self, env, agent, episode, sessions):
        total_reward = np.mean([sum(session['rewards']) for session in sessions])
        
        self.total_rewards.append(total_reward)
        self.noise_thresholds.append(agent.noise.threshold)
        
        if episode % self.waiting_for_show ==0:
            self.show_fig(env, agent, sessions, episode)
            
    def clean(self):
        self.total_rewards = []
        self.noise_thresholds = []