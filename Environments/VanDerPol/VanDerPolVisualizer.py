import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output


class VanDerPolVisualizer:
    def __init__(self, waiting_for_show=10):
        self.waiting_for_show = waiting_for_show
        self.total_rewards = []
        self.noise_thresholds = []

    def clean(self):
        self.total_rewards = []
        self.noise_thresholds = []
        
    def show_fig(self, env, agent, sessions):
        
        states = np.array([np.mean([session['states'][j] for session in sessions], axis=0) 
                           for j in range(len(sessions[0]['states']))])
        actions = np.array([np.mean([session['actions'][j] for session in sessions]) 
                              for j in range(len(sessions[0]['actions']))])
        plt.figure(figsize=[12, 12])

        plt.subplot(221)
        plt.plot([state[1] for state in states], [state[2] for state in states], 'm--', label='trajectory')
        plt.plot(states[-1][1],states[-1][2],'bo', label='terminal state')
        plt.xlim((-5, 5))
        plt.ylim((-5, 5))
        plt.grid()
        
        plt.subplot(222)
        plt.step(np.arange(len(actions)) * env.dt, [action for action in actions], 'g', label='actions')
        plt.xlim((0, env.terminal_time))
        plt.legend()
        plt.grid()

        plt.subplot(223)
        mean_total_rewards = np.mean(self.total_rewards[-20:])
        label = f'total_rewards: \n current={self.total_rewards[-1]:.2f} \n mean={mean_total_rewards:.2f}'
        plt.plot(self.total_rewards, 'g', label=label)
        plt.legend()
        plt.grid()

        plt.subplot(224)
        plt.plot(self.noise_thresholds, 'g', label='noise threshold')
        plt.legend()
        plt.grid()

        clear_output(True)
        
        plt.show()

    def show(self, env, agent, episode, sessions):
        total_reward = np.mean([sum(session['rewards']) for session in sessions])
        
        self.total_rewards.append(total_reward)
        self.noise_thresholds.append(agent.noise.threshold)
        
        if episode % self.waiting_for_show ==0:
            self.show_fig(env, agent, sessions)
