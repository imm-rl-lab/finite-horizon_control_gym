# Finite-Horizon Control Gym

The repository contains examples of finite-horizon optimal control problems implemented as environments (MDPs) for reinforcement learning algorithms. Since the problems are initially described by differential equations, in order to formalize them as MDPs, a uniform time-discretization with the diameter <code>dt</code> is used. In addition, it is important to emphasize that, in the problems with a finite horizon, optimal policies depend not only on the phase vector $x$, but also on time $t$. Thus, we obtain MDPs, depending on <code>dt</code>, with continuous stats space $S$ containing states $s=(t,x)$ and continuous action space $A$. They are implemented as environments with an interface close to [OpenAI Gym](https://www.gymlibrary.ml/) with the following attributes: 

- <code>state_dim</code> - the state space dimension; 
- <code>action_dim</code> - the action space dimension;
- <code>terminal_time</code> - the action space dimension;
- <code>dt</code> - the time-discretization diameter;
- <code>reset()</code> - to get an initial <code>state</code> (deterministic);
- <code>step(action)</code> - to get <code>next_state</code>, current <code>reward</code>, <code>done</code> (<code>True</code> if <code>t > terminal_time</code>, otherwise <code>False</code>), <code>info</code>;
- <code>virtual_step(state, action)</code> - to get the same as from <code>step(action)</code>, but but the current <code>state</code> is also set.
