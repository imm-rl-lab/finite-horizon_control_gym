# Finite-Horizon Control Gym

The repository contains examples of finite-horizon optimal control problems implemented as environments (MDPs) for reinforcement learning algorithms. Since the problems are initially described by differential equations, in order to formalize them as MDPs, a uniform time-discretization with the diameter <code>dt</code> is used. In addition, it is important to emphasize that, in the problems with a finite horizon, optimal policies depend not only on the phase vector $x$, but also on time $t$. Thus, we obtain MDPs, depending on <code>dt</code>, with continuous stats space $S$ containing states $s=(t,x)$ and continuous action space $A$. 

## Interface

The finite-horizon optimal control problems are implemented as environments with an interface close to [OpenAI Gym](https://www.gymlibrary.ml/) with the following attributes: 

- <code>state_dim</code> - the state space dimension; 
- <code>action_dim</code> - the action space dimension;
- <code>terminal_time</code> - the action space dimension;
- <code>dt</code> - the time-discretization diameter;
- <code>reset()</code> - to get an initial <code>state</code> (deterministic);
- <code>step(action)</code> - to get <code>next_state</code>, current <code>reward</code>, <code>done</code> (<code>True</code> if <code>t > terminal_time</code>, otherwise <code>False</code>), <code>info</code>;
- <code>virtual_step(state, action)</code> - to get the same as from <code>step(action)</code>, but but the current <code>state</code> is also set

## Examples of finite-horizon optimal control problems

The following examples of finite-horizon optimal control problems are implemented:

- **SimpleControlProblem (2,1)** is described by a dynamical system with simple motion. The control tends to bring the system to $0$ at the terminal_time from the point $1$;

- **Pendulum (3,1)** is a traditional problem for testing control algorithms. The dynamic model is taken from [OpenAI Gym](https://www.gymlibrary.ml/environments/classic_control/pendulum/). The aim of the control is the stabilization of the pendulum in the top position at the terminal time;

- **VanDerPol (3,1)** oscillator is a famous model of a non-conservative oscillator with non-linear damping (see, e.g. [wikipedia](https://en.wikipedia.org/wiki/Van_der_Pol_oscillator)). The aim of the control is to stabilize the oscillator at the terminal time;

- **DubinsCar (4,1)**  is a quite famous model which describes a motion of the point particle moving at a constant speed on the plane. The problem is to find a control providing the closeness of the motion with a target point at the terminal time;

- **TargetProblem (7,2)** is an optimal control problem presented in [Munos (2006)](https://www.jmlr.org/papers/volume7/munos06b/munos06b.pdf). The dynamical system
describes a hand holding a spring to which is attached a mass. It is required to control the hand such that the mass achieve the target point at the terminal time

