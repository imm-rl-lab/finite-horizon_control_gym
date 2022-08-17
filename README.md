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

## Environments

The following examples of finite-horizon optimal control problems are implemented:

### SimpleControlProblem (2,1)

The dynamical system is described by the simple motion:

$$
\dot{x}(t) = u(t),\quad t \in [0,2],\quad x(t) \in \mathbb R,\quad u(t) \in [-2,2],\quad x(0) = 1.
$$

The control tends to bring the system to $0$ at the terminal time:

$$
\gamma = - \|x(2)\|^2 - 0.5 \int_0^{11} u^2(t) d t
$$

### Pendulum (3,1)

Pendulum is a traditional problem for testing control algorithms. The dynamic model is taken from [OpenAI Gym](https://www.gymlibrary.ml/environments/classic_control/pendulum/):

$$
\dot{x}_1(t) = x_2(t),\quad \dot{x}_2(t) = \frac{3 g}{2 l} \sin(x_1(t)) + \frac{3}{m l^2} u(t),\quad t \in [0,5],\quad x(t) \in \mathbb R^2,\quad u(t) \in [-2,2],
$$

$$
x_1(0) = \pi,\quad x_2(0) = 0,\quad g=9.8,\quad m=l=1.
$$

The aim of the control is the stabilization of the pendulum in the top position at the terminal time:

$$
\gamma = - |x_1(5)\| - 0.1 |x_1(5)\| - 0.05 \int_0^{5} u^2(t) d t
$$

### VanDerPol (3,1)

VanDerPol oscillator is a famous model of a non-conservative oscillator with non-linear damping (see, e.g. [wikipedia](https://en.wikipedia.org/wiki/Van_der_Pol_oscillator)):

$$
\ddot{x}(t) - (1 - x^2(t)) \dot{x}(t) + x(t) = u(t),\quad t \in [0,11],\quad x(t) \in \mathbb R,\quad u(t) \in [-1,1],
$$
$$
x(0) = 1,\quad \dot{x}(0) = 0.
$$

The aim of the control is to stabilize the oscillator at the terminal time:

$$
\gamma = - \|x(11)\|^2 - 0.05 \int_0^{11} u^2(t) d t
$$

### DubinsCar (4,1)

DubinsCar  is a quite famous model which describes a motion of the point particle moving at a constant speed on the plane:

$$
\dot{x}(t) = \cos(\varphi(t)),\quad \dot{y}(t) = \sin(\varphi(t)),\quad \dot{\varphi}(t) = u(t),\quad t \in [0, 2 \pi],\quad (x(t),y(t),\varphi(t)) \in \mathbb R^3,\quad u(t) \in [-0.5, 1],
$$

$$
x(0) = 0,\quad y(0) = 0,\quad \varphi(0) = 0.
$$

The problem is to find a control providing the closeness of the motion with a target point at the terminal time:

$$
\gamma = - |x(2 \pi) - 4| - |y(2 \pi)| - |\varphi(2 \pi) - 0.75 \pi| - 0.01 \int_0^{2 \pi} u^2.
$$

### TargetProblem (7,2)

TargetProblem (7,2) is an optimal control problem presented in [Munos (2006)](https://www.jmlr.org/papers/volume7/munos06b/munos06b.pdf). The dynamical system
describes a hand holding a spring to which is attached a mass:

$$
\ddot{x}(t) = y(t) - x(t),\quad \dot{y}(t) = u(t),\quad t \in [0,10],\quad x(t), y(t) \in \mathbb R^2,\quad u_1(t), u_2(t) \in [-1,1],
$$

$$
x(0) = \dot{x}(0) = y(0) = 0. 
$$

It is required to control the hand such that the mass achieve the target point at the terminal time:

$$
\gamma = - \|x(10) - x_T\|^2 - \|y(10)\|^2 - 0.001 \int_0^{10} u^2,\quad x_T = (2,2).
$$

