This repository contains solutions to three assignments about Monte Carlo simulations and Reinforcement Learning concepts.

## Assignment 1: Gambler's Ruin

What was implemented:
- Simulated many gamblers playing a coin-flip game in parallel using NumPy arrays.
- Each gambler starts with the same money and gains or loses 1 unit on each coin flip.
- Used cumulative sums to build the full wealth path over time for all gamblers.
- Applied a mask so that once a gambler reaches zero, their wealth stays at zero for all future steps.
- Plotted:
  - A line plot showing the first set of gambler paths, along with the average path, best winner, and worst loser.
  - A histogram of final wealth across all gamblers.

## Assignment 2: Monte Carlo Geometry and Integration

What was implemented:
- Estimated pi by throwing random points into a square and checking how many fall inside the unit circle.
- Estimated e using random sampling methods based on areas under curves and a decreasing random sequence method.
- Wrote general functions that take a condition and bounds, then run a Monte Carlo simulation for different shapes or functions.
- Used these functions to:
  - Estimate the area of a circle.
  - Estimate areas under simple functions like a parabola and a Gaussian.
- Studied how the error changes when the number of random points increases.

## Assignment 3: Markov Decision Processes and Bellman Equations

What was implemented:
- Solved small Markov Decision Process examples on paper.
- Computed returns for fixed sequences of states and rewards with a discount factor.
- Wrote Bellman equations for value functions under a given policy.
- Discussed how different reward designs can lead to unintended behavior (reward hacking).
- Explained why discounting is needed and how it affects whether an agent is short-sighted or long-term.

## Fundamental Concepts

### Monte Carlo Simulation

Monte Carlo methods use random samples to approximate quantities that are hard to compute exactly. To estimate an area, a value like pi, or an integral, random points are generated in a known region, then the fraction of points satisfying a condition is used to estimate the desired value. As the number of samples grows, the estimate usually becomes more accurate, but the improvement is slow, and the error typically shrinks like 1 over the square root of the number of samples.

### Bellman Equation

A Markov Decision Process describes an environment where an agent moves between states, takes actions, and gets rewards. The value of a state is the expected total future reward from that state when following a given policy. The Bellman equation links the value of a state to the immediate reward and the value of the next state, which allows values to be computed or learned step by step instead of considering full trajectories at once.
