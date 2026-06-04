# Autonomous Pacman Agent — Markov Decision Processes & Value Iteration

## Overview
Developed a fully autonomous AI agent that plays Pacman, navigating small and medium mazes while avoiding ghosts — without any hardcoded rules. The agent uses **Markov Decision Processes (MDP)** and **Value Iteration** to compute an optimal policy at every game step, reasoning about rewards, ghost positions, and future outcomes.

Built on the UC Berkeley CS188 Pacman AI framework as part of the Agent Reasoning & Decision Making module at King's College London.

## How it works

### MDP Formulation
| Component | Definition |
|-----------|-----------|
| States (S) | All valid (x, y) grid positions excluding walls |
| Actions (A) | {North, South, East, West} |
| Transition T(s, a, s') | Deterministic — wall collision returns current state |
| Reward R(s, a, s') | Food pellets, ghost proximity penalties, step cost |
| Discount factor γ | 0.88 — balances immediate vs future rewards |

### Reward Function Design
```
+18   Food pellet at next position
-1    Step penalty (encourages efficiency)
-120  Ghost at Manhattan distance = 1 (immediate danger)
-12   Ghost at Manhattan distance = 2 (proximity warning)
-5000 Ghost collision (terminal penalty)
+1    Moving closer to nearest food (shaping reward)
```

### Value Iteration
Solves the Bellman optimality equation iteratively:
```
V(s) = max_a [ R(s, a, s') + γ · V(s') ]
```
Runs **20 iterations per game step** across all valid states, converging to an optimal value function. The best action is then selected by computing Q-values for each legal action at Pacman's current position.

## Architecture

```
getAction()
  └─ extractWorld()        # reads pacman, food, ghost positions from API
  └─ runValueIteration()   # 20 iterations, updates V(s) for all states
  └─ getBestAction()       # argmax_a Q(pacman_pos, a)
  └─ api.makeMove()        # executes chosen action
```

## Key implementation details
- **Wall handling** — grid built from `state.getWalls()` (correct Grid object, not raw list), stored as a set for O(1) lookup
- **Ghost radius** — dual-distance penalty (distance 1 and 2) creates a soft repulsion field around ghosts
- **Legal action filtering** — fallback to random legal move if value iteration returns an illegal action
- **State initialisation** — `registerInitialState()` pre-computes all valid states once at game start

## Results
- Successfully navigated **small mazes** — consistently avoided ghosts and collected all food
- Successfully navigated **medium mazes** — policy generalised to larger state spaces with more complex ghost interactions

## Tech stack
`Python` `Berkeley AI Framework (CS188)` `MDP` `Value Iteration`

## Files
| File | Description |
|------|-------------|
| `mdpAgents.py` | Main agent — MDP formulation, Value Iteration, reward function, policy extraction |

## Attribution
Built on the UC Berkeley Pacman AI Projects (ai.berkeley.edu). Framework by John DeNero and Dan Klein. Academic use only.
