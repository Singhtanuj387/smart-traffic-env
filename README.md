---
title: Smart Traffic
emoji: 🚦
colorFrom: green
colorTo: blue
sdk: docker
app_port: 8000
tags:
  - openenv
  - reinforcement-learning
  - smart-traffic
  - marl
---

# Smart Traffic MARL Environment

## Motivation and Description
Smart Traffic is an 81-agent Multi-Agent Reinforcement Learning (MARL) environment set on a 9x9 grid representation of a city center. Optimizing global traffic flow is incredibly sensitive to hyper-local phase choices, leading to cascading congestion from even a single poorly-timed intersection. We motivate this environment as a highly non-stationary stress test for coordination, generalization, and failure-recovery algorithms under complex dynamics.

## Action Space
Each of the 81 agents (representing an intersection) submits a single integer `[0-4]`.
This is a standard global `MultiDiscrete([5]*81)` space.

* `0`: NS_STRAIGHT_GREEN
* `1`: EW_STRAIGHT_GREEN
* `2`: PROTECTED_NS_LEFT
* `3`: PROTECTED_EW_LEFT
* `4`: ALL_RED_HOLD

*Note:* Dynamic yellow transitions are automatically enforced in the environment wrapper to ensure safety. Switches to distinct green phases automatically inject a 3-step yellow/red phase internally, meaning the action space is safe.

## Observation Space
The environment returns a complex structured state for each agent. The unrolled flat Gym observation space per agent is size `47`.
- **Queues (12)**: Queue fullness `0-1` mapped onto all 12 inbound lanes.
- **Wait Times (12)**: Avg wait time per lane up to 120 secs mapped to `0-1`.
- **Current Phase (5)**: One-hot encoded signal state.
- **Phase Elapsed (1)**: Seconds since last switch normalized mapped `0-1`.
- **Yellow Flag (1)**: Boolean state `[0,1]` representing intermediate switch.
- **Neighbor Queues (8)**: Neighbor intersections `Mean Queue` + `Max Queue` per direction.
- **Congestion Index (4)**: `[N, S, E, W]` directional congestion tracking.
- **Flags (4)**: Dynamic scenario indicators (`Rush Hour Active`, `Emergency Pass`, `Adverse Weather`, `Block Active`).

## Task Scenarios
The environment includes 14 stress-test sub-scenarios:
1. **Foundation (Baseline)** - Base deterministic network flow.
2. **Rush Hour Wave (Easy)** - Heavy spawn skew progressing from one boundary.
3. **Directional Imbalance (Easy)** - 90/10 skew E/W. 
4. **Adaptive Demand (Medium)** - 24hr realistic circadian rhythms. 
5. **Vehicle Mix (Medium)** - Heterogenous vehicle types (cars vs bikes vs 3x-clear-time trucks).
6. **Pedestrian Surge (Medium)** - Random intersections forced to 30s All-Red. 
7. **Emergency Priority (Hard)** - Mandatory clearing paths for ambulance routes. 
8. **Event Spike (Hard)** - 15,000+ car burst spawn from a central 'stadium'.
9. **Road Block (Hard)** - Dynamic edge invalidation + A* rerouting.
10. **Network Partition (Extreme)** - Complete horizontal/vertical bisect logic of city network.
11. **Cascading Failure (Extreme)** - Singular node overload spilling exponentially outwards.

*(Includes complex aggregate incidents like Multi-Incident combination logic).*

## Setup and Usage
The environment acts out of the box as a fully compliant `openenv-core` HTTP Fast API endpoint. Both standard environments (`inference.py`) and standard multi-agent Gymnasium adapters are included out of the box.

### Container (HF Space/Local)
The easiest way to execute is via Docker:
```bash
docker build -t smart-traffic-env .
docker run -p 8000:8000 smart-traffic-env
```

### Direct Python Usage (Gym Adapter)
```python
from smart_traffic.training import TrafficGymAdapter
import numpy as np

# Instantiate standard wrapper hitting your local/HF URL
env = TrafficGymAdapter(server_url="http://localhost:8000", scenario="rush_hour")
obs, info = env.reset()

actions = np.random.randint(0, 5, size=81)
obs, reward, done, trunc, info = env.step(actions)
```

## Baselines
We evaluate using a GPT-4o-mini structured-output baseline script `inference.py` over 100 steps.

| Scenario | Average Step Reward | Note |
|----|----|----|
| Baseline (none) | +1.2 | Normal flow handled easily |
| Rush Hour | -3.42 | Edges heavily saturated |
| Emergency Priority | -6.11 | Ambulance pathing forces sub-optimal global flow |
| Cascading Failure | -12.45 | Radial congestion overflow tests recovery limit |

*You can evaluate all scenarios reproducibly via the provided `inference.py` script.*
