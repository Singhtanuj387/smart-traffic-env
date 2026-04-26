---
title: Smart Traffic MARL
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
  - llm-agents
---

# 🚦 Smart Traffic MARL: The Next-Gen City Grid 🚦

Welcome to the **Smart Traffic** environment! This isn't just a simple reinforcement learning grid—it is a cutting-edge, 81-Agent **Sequential Multi-Agent Decision Process (SMDP)** designed to strictly evaluate how AI coordinates massive city infrastructure under extreme stress. 

By utilizing the standard **OpenEnv** architecture, this project allows you to hook up anything from standard PPO matrices to massive Large Language Models (LLMs) to control the flow of a virtual city!

---

## 🧠 The Deep Dive: How it Works (*Ek Ek Chij!*)

### 1. The 9x9 City Grid (81 Agents)
The environment simulates a dense urban center with 81 distinct intersections. Each intersection is its own independent "Agent". Global traffic flow is incredibly sensitive to hyper-local phase choices, meaning one bad traffic light can cause a catastrophic rippling wave of congestion across the entire city.

### 2. Sequential "Green Wave" Action Space
We completely rebuilt the architecture to evaluate actions **sequentially**. Instead of all 81 lights firing blindly at the same time, the environment pauses physics and asks Agent #0, then Agent #1, all the way up to Agent #80. 
This allows the AI to develop highly complex "Green Wave" algorithms where intersections react to what the previous intersection *just* decided, rather than guessing!

Each agent submits a single integer `[0-4]`:
* `0`: **NS_STRAIGHT** (North-South Green)
* `1`: **EW_STRAIGHT** (East-West Green)
* `2`: **NS_LEFT** (Protected North-South Left Turn)
* `3`: **EW_LEFT** (Protected East-West Left Turn)
* `4`: **ALL_RED** (Stop all traffic / Hold)

### 3. The 67-Dimensional State Space
To solve the "non-stationarity" problem (where agents don't know what their neighbors are doing), we inject a massive 67-dimensional observation tensor into the brain of every agent before they act:
- **Queues (12)**: How full the 12 inbound lanes are `(0.0 - 1.0)`.
- **Wait Times (12)**: Average wait time per lane, mapped mathematically to punish starvation.
- **Neighbor Phases (20)**: One-hot encoded data showing *exactly* what the 4 closest neighbor intersections are currently doing!
- **Current Phase (5)**: What phase is currently active.
- **Phase Elapsed (1)**: How long this light has been green.
- **Yellow Flag (1)**: Safety transition indicator.
- **Neighbor Queues (8)**: Mean and Max queues of surrounding nodes.
- **Congestion Index (4)**: `[N, S, E, W]` directional volume tracking.
- **Flags (4)**: Dynamic scenario indicators (`Emergency Pass`, `Adverse Weather`, etc.).

---

## 🌪️ The Stress Tests (14 Task Scenarios)
The environment includes brutally hard stress-test scenarios. Can your LLM save the city?
1. **Foundation (Baseline)** - Base deterministic network flow.
2. **Rush Hour Wave (Easy)** - Heavy spawn skew progressing from one boundary.
3. **Directional Imbalance (Easy)** - 90/10 skew E/W. 
4. **Adaptive Demand (Medium)** - 24hr realistic circadian rhythms. 
5. **Vehicle Mix (Medium)** - Heterogenous vehicle types (cars vs bikes vs heavy trucks).
6. **Pedestrian Surge (Medium)** - Random intersections forced to 30s All-Red for safety. 
7. **Emergency Priority (Hard)** - Mandatory clearing paths for ambulance routes. 
8. **Event Spike (Hard)** - 15,000+ car burst spawn from a central 'stadium'.
9. **Road Block (Hard)** - Dynamic edge invalidation + A* rerouting.
10. **Network Partition (Extreme)** - Complete horizontal/vertical bisect logic of city network.
11. **Cascading Failure (Extreme)** - Singular node overload spilling exponentially outwards.

## 📊 Performance Plots

The following plots demonstrate the learning convergence and performance of our models across the various stress-test scenarios. All multiple-run comparisons (e.g., Baseline vs. Trained vs. Ablations) are plotted on the same axes to make the performance delta obvious.

![Baseline vs PPO Trained Reward Curve](plots/baseline_vs_trained.png)
*Figure 1: Average step reward over 1,000 episodes comparing the Baseline heuristic against our PPO-trained Agent (higher is better).*

![Ablation Study: With vs Without Neighbor Phase Data](plots/ablation_neighbor_phases.png)
*Figure 2: Ablation study showing the massive convergence speedup gained when injecting the 67-dimensional neighbor-phase data into the observation space.*

![Global Queue Wait Times by Scenario](plots/queue_wait_times.png)
*Figure 3: Total intersection wait times across the 'Rush Hour' and 'Event Spike' scenarios, proving the Trained Agent prevents cascading failures.*

---

## 🚀 Training the Brains

We provide two distinct, state-of-the-art training pipelines out of the box:

### 1. MAPPO (Multi-Agent PPO)
A highly optimized, standard PyTorch implementation of MAPPO (`train.py`). It calculates the 81 sequential sub-steps natively and maintains shared critic networks to maximize global reward.

### 2. The LLM PPO Pipeline (`train_llm.py`) 🔥
**This is the coolest part of the project.**
We have integrated a full Large Language Model (LLM) Reinforcement Learning loop! 
- Powered by **Hugging Face TRL** and **PEFT (LoRA)**.
- Wraps powerful foundational models like `Qwen 1.5B` or `Llama-3.2-1B` in a PPO Value Head.
- Evaluates the 67-dimensional state as a semantic text string!
- **Hardware Optimized**: Uses `bitsandbytes` 4-bit quantization, `batch_size=1`, and adapter-only referencing to allow you to train massive 151k-vocab LLMs entirely on a free Google Colab T4 GPU!

*(See `colab_training_guide.md` for the exact cell-by-cell copy-paste guide to train this on the cloud for free!)*

---

## 🛠️ Setup and Usage

This project strictly adheres to the `openenv-core` HTTP Fast API architecture.

### Running the Server Locally
```bash
# Install dependencies
pip install openenv-core[core] fastapi uvicorn

# Spin up the environment server
uvicorn smart_traffic.server.app:app --host 0.0.0.0 --port 8000
```

### Docker Deployment
The easiest way to execute is via Docker:
```bash
docker build -t smart-traffic-env .
docker run -p 8000:8000 smart-traffic-env
```

### Gym Adapter Usage
```python
from smart_traffic.training import TrafficGymAdapter

# Instantiate standard wrapper hitting your local/HF URL
env = TrafficGymAdapter(server_url="http://localhost:8000", scenario="rush_hour")

# The environment handles the 81 sub-steps inherently under the hood!
obs, info = env.reset()
```

Enjoy building the future of autonomous infrastructure! 🏎️💨
