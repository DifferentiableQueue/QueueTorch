<!-- <img width="350" src="./assets/queue_torch_light2.png" alt="logo" /> -->
<img width="350" src="./assets/queuetorch_light.png" alt="logo" />
<hr/>

[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**QueueTorch** is a PyTorch-based package for differentiable queuing network control. It provides a flexible and efficient framework for modeling, simulating, and optimizing queuing systems where reinforcement learning (RL) policies can be learned via direct backpropagation.

## 🚀 Features

- ✅ Differentiable simulation of queuing networks
- 🔁 Support for reinforcement learning and gradient-based optimization
- ⚙️ Discrete-event simulation with customizable service and arrival processes
- 🚄 GPU-accelerated simulation
- 📦 Built with PyTorch
- 🧠 Designed for research in operations research, RL, and stochastic control

## 📦 Installation

Ensure you have Python and build installed, then install the necessary dependencies:

```
python3 -m build
python3 -m pip install .
```

## 🎬 Quick Start

Simulate an M/M/1 queue.
```
import queuetorch as qt
import queuetorch.env as env

# Setup M/M/1 queue
arrival_rates = lambda rng, t, batch: 0.9
inter_arrival_dists = lambda state, batch: state.exponential(1, (batch, 1))
service_dists = lambda state, batch, t: state.exponential(1, (batch, 1))

network = torch.tensor([[1.]])
mu = torch.tensor([[1.0]])
h = torch.tensor([1.])

dq = env.QueuingNetwork(network, mu, h, arrival_rates, inter_arrival_dists, service_dists, batch = 1, temp = 0.5)

# Initialize environment
obs, state = dq.reset(seed = 42)
total_cost = torch.tensor([[0.]])
    
# Obtain Steady State
for _ in trange(10000):
    # state info and action
    action = torch.tensor([[1.]])
    
    # step
    obs, state, cost, event_time = dq.step(state, action)
    total_cost += cost

steady_state_mean = torch.mean(total_cost / obs.time)
```

See `notebooks/criss_cross.ipynb` to train a neural policy to control the criss-cross network. 

## Policy Training

Environment config files are saved in `/configs/env/` and policy config files (e.g. learning rate, neural net architecture parameters, etc.) are saved in `/configs/model/`. For an environment specified by `ENV.yaml` and model specified by `MODEL.yaml`, train a policy as follows:

```
cd train
python3 ./train_policy.py -e=ENV.yaml -m=MODEL.yaml
```

Model checkpoints are saved in the `model` directory, and if applicable, switching plots are saved
in the `plots` directory.