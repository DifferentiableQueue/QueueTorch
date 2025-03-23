import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math
import torch.optim as optim
from queuetorch.env import QueuingNetwork
import queuetorch.routing as rt
from tqdm import trange
import torch.distributions.one_hot_categorical as one_hot_sample


def dict_to_list(d):
    d_list = [None]*len(d.keys())
    for key in d.keys():
        d_list[key] = d[key]

    dist = np.array(d_list)
    dist = dist/sum(dist)

    return dist

def mms_dist(x, s, rho):
    pi_0 = sum([((s*rho)**q)/math.factorial(q) for q in range(s)]) + (((s*rho)**(s))/math.factorial(s))*(1/(1-rho))
    pi_0 = 1/pi_0
    if x < s:
        return pi_0 * ((s*rho)**x)/(math.factorial(x))
    else:
        return pi_0 * (s**(s-x))*((s*rho)**x)/(math.factorial(s))
    
def ErlangC(s,rho):
    denom = 1 + (1-rho) * (math.factorial(s) / ((s * rho)**(s))) * sum([((s*rho)**(k))/math.factorial(k) for k in range(s)])
    return 1 / denom


def mms(s, true_rho):
    
    # Network Parameters
    rho = true_rho * s

    arrival_rates = lambda rng, t, batch: rho
    inter_arrival_dists = lambda state, batch: state.exponential(1, (batch, 1))
    service_dists = lambda state, batch, t: state.exponential(1, (batch, 1))

    network = torch.tensor([[1.]]*s)
    mu = torch.tensor([[1.0]]*s)
    h = torch.tensor([1.])
    batch = 1000

    seed = 200003
    torch.manual_seed(seed)
    
    dq = QueuingNetwork(network, mu, h, arrival_rates, inter_arrival_dists, service_dists, batch = batch, temp = 0.5)

    # Initialize
    obs, state = dq.reset(seed = 43)
    # values = torch.tensor([[[i + 1 for i in range(s)]]]*batch)
    values = torch.tensor([[i + 1 for i in range(s)]]).T.unsqueeze(0).repeat(batch, 1, 1)
    total_cost = torch.tensor([[0.]]*batch)
    
    # Obtain Steady State
    for _ in trange(200000):
        # state info and action
        action = F.one_hot(torch.argmax(values, dim = 2), num_classes = dq.q)
        
        # step
        obs, state, cost, event_time = dq.step(state, action)
        total_cost += cost

    # Compare Avg Queue-Length
    avg_cost = torch.mean(total_cost / state.time).detach().numpy()
    mms_queue_len = (true_rho / (1 - true_rho)) * ErlangC(s,true_rho) + s * true_rho

    print(avg_cost)
    
    assert np.abs(avg_cost - mms_queue_len) <= 0.2, f"Simulated cost: {avg_cost}, True cost: {mms_queue_len}"


def priority_queue(rho1, rho2):

    arrival_rates = lambda rng, t, batch: np.array([rho1, rho2])
    inter_arrival_dists = lambda state, batch: state.exponential(1, (batch, 2))
    service_dists = lambda state, batch, t: state.exponential(1, (batch, 2))

    network = torch.tensor([[1., 1.]])
    mu = torch.tensor([[1., 1.]])
    h = torch.tensor([1., 1.])
    priorities = torch.tensor([2., 1.])
    batch = 1000

    seed = 17
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    dq = QueuingNetwork(network, mu, h, arrival_rates, inter_arrival_dists, service_dists, batch = batch, temp = 0.5)

    obs, state = dq.reset(seed = 17)

    time_avg_queue_lens = torch.zeros(obs.queues.size())

    for _ in trange(200000):
        # state info and action
        queues, time = obs

        pr = F.one_hot(torch.argmax(priorities * 1.*(queues > 0.).unsqueeze(1), dim = 2), num_classes = dq.q)
        pr = torch.minimum((pr * dq.network), queues.unsqueeze(1).repeat(1, dq.s, 1))
        pr += 1*torch.all(pr == 0., dim = 2).reshape(dq.batch,dq.s,1) * dq.network
        pr /= torch.sum(pr, dim = -1).reshape(dq.batch, dq.s, 1)
        
        action = one_hot_sample.OneHotCategorical(probs = pr).sample()
        
        # step
        obs, state, cost, event_time = dq.step(state, action)
        time_avg_queue_lens += event_time * queues

    # Compare Avg Queue-Length
    time_avg_queue_lens = torch.mean(time_avg_queue_lens / state.time, dim = 0)
    time_avg_queue_lens = time_avg_queue_lens.detach().numpy()
    print(time_avg_queue_lens)

    high_priority_mean = (rho1 / (1 - rho1))
    low_priority_mean = (rho2 / (1 - rho1 - rho2)) * ((rho1 / (1 - rho1)) + 1)

    assert np.abs(time_avg_queue_lens[0] - high_priority_mean) <= 0.1, f"Simulated cost: {time_avg_queue_lens[0]}, True cost: {high_priority_mean}"
    assert np.abs(time_avg_queue_lens[1] - low_priority_mean) <= 0.1, f"Simulated cost: {time_avg_queue_lens[1]}, True cost: {low_priority_mean}"


def tandem_queue(rho, fig_dir):
    s = 2
    arrival_rate = rho

    # inter_arrival_dists = [exp_arrival, no_arrival]
    # service_dists = [exp_service, exp_service]
    arrival_rates = lambda rng, t: np.array([rho, 0.0000001])
    inter_arrival_dists = lambda state, batch: state.exponential(1, (batch, 2))
    service_dists = lambda state, batch, t: state.exponential(1, (batch, 2))

    network = torch.tensor([[1., 0.],
                            [0., 1.]])

    mu = torch.tensor([[1., 0.],
                        [0., 1.]])

    queue_event_options = torch.tensor([[1., 0.],
                                        [0., 0.],
                                        [-1., 1.],
                                        [0., -1.]]).float()

    h = torch.tensor([1., 1.])
    batch = 1

    # eval max weight
    dq = QueuingNetwork(network, mu, h, arrival_rates, inter_arrival_dists, service_dists, queue_event_options = queue_event_options, 
                                     batch = batch, temp = 0.5)
    
    values = torch.tensor([[[1., 0.],
                           [0., 1.]]])
    
    obs, state = dq.reset()
    marg_queue_len_dist = [{} for _ in range(dq.q)]

    for _ in trange(20000):
        # state info and action
        queues, time = obs
        v, s_bar, q_bar = rt.pad(values, queues.detach(), network = dq.network)
        action = rt.linear_assignment_batch(v.detach(), s_bar, q_bar)
        
        # step
        obs, state, cost, event_time = dq.step(state, action)

        # update state distributions
        # state_record = queues.data[0].numpy().astype("int")
        
        # # update marginal state dist:
        # for qu, qu_len in enumerate(state_record):
        #     if qu_len in marg_queue_len_dist[qu].keys():
        #         marg_queue_len_dist[qu][int(qu_len)] += float(event_time.data.numpy())
        #     else:
        #         marg_queue_len_dist[qu][int(qu_len)] = float(event_time.data.numpy())

    # stat_dist_list = dict_to_list(marg_queue_len_dist[0])

    # vals = [i for i in range(len(stat_dist_list))]

    # plt.plot(vals, stat_dist_list, label = "stationary")
    # plt.plot(vals, [mms_dist(val, s = 1, rho = rho) for val in vals], label = f"M/M/S ({s}, {rho:0.2f})")
    # plt.legend()
    # plt.savefig(fig_dir + "_tandem_0.png")
    # plt.close()

    # stat_dist_list = dict_to_list(marg_queue_len_dist[1])

    # vals = [i for i in range(len(stat_dist_list))]

    # plt.plot(vals, stat_dist_list, label = "stationary")
    # plt.plot(vals, [mms_dist(val, s = 1, rho = rho) for val in vals], label = f"M/M/S ({s}, {rho:0.2f})")
    # plt.legend()
    # plt.savefig(fig_dir + "_tandem_1.png")
    # plt.close()


if __name__ == '__main__':

    ## multiserver queue
    mms(1, 0.9)
    mms(2, 0.9)
    priority_queue(0.5, 0.4)
    