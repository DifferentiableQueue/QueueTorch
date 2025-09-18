import numpy as np
import tqdm
from tqdm import trange
import torch
from torch import nn
import torch.nn.functional as F
import argparse
import math
import os
import torch.optim as optim
from queuetorch.env import QueuingNetwork
import queuetorch.env as env
import matplotlib.pyplot as plt
import json
import torch.distributions.one_hot_categorical as one_hot_sample
import yaml

def plot_policy_switching_curve(net, 
                                fig_dir = None, 
                                device = "cpu", 
                                base_level = 5, 
                                q = 2, 
                                max_queue = 50, 
                                inds = (0,1),
                                val_inds = (0,0)):
    
    X = np.arange(0, max_queue, 1)
    Y = np.arange(0, max_queue, 1)
    Z = np.zeros((max_queue,max_queue))

    for i in range(max_queue):
        for j in range(max_queue):
            obs = torch.tensor([base_level]*q)
            obs[inds[0]] = X[i]
            obs[inds[1]] = Y[j]

            obs = obs.float().unsqueeze(0).to(device)
            Z[i][j] = net(obs)[0][val_inds[0]][val_inds[0]]

    plt.imshow(Z, interpolation='nearest', origin='lower')
    if fig_dir is None:
        plt.show()
        plt.close()
    else:
        plt.savefig(fig_dir)
        plt.close()
    
class PriorityNet(nn.Module):
    def __init__(self, s, q, layers, hidden_dim, f_time = False, x_stats = None, t_stats = None):
        super().__init__()
        self.s = s
        self.q = q
        self.x_stats = x_stats
        self.t_stats = t_stats
        self.layers = layers
        self.hidden_dim = hidden_dim
        
        self.f_time = f_time
        
        if self.f_time:
            self.input_fc = nn.Linear(self.q + 1, hidden_dim)
        else:
            self.input_fc = nn.Linear(self.q, hidden_dim)
            
        self.layers_fc = nn.ModuleList()
        for _ in range(layers):
            self.layers_fc.append(nn.Linear(hidden_dim, hidden_dim))
        
        self.output_fc = nn.Linear(hidden_dim, self.s * self.q)
        #self.output_fc = nn.Linear(hidden_dim, self.q * self.s)
        
    def forward(self, x, t = 0):
        
        # Input layer
        batch = x.size()[0]
        
        if self.x_stats is not None:
            x = (x - self.x_stats[0]) / self.x_stats[1]

        if self.t_stats is not None:    
            t = (t - self.t_stats[0]) / self.t_stats[1]
        
        if self.f_time:
            x = torch.cat((x, t), 1)
            
        x = F.relu(self.input_fc(x))

        # Hidden layer
        for l in range(self.layers):
            x = F.relu(self.layers_fc[l](x))

        # Output layer
        x = self.output_fc(x)
        return F.softmax(torch.reshape(x, (batch, self.s , self.q)), dim = 2)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', type=str)
    parser.add_argument('-m', type=str)
    parser.add_argument('-d', type=str, default = None)

    args = parser.parse_args()

    # load config
    with open(f'./configs/env/{args.e}', 'r') as f:
        env_config = yaml.safe_load(f)

    with open(f'./configs/model/{args.m}', 'r') as f:
        model_config = yaml.safe_load(f)
    
    name = env_config['name']
    print(f'env: {name}')

    model_name = model_config['name']
    print(f'model: {model_name}')

    # Set seed
    if model_config['env']['model_seed'] is not None:
        torch.manual_seed(model_config['env']['model_seed'])

    # repeat if server pools
    if 'env_type' in env_config:
        env_type = env_config['env_type']
    else:
        env_type = name
    if env_config['network'] is None:
        network = np.load(f'./env_data/{env_type}/{env_type}_network.npy')
    else:
        network = env_config['network']
    network = torch.tensor(network).float()
    orig_s, orig_q = network.size()

    init_test_queues = torch.tensor([env_config['init_queues']]).float()
    init_train_queues = torch.tensor([env_config['init_queues']]).float()

    train_T = env_config['train_T']
    test_T = env_config['test_T']
    
    ## Model parameters
    model_name = model_config['name']
    print(f'model: {model_name}')

    checkpoint = model_config['checkpoint']
    checkpoint_dir = f"./models/{name}/{model_name}"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    num_epochs = model_config['opt']['num_epochs']
    train_batch = model_config['opt']['train_batch']
    test_batch = model_config['opt']['test_batch']
    lr = model_config['opt']['lr']
    betas = model_config['opt']['betas']
    threshold = model_config['opt']['threshold']
    
    if args.d is None:
        device = model_config['env']['device']
    else:
        device = args.d
    
    test_seed = model_config['env']['test_seed']
    train_seed = model_config['env']['train_seed']
    env_temp = model_config['env']['env_temp']
    test_freq = model_config['env']['test_freq']
    straight_through_min = model_config['env']['straight_through_min']

    layers = model_config['param']['layers']
    width = model_config['param']['width']
    f_time = model_config['param']['f_time']

    test_policy = model_config['policy']['test_policy']
    train_policy = model_config['policy']['train_policy']
    randomize = model_config['policy']['randomize']

    if checkpoint is None:
        net = PriorityNet(orig_s, orig_q, layers, width, f_time = f_time).to(device)
    else:
        net = torch.load(f'{checkpoint_dir}_{checkpoint}.pt')
    optimizer = optim.Adam(net.parameters(), lr = lr, betas = betas)

    test_loss = []

    ## Train Loop
    
    for epoch in range(num_epochs):
        
        if f_time:
            if epoch == 0:
                net.t_stats = [train_T / 2, train_T / 4]
            elif epoch == 1:
                net.t_stats = [torch.mean(state.time.detach()) / 2, torch.mean(state.time.detach()) / 4]
                # net.x_stats = [torch.mean(torch.stack(queues_path, 0), 0), torch.std(torch.stack(queues_path, 0), 0)]
            else:
                pass
        
        if epoch % test_freq == 0:
            ## _________________________ Test _________________________
            dq = env.load_env(env_config, temp = env_temp, batch = test_batch, seed = test_seed, device = device)
            
            obs, state = dq.reset(seed = test_seed, init_queues = init_test_queues)
            total_cost = torch.tensor([[0.]]*test_batch).to(device)
            time_weight_queue_len = torch.tensor([[0.]]*test_batch).to(device)

            all_action_q_checks = []
            all_action_s_checks = []
            
            with torch.no_grad():
                for _ in trange(test_T):

                    queues, time = obs
                    pr = net(queues, time)

                    # test policy
                    if test_policy == 'softmax':
                        pr = pr * dq.network
                        pr = torch.minimum(pr, queues.unsqueeze(1).repeat(1, dq.s, 1))
                        pr += 1*torch.all(pr == 0., dim = 2).reshape(test_batch,dq.s,1).repeat(1,1,dq.q) * dq.network
                        pr /= torch.sum(pr, dim = -1).reshape(test_batch, dq.s, 1)
                    elif test_policy == 'argmax':
                        pr = pr * dq.network
                        pr = torch.minimum(pr, queues.unsqueeze(1).repeat(1, dq.s, 1))
                        pr = F.one_hot(torch.argmax(pr, dim = 2), num_classes = dq.q).float()
                    else:
                        pass

                    # randomize policy or not
                    if randomize:
                        action = one_hot_sample.OneHotCategorical(probs = pr).sample()
                    else:
                        action = torch.round(pr)

                    obs, state, cost, event_time = dq.step(state, action)
                    total_cost = total_cost + cost
                    time_weight_queue_len = time_weight_queue_len + queues * event_time

                    # verify action
                    eff_action = torch.minimum(action * dq.network, queues.unsqueeze(1).repeat(1,dq.s,1))
                    eff_action_servers = torch.sum(eff_action, 2)
                    eff_action_queues = torch.sum(eff_action, 1)
                    action_s_check = torch.all(eff_action_servers <= torch.ones(eff_action_servers.size()).to(device))
                    action_q_check = torch.all(eff_action_queues <= queues)

                    all_action_s_checks.append(action_s_check)
                    all_action_q_checks.append(action_q_check)

            # Test cost metrics
            all_s_valid = torch.all(torch.stack(all_action_s_checks))
            all_q_valid = torch.all(torch.stack(all_action_q_checks))
            all_valid = torch.all(torch.stack([all_s_valid, all_q_valid]))

            # Test cost metrics
            test_cost = torch.mean(total_cost / state.time)
            test_loss_std = float(torch.std(total_cost / state.time))
            
            print(f"queue lengths: \t{torch.mean(time_weight_queue_len / state.time, dim = 0)}")
            print(f"final cost: \t{torch.mean(torch.matmul(queues, dq.h))}")
            print(f"test cost: \t{test_cost}")
            print(f"all valid: \t{all_valid}")

            if not model_config['env']['test_restart']:
                # for each epoch start where you left off
                init_test_queues = queues.detach()

            if device == 'cpu':
                # Plot
                fig_dir = f'./plot/{name}/{model_name}'
                if not os.path.exists(fig_dir):
                    os.makedirs(fig_dir)
                if model_config['plot']['plot_policy_curve']:
                    plot_policy_switching_curve(net, 
                                                fig_dir = f'{fig_dir}_{epoch}.png', 
                                                device = "cpu", 
                                                base_level = 5, 
                                                q = dq.q, 
                                                max_queue = 50, 
                                                inds = model_config['plot']['inds'],
                                                val_inds = model_config['plot']['val_inds'])

        ## _________________________ Train _________________________
        
        if model_config['env']['train_seed'] is None:
            train_seed = int.from_bytes(os.urandom(4), 'big')

        # When training, 'straight_through_min = True'
        net = net.to(device)

        dq = env.load_env(env_config, temp = env_temp, batch = train_batch, seed = train_seed, device = device)

        # zero out the optimizer
        optimizer.zero_grad()

        # save grads
        back_outs = []
        def action_hook(grad):
            #grad = torch.clamp(grad, -threshold,threshold)
            back_outs.append(grad.tolist())
            #return grad
        
        nn_back_ins = []
        def priority_hook(grad):
            nn_back_ins.append(grad.tolist())

        # Train loop
        obs, state = dq.reset(seed = train_seed, init_queues = init_train_queues)
        total_cost = torch.tensor([[0.]]*train_batch).to(device)
        time_weight_queue_len = torch.tensor([[0.]]*train_batch).to(device)

        print(obs.queues)
        print(net(obs.queues, obs.time))
        
        queues_path = []
        
        for _ in trange(train_T):
            queues, time = obs
            
            pr = net(queues, time.detach())
            pr.register_hook(priority_hook)

            if train_policy == 'softmax':
                pr = pr * dq.network
                pr = torch.minimum(pr, queues.unsqueeze(1).repeat(1, dq.s, 1))
                pr += 1*torch.all(pr == 0., dim = 2).reshape(train_batch,dq.s,1).repeat(1,1,dq.q) * dq.network
                pr /= torch.sum(pr, dim = -1).reshape(train_batch, dq.s, 1) 
                pr.register_hook(priority_hook)
            else:
                pass

            action = pr
            action.register_hook(action_hook)
            obs, state, cost, event_time = dq.step(state, action)
            
            total_cost = total_cost + cost
            time_weight_queue_len = time_weight_queue_len + queues * event_time

            queues_path.append(queues.detach())
            
        # Backward
        loss = torch.mean(total_cost / train_T)
        loss.backward()

        print(f"train cost:\t{torch.mean(total_cost / state.time)}")
        print(f"queue lengths: \t{torch.mean(time_weight_queue_len / state.time, dim = 0)}")

        # Gradient clipping and step
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm = model_config['opt']['grad_clip_norm'])
        optimizer.step()

        if not model_config['env']['train_restart']:
            init_train_queues = queues.detach()

        if model_config['env']['print_grads']:
            print("Action Grads")
            print(torch.mean(torch.sum(torch.tensor(back_outs),0),0))

            print("Priority Grads")
            print(torch.mean(torch.sum(torch.tensor(nn_back_ins),0),0))

        # Save checkpoint
        torch.save(net, checkpoint_dir + f'_{epoch}.pt')

        test_loss.append({'epoch': epoch,
                            'test_loss': float(test_cost),
                            'train_loss': float(torch.mean(total_cost / state.time)),
                            'test_loss_std': test_loss_std})
        
        if not os.path.exists('./loss/'):
            os.makedirs('./loss/')
        with open(f'./loss/{name}_{model_name}.json', 'w') as f:
            json.dump(test_loss, f)

