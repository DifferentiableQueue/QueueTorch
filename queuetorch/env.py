from typing import NamedTuple
import numpy as np
import torch
import torch.nn.functional as F

class Obs(NamedTuple):
    queues: torch.Tensor
    time: torch.Tensor

class EnvState(NamedTuple):
    queues: torch.Tensor
    time: torch.Tensor
    service_times: torch.Tensor
    arrival_times: torch.Tensor
            
class QueuingNetwork():
    def __init__(self, 
                 network:torch.Tensor, 
                 mu:torch.Tensor, 
                 h:torch.Tensor,
                 arrival_rates,
                 inter_arrival_dists, 
                 service_dists, 
                 queue_event_options = None,
                 batch:int = 1, 
                 temp:float = 1,
                 seed:int = 42,
                 device:torch.device = torch.device('cpu'), 
                 buffer_control:bool = False,
                 b:torch.Tensor = None):
        
        '''
        Initializes the discrete event system.
        
        Args:
        network: Tensor representing the network structure.
        mu: Tensor representing the service rates for the servers.
        h: Tensor used for computing costs.
        arrival_rates: Function to model arrival rates.
        inter_arrival_dists: Function to model inter-arrival distributions.
        service_dists: Function to model service distributions.
        queue_event_options: Optional tensor for custom queue events.
        batch: Batch size for simulation.
        temp: Temperature parameter for softmax.
        seed: Random seed for reproducibility.
        device: Device to perform computations (CPU or GPU).
        buffer_control: Flag to control if queue has buffers.
        b: Costs of buffer overflow for each queue.
        '''
         
        self.device = device
        self.network = network.repeat(batch,1,1).to(self.device)
        self.mu = mu.repeat(batch,1,1).to(self.device)
        self.arrival_rates = arrival_rates
        self.inter_arrival_dists = inter_arrival_dists
        self.service_dists = service_dists
        self.q = self.network.size()[2]
        self.s = self.network.size()[1]
        self.h = h.float().to(device)
        self.temp = temp
        self.buffer_control = buffer_control

        if self.buffer_control:
            self.b = b
        
        default_event_mat = torch.cat((F.one_hot(torch.arange(0,self.q)), -F.one_hot(torch.arange(0,self.q)))).float().to(self.device)
        if queue_event_options is None:
            self.queue_event_options = default_event_mat
        else:
            self.queue_event_options = queue_event_options.float().to(self.device)

        self.temp = temp
        self.eps = 1e-8
        self.inv_eps = 1/self.eps
        self.batch = batch
        self.state = np.random.default_rng(seed)

        self.free_servers = torch.ones((self.batch, self.s)).to(self.device)

    def draw_service(self, time):
        '''
        Draws random service times from the service distribution.

        Args:
        time: Current time in the simulation.

        Returns:
        A tensor representing service times for each batch and server.
        '''
        service_mat = torch.tensor(self.service_dists(self.state, self.batch, time)).float().to(self.device)
        return service_mat

    def draw_inter_arrivals(self, time):
        '''
        Draws random inter-arrival times from the inter-arrival distribution.

        Args:
        time: Current time in the simulation.

        Returns:
        A tensor representing inter-arrival times for each batch and queue.
        '''
        interarrivals = torch.tensor(self.inter_arrival_dists(self.state, self.batch)).float().to(self.device)
        lam = torch.tensor(self.arrival_rates(self.state, time, self.batch)).to(self.device)
        return interarrivals / lam
        
    def reset(self, 
              init_queues:torch.Tensor = None,
              init_time:torch.Tensor = None,
              seed:int = None,
              buffer:torch.Tensor = None):
        
        '''
        Resets the system's state to initial conditions.

        Args:
        init_queues: Initial state of the queues.
        init_time: Initial time to start the simulation.
        seed: Random seed to ensure reproducibility.
        buffer: Initial buffer state if buffer control is enabled.

        Returns:
        Initial observations and environment state.
        '''

        cost = torch.tensor([0.]).repeat(self.batch).to(self.device)
        
        if init_time is None:
            time = torch.tensor([[0.]]).repeat(self.batch, 1).to(self.device)
        else:
            time = init_time.repeat(self.batch, 1).to(self.device)

        if init_queues is None:
            queues = torch.tensor([[0.]*self.q]).repeat(self.batch, 1).to(self.device)
        elif init_queues.size()[0] == 1:
            queues = init_queues.float().repeat(self.batch, 1).to(self.device)
            if self.buffer_control:
                queues = torch.min(torch.stack((queues, buffer), dim = 2), dim = 2).values
        else:
            queues = init_queues.float().to(self.device)
            if self.buffer_control:
                queues = torch.min(torch.stack((queues, buffer), dim = 2), dim = 2).values
        
        if seed is not None:
            self.state = np.random.default_rng(seed)

        #service_times = self.inv_eps * torch.ones(self.draw_service(time).size()).to(self.device)
        arrival_times = self.draw_inter_arrivals(time)
        service_times = self.draw_service(time)

        return Obs(queues, time), EnvState(queues, time, service_times, arrival_times)

    def step(self, 
             state:EnvState, 
             action:torch.Tensor,
             buffer:torch.Tensor = None):
        
        '''
        Performs a simulation step by processing the current action and updating the state.

        Args:
        state: The current state of the system (queues, time, service, and arrival times).
        action: The action taken by the agent (queue routing decision).
        buffer: Optional buffer tensor for buffer control.

        Returns:
        Updated observations, next environment state, costs, and event time.
        '''

        #unpack state
        queues, time, service_times, arrival_times = state
        
        # Compliance with network
        action = action * self.network

        # action is zero if queues are zero
        #action = torch.minimum(action, queues.unsqueeze(1).repeat(1,self.s,1))
        action = torch.min(torch.stack((action, queues.unsqueeze(1).repeat(1,self.s,1)), dim = 3), dim = 3).values
        
        # effective service times are service_time divided by mu
        service_mat = service_times.unsqueeze(1).expand(-1, self.s, -1)
        #eff_service_times = torch.minimum(service_mat / torch.clamp(action * self.mu, min = self.eps), self.inv_eps * torch.ones(service_mat.size()).to(self.device))
        eff_service_times = torch.min(torch.stack((service_mat / torch.clamp(action * self.mu, min = self.eps), self.inv_eps * torch.ones(service_mat.size()).to(self.device)), dim = 3), dim = 3).values
        
        # for each queue, get next service time (B x q)
        min_eff_service_times = torch.min(eff_service_times, dim = 1).values
        
        # arrival times and service times are both q vectors
        event_times = torch.cat((arrival_times, min_eff_service_times), dim = 1).float()
        
        # outcome is one_hot argmin of the event times
        outcome = F.one_hot(torch.argmin(event_times, dim = 1), num_classes = event_times.size()[1]) - F.softmax(-event_times/self.temp, dim = -1).detach() + F.softmax(-event_times/self.temp, dim = -1)
        
        # Determine event
        delta_q = torch.matmul(outcome, self.queue_event_options)
        # if self.straight_through_min:
        #     event_time = torch.sum(event_times * outcome, dim = 1)[:,None]
        # else:
        event_time = torch.min(event_times, dim = 1).values[:,None]
        
        # update time elapsed, cost, queues
        time = time + event_time
        cost = torch.matmul(event_time * queues, self.h).unsqueeze(1)

        if self.buffer_control:
            next_queues = F.relu(queues + delta_q)
            overflow = F.relu(next_queues - buffer)
            queues = next_queues - overflow
            buffer_cost = torch.matmul(overflow, self.b).unsqueeze(1)
        else:
            queues = F.relu(queues + delta_q)

        # Reduce non-focal times
        service_times = service_times - torch.sum(action * self.mu, 1) * event_time
        arrival_times = arrival_times - event_time

        new_arrival_times = self.draw_inter_arrivals(time)
        arrival_times = arrival_times + (new_arrival_times) * outcome[:,:self.q].detach()

        with torch.no_grad():
            # Reset service
            new_service_times = self.draw_service(time)
            service_times = service_times + (new_service_times) * outcome[:,self.q:].detach()

        next_state = EnvState(queues, time, service_times, arrival_times)
        obs = Obs(queues, time)

        if self.buffer_control:
            return obs, next_state, cost, buffer_cost, event_time
        else:
            return obs, next_state, cost, event_time
    
        
    def print_state(self, state:EnvState):
        '''
        Prints the current state of the system for debugging purposes.
        
        Args:
        state: The current state of the system.
        '''
        print(f"Total Cost:\t{state.cost}")
        print(f"Time Elapsed:\t{state.time}")
        print(f"Queue Len:\t{state.queues}")
        print(f"Service times:\t{state.service_times}")
        print(f"Arrival times:\t{state.arrival_times}")


def load_env(env_config, temp, batch, seed, device):

    name = env_config['name']

    if 'env_type' in env_config:
        env_type = env_config['env_type']
    else:
        env_type = name

    ## Environment Parameters
    # load network
    if env_config['network'] is None:
        network = np.load(f'./env_data/{env_type}/{env_type}_network.npy')
    else:
        network = env_config['network']

    if 'scale_factor' in env_config:
        scale_factor = env_config['scale_factor']
    else:
        scale_factor = 1

    # load mu
    if env_config['mu'] is None:
        mu = np.load(f'./env_data/{env_type}/{env_type}_mu.npy')
    else:
        mu = env_config['mu']

    network = torch.tensor(network).float()
    mu = torch.tensor(mu).float()

    orig_s, orig_q = network.size()

    # repeat if server pools
    num_pool = env_config['num_pool']
    network = network.repeat_interleave(num_pool, dim = 0)
    mu = mu.repeat_interleave(num_pool, dim = 0)

    queue_event_options = env_config['queue_event_options']
    if queue_event_options is not None:
        if queue_event_options == 'custom':
            queue_event_options = torch.tensor(np.load(f'./env_data/{env_type}/{env_type}_delta.npy'))
        else:
            queue_event_options = torch.tensor(queue_event_options)

    h = torch.tensor(env_config['h'])

    # arrival and service rates
    lam_type = env_config['lam_type']
    lam_params = env_config['lam_params']

    if lam_type == 'constant':
        if lam_params['val'] is None:
            lam_r = np.load(f'./env_data/{env_type}/{env_type}_lam.npy')
        else:
            lam_r = lam_params['val']

        def lam(rng, t, batch, lam_r):
            return lam_r
        arrival_rates = lambda rng, t, batch: lam(rng, t, batch, lam_r = lam_r)
    elif lam_type == 'step':
        def lam(rng, t, step, p, scale, val1, val2):
            if not rng:
                is_surge = 1*(t <= step)
                init_lam = is_surge * np.array(val1) + (1 - is_surge) * np.array(val2)
                return init_lam
            else:
                is_surge = 1*(t.detach().cpu().numpy() <= step)
                init_lam = is_surge * np.array(val1) + (1 - is_surge) * np.array(val2)
                switch = rng.binomial(1, p)
                return switch * (init_lam / (1 + scale)) + (1 - switch) * (init_lam / (1 - scale))
        arrival_rates = lambda rng, t, batch: lam(rng, t, step = lam_params['t_step'], 
                                                          p = 0.5, scale = lam_params['scale'],
                                                          val1 = lam_params['val1'], val2 = lam_params['val2'])
    elif lam_type == 'sawtooth':
        def lam(rng, t, batch, step, val1, val2):
            is_surge = 1*(np.floor((t.detach().cpu().numpy() / step)) % 2 == 0)
            return is_surge * np.array(val1) + (1 - is_surge) * np.array(val2)
        arrival_rates = lambda rng, t, batch: lam(rng, t, step = lam_params['t_step'], val1 = lam_params['val1'], val2 = lam_params['val2'])
    elif lam_type == 'hyper':
        if lam_params['val'] is None:
            lam_r = np.load(f'./env_data/{env_type}/{env_type}_lam.npy')
        else:
            lam_r = np.array(lam_params['val'])

        scale = lam_params['scale']

        def lam(rng, t, batch, p, lam_r, scale):
            if not rng:
                return lam_r
            else:
                lam_r = lam_r.reshape((1,len(lam_r))).repeat(batch, axis = 0)
                switch = rng.binomial(1, p, (batch, 1))
                return switch * (lam_r / (1 + scale)) + (1 - switch) * (lam_r / (1 - scale))
        arrival_rates = lambda rng, t, batch: lam(rng, t, batch, p = 0.5, lam_r = lam_r, scale = scale)
    else:
        print('Nonvalid arrival rate')
    
    def inter_arrival_dists(state, batch):
        exps = state.exponential(1, (batch, orig_q))
        return exps

    if 'service_type' in env_config:
        service_type = env_config['service_type']
    else:
        service_type = 'exp'

    def service_dists(state, batch, t):
        if service_type == 'exp':
            return state.exponential(1, (batch, orig_q))
        if service_type == 'lognormal':
            return state.lognormal(0, 1, (batch, orig_q)) / np.exp(1/2)
        if service_type == 'hyper':
            scale = 0.8
            coins = state.binomial(1,0.5, size = (batch, orig_q))
            a = state.exponential((1 + scale), (batch, orig_q))
            b = state.exponential((1 - scale), (batch, orig_q))
            return coins * a + (1 - coins) * b
        else:
            pass

    dq = QueuingNetwork(network, mu, h, arrival_rates, 
                        inter_arrival_dists, service_dists, 
                        queue_event_options= queue_event_options,
                        batch = batch, 
                        temp = temp, seed = seed,
                        device = torch.device(device))

    return dq


    


    