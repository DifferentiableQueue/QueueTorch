import numpy as np
import scipy.optimize as opt
import scipy.sparse as sparse
import torch
from torch import nn
import torch.nn.functional as F

def match_constraint_mat(num_s, num_q, f_fluid = False):
    N = num_s
    M = num_q
    NZ = 2*N*M
    irow = np.zeros(NZ, dtype=int)
    jcol = np.zeros(NZ, dtype=int)
    value = np.zeros(NZ)
    for i in range(N):
        for j in range(M):
            k = M*i+j
            k1 = 2*k
            k2 = k1+1
            irow[k1] = i
            jcol[k1] = k
            value[k1] = 1.0
            if not f_fluid:
                irow[k2] = N+j
                jcol[k2] = k
                value[k2] = 1.0

    A = sparse.coo_matrix((value, (irow, jcol)))
    
    return A


def linear_assignment(values, servers, jobs):

    s,q = values.shape
    A = match_constraint_mat(s, q).toarray()
    c = np.reshape(-values, s * q)
    b = np.append(servers, jobs)

    res = opt.linprog(c=c,A_ub=A,b_ub=b, method = 'highs-ds')
    X = np.reshape(res.x, (s,q))

    X = np.rint(X)[:s-1,:q-1].tolist()
        
    return torch.tensor(X)


def linear_assignment_batch(values, s_bar, q_bar):

    batch,s,q = values.size()
    action = []

    for b in range(batch):
        v = values[b].numpy()
        servers = s_bar[b].numpy()
        jobs = q_bar[b].numpy()

        A = match_constraint_mat(s, q).toarray()
        c = np.reshape(-v, s * q)
        b = np.append(servers, jobs)

        res = opt.linprog(c=c,A_ub=A,b_ub=b, method = 'highs-ds')
        X = np.reshape(res.x, (s,q))

        X = np.rint(X)[:s-1,:q-1].tolist()
        action.append(X)
    
    return torch.tensor(action)


def pad(vals, queues, network, 
        device = 'cpu', compliance = True):

    # setup mu bar
    batch = network.size()[0]
    s = network.size()[1]
    q = network.size()[2]

    free_servers = torch.ones((batch, s)).to(device)
    
    # pad_q = torch.zeros((batch, 1,q)).to(device)
    # pad_s = torch.zeros((batch, s + 1,1)).to(device)
    pad_q = -torch.ones((batch, 1,q)).to(device)
    pad_s = -torch.ones((batch, s + 1,1)).to(device)

    if compliance:
        vals = vals * network - 1*(network == 0.).to(device)

    v = torch.cat((vals, pad_q), 1)
    v = torch.cat((v, pad_s), 2)

    excess_server = F.relu(s - torch.sum(queues, dim = 1)).unsqueeze(1).to(device)
    q_bar = torch.hstack((queues, excess_server)).to(device)

    excess_queues = F.relu(torch.sum(queues, dim = 1) - s).unsqueeze(1).to(device)
    s_bar = torch.hstack((free_servers, excess_queues)).to(device)

    return v, s_bar, q_bar

        
class Sinkhorn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, c, a, b, num_iter, temp, eps = 1e-6, back_temp = None, device = 'cpu'):
        
        log_p = -c / temp
        
        a_dim = 2
        b_dim = 1

        log_a = torch.log(torch.clamp(a, eps)).unsqueeze(dim=2)
        log_b = torch.log(torch.clamp(b, eps)).unsqueeze(dim=1)

        for _ in range(num_iter):
            log_p -= (torch.logsumexp(log_p, dim=1, keepdim=True) - log_b)
            log_p -= (torch.logsumexp(log_p, dim=2, keepdim=True) - log_a)
        
        p = torch.exp(log_p)
        ctx.save_for_backward(p, torch.sum(p, dim=2), torch.sum(p, dim=1))
        ctx.temp = temp
        ctx.back_temp = back_temp
        ctx.device = device
        
        return p

    @staticmethod
    def backward(ctx, grad_p):
        
        p, a, b = ctx.saved_tensors
        batch, m, n = p.shape

        device = ctx.device
        
        a = torch.clamp(a, 1e-1)
        b = torch.clamp(b, 1e-1)
        
        if ctx.back_temp is not None:
            grad_p *= -1 / ctx.back_temp * p
        else:
            grad_p *= -1 / ctx.temp * p

        K_b = torch.cat((
            torch.cat((torch.diag_embed(a), p), dim=2),
            torch.cat((torch.transpose(p, 1, 2), torch.diag_embed(b)), dim=2)),
            dim = 1)[:,:-1,:-1]
        
        I = torch.eye(K_b.size()[1]).to(device)
        n_batch = torch.tensor([1.0]*batch).to(device)
        batch_eye = torch.einsum('ij,k->kij', I, n_batch)
        
        K_b = K_b + 0.01*batch_eye

        t_b = torch.cat((
            grad_p.sum(dim=2),
            grad_p[:,:,:-1].sum(dim=1)),
            dim = 1).unsqueeze(2)


        grad_ab_b = torch.linalg.solve(K_b, t_b)
        grad_a_b = grad_ab_b[:, :m, :]
        grad_b_b = torch.cat((grad_ab_b[:, m:, :], torch.zeros((batch, 1, 1), dtype=torch.float32).to(device)), dim=1)

        U = grad_a_b + torch.transpose(grad_b_b, 1, 2)

        grad_p -= p * U
        
        return grad_p, None, None, None, None, None, None, None
