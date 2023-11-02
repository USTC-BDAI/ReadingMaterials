# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 2021

@author: Askeladd
"""

import numpy as np
import torch
import torch.nn as nn
import math
from scipy.linalg import lstsq,pinv
from numpy.linalg import matrix_rank
import matplotlib.pyplot as plt
import time
import random

torch.set_default_dtype(torch.float64)

def set_seed(x):
    random.seed(x)
    np.random.seed(x)
    torch.manual_seed(x)
    torch.cuda.manual_seed_all(x)
    torch.backends.cudnn.deterministic = True



def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.uniform_(m.weight, a = -1, b = 1)
        nn.init.uniform_(m.bias, a = -1, b = 1)
        #nn.init.normal_(m.weight, mean=0, std=1)
        #nn.init.normal_(m.bias, mean=0, std=1)


class local_rep(nn.Module):
    def __init__(self, in_features, out_features, hidden_layers, M, R_m, x_max, x_min):
        super(local_rep, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = M
        self.hidden_layers  = hidden_layers
        self.R_m = R_m
        self.M = M
        self.x_min = x_min
        self.x_max = x_max
        self.a = 2.0/(x_max - x_min)
        self.x_0 = (x_max + x_min)/2
        self.hidden_layer = nn.Sequential(nn.Linear(self.in_features, self.hidden_features, bias=True),nn.Tanh())

    def forward(self,x):
        y = self.a * (x - self.x_0)
        y = self.hidden_layer(y)

        return y


AA = 1
aa = 2.0*np.pi
bb = 3.0*np.pi
interval_length = 8.

def anal_u(x):
    return AA * np.sin(bb * (x + 0.05)) * np.cos(aa * (x + 0.05)) + 2.0

def anal_dudx_2nd(x):
    return -AA*(aa*aa+bb*bb)*np.sin(bb*(x+0.05))*np.cos(aa*(x+0.05))\
           -2.0*AA*aa*bb*np.cos(bb*(x+0.05))*np.sin(aa*(x+0.05))

def Lu_f(pointss, lambda_ = 4):
    r = []
    for x in pointss:
        f = anal_dudx_2nd(x) - lambda_*anal_u(x)
        r.append(f)
    return(np.array(r))


def pre_define(N_e,M,Q):
    models = []
    points = []
    for k in range(N_e):
        x_min = 8.0/N_e * k
        x_max = 8.0/N_e * (k+1)
        model = local_rep(in_features = 1, out_features = 1, hidden_layers = 1, M = M, R_m = R_m, x_min = x_min, x_max = x_max)
        model = model.apply(weights_init)
        model = model.double()
        for param in model.parameters():
            param.requires_grad = False
        models.append(model)
        points.append(torch.tensor(np.linspace(x_min, x_max, Q+1),requires_grad=True).reshape([-1,1]))
    return(models,points)


def cal_matrix(models,points,N_e,M,Q):
    # matrix define (Aw=b)
    A_1 = np.zeros([N_e*Q,N_e*M])
    A_2 = np.zeros([2,N_e*M])
    A_c0 = np.zeros((N_e - 1, N_e * M))
    A_c1 = np.zeros((N_e - 1, N_e * M))
    f = np.zeros([N_e*Q + 2 + 2 * (N_e - 1), 1])
    
    for k in range(N_e):
        # forward and grad
        out = models[k](points[k])
        values = out.detach().numpy()
        grads = []
        grads_2 = []
        M_0 = k * M
        M_1 = M_0 + M
        for i in range(M):
            g_1 = torch.autograd.grad(outputs=out[:,i], inputs=points[k],
                                    grad_outputs=torch.ones_like(out[:,i]),
                                    create_graph = True, retain_graph = True)[0]
            grads.append(g_1.squeeze().detach().numpy())
            g_2 = torch.autograd.grad(outputs=g_1[:,0], inputs=points[k],
                                    grad_outputs=torch.ones_like(out[:,i]),
                                    create_graph = False, retain_graph = True)[0]
            grads_2.append(g_2.squeeze().detach().numpy())
        grads = np.array(grads).T
        grads_2 = np.array(grads_2).T
        Lu = grads_2 - lamb * values
        # Lu = f condition
        A_1[k*Q:(k + 1)*Q, M_0: M_1] = Lu[:Q,:]
        # boundary condition
        if k == 0:
            A_2[0, M_0: M_1] = values[0,:]
        if k == N_e - 1:
            A_2[1, M_0: M_1] = values[-1,:]

        if k > 0:
            A_c0[k - 1, M_0: M_1] = -values[0, :]
            A_c1[k - 1, M_0: M_1] = -grads[0, :]
        if k < N_e - 1:
            A_c0[k, k * M:(k + 1) * M] = values[-1, :]
            A_c1[k, k * M:(k + 1) * M] = grads[-1, :]
                
        true_f = Lu_f(points[k].detach().numpy(), lamb).reshape([(Q + 1),1])
        f[k*Q:(k + 1)*Q,: ] = true_f[:Q]
    A = np.concatenate((A_1,A_2,A_c0, A_c1),axis=0)
    f[N_e*Q,:] = anal_u(0.)
    f[N_e*Q+1,:] = anal_u(8.)
    print(f.shape)
    return(A,f)


def test(models,N_e,M,Q,w,plot = False):
    epsilon = []
    true_values = []
    numerical_values = []
    test_Q = int(1000/N_e)
    for k in range(N_e):
        points = torch.tensor(np.linspace(8.0/N_e * (k), 8.0/N_e * (k+1), test_Q+1),requires_grad=False).reshape([-1,1])
        out_total = models[k](points).detach().numpy()
        true_value = anal_u(points.numpy()).reshape([-1,1])
        numerical_value = np.dot(np.array(out_total), w[k * M: (k + 1) * M])
        true_values.extend(true_value)
        numerical_values.extend(numerical_value)
        epsilon.extend(true_value - numerical_value)
    true_values = np.array(true_values)
    numerical_values = np.array(numerical_values)
    epsilon = np.array(epsilon)
    epsilon = np.maximum(epsilon, -epsilon)
    print('********************* ERROR *********************')
    print('N_e=%s,M=%s,Q=%s'%(N_e,M,Q))
    print('L_inf=',epsilon.max(),'L_2=',math.sqrt(8*sum(epsilon*epsilon)/len(epsilon)))
        
    x = [(interval_length/N_e)*i / test_Q  for i in range(N_e*(test_Q+1))]
    if plot == True:
        plt.figure()
        plt.plot(x, true_values, label = "exact solution", color='black')
        #plt.plot(x, numerical_values, label = "numerical solution", color='darkblue', linestyle='--')
        plt.legend()
        plt.title('exact solution')
        #plt.savefig('D:/result/3-1-1/exact solution.pdf', dpi=100)
        
        plt.figure()
        plt.plot(x, epsilon, label = "absolute error", color='black')
        plt.legend()
        plt.title('RFM error, $\psi^2$, M=%s Q=%s'%(N_e*M,N_e*Q))
        #plt.savefig('D:/result/3-1-1/error_Ne=%sM=%sQ=%s.pdf'%(N_e,M,Q), dpi=100)
    return(epsilon.max(),math.sqrt(8*sum(epsilon*epsilon)/len(epsilon)))


def main(N_e,M,Q,lamb, plot = False, moore = False):
    # prepare models and collocation pointss
    models,points = pre_define(N_e,M,Q)
    
    # matrix define (Aw=b)
    A,f = cal_matrix(models,points,N_e,M,Q)
    
    # solve
    if moore:
        inv_coeff_mat = pinv(A)  # moore-penrose inverse, shape: (n_units,n_colloc+2)
        w = np.matmul(inv_coeff_mat, f)
    else:
        w = lstsq(A,f)[0]
    
    # test
    return(test(models,N_e,M,Q,w,plot))



if __name__ == '__main__':
    # set_seed(2000)
    #N_e = 16 # the number of sub-domains
    M = 10 # the number of training parameters per sub-domain
    Q = 20 # the number of collocation pointss per sub-domain 
    R_m = 1 # the maximum magnitude of the random coefficients
    lamb = 4
    for N_e in [4,8,16,32]:
        main(N_e,M,Q,lamb,True)