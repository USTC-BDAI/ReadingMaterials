# -*- coding: utf-8 -*-

from email.mime import base
import numpy as np
import torch
import torch.nn as nn
import itertools
from scipy.linalg import lstsq,pinv
from scipy.fftpack import fftshift,fftn

from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt

import utils

rand_mag = 1.0
def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        if m.weight is not None:
            nn.init.uniform_(m.weight, a = 0, b = rand_mag)
        if m.bias is not None:
            nn.init.uniform_(m.bias, a = -rand_mag, b = rand_mag)
        #nn.init.normal_(m.weight, mean=0, std=1)
        #nn.init.normal_(m.bias, mean=0, std=1)

class Ada_Rescale_W(object):

    def __init__(self, l=0, f=0.5, s=1.0) -> None:

        self.l = l
        self.f = f
        self.s = s

    def rescale(self, m, ):

        if isinstance(m, (nn.Conv2d, nn.Linear)):
            if m.weight is not None:
                c = m.weight.data.size(0)

                for l in range(self.l):
                    e = int(c >> (l + 1))
                    m.weight.data[:e] *= self.f

                # m.weight.data[:] *= self.f

    def shift(self, m, ):

        if isinstance(m, (nn.Conv2d, nn.Linear)):
            if m.weight is not None:
                c = m.weight.data.size(0)

                for l in range(self.l):
                    # e = (l + 1) * c // (self.l + 1)
                    # m.weight.data[e:] += self.s * rand_mag
                    m.weight.data += self.s * rand_mag

                # m.weight.data[:] *= self.f


class SinAct(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sin(x)


class CosAct(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.cos(x)


class ChebPoly(nn.Module):
    def __init__(self, n : int) -> None:
        super().__init__()

        self.n = n

    def forward(self, x):

        v = torch.cos(self.n * torch.acos(x))

        return v


class Poly(nn.Module):
    def __init__(self, n, coefs = None) -> None:
        super().__init__()

        self.n = n

        if coefs is not None:
            assert len(coefs) == self.n
            self.coefs = coefs
        else:
            self.coefs = [0] * self.n

    def forward(self, x):
        
        y = torch.ones_like(x)
        u = 0

        for c in self.coefs:
            u += 1.0 * c * y
            y = y * x

        return u

    def __repr__(self):
        return f"{self.coefs}"


class ChebBasis(nn.Module):
    def __init__(self, n : int) -> None:
        super().__init__()

        assert n > 0
        self.n = n

        # self.basis = [ChebPoly(i) for i in range(n)]
        self.basis = [None for _ in range(n)]

        self.basis[0] = Poly(n=1, coefs=[1,])

        if n > 1:
            self.basis[1] = Poly(n=2, coefs=[0, 1])

        if n > 2:
            for i in range(2, n):
                coef_0 = self.basis[i - 2].coefs
                coef_1 = self.basis[i - 1].coefs
                coefs = [0] * (i + 1)

                coefs[0] = -coef_0[0]

                for j in range(1, i - 1):
                    coefs[j] = 2 * coef_1[j - 1] - coef_0[j]

                coefs[i - 1] = 2 * coef_1[i - 2]
                coefs[i] = 2 * coef_1[i - 1]

                self.basis[i] = Poly(n=i + 1, coefs=coefs)

        # for b in self.basis:
        #     print(b)


    def forward(self, x : torch.Tensor):

        v = [self.basis[i](x) for i in range(self.n)]

        v = torch.stack(v, dim=-1)

        return v


class MultiDChebBasis(nn.Module):
    def __init__(self, n : int=1, ndim : int=1) -> None:
        super().__init__()

        self.n = n
        self.ndim = ndim

        self.basis = ChebBasis(self.n)

        self.identity = (np.eye(self.ndim, dtype=np.int64) * (self.n - 1) + 1).tolist()

    def forward(self, x):

        data_shape = x.size()[:-1]

        y = None

        for d in range(self.ndim):
            if y is None:
                y = self.basis(x[..., d]).view(*data_shape, *self.identity[d])
            else:
                y = y * self.basis(x[..., d]).view(*data_shape, *self.identity[d])

        y = y.view(*data_shape, -1)

        # y = None

        # for d in range(self.ndim):
        #     if y is None:
        #         y = self.basis(x[..., d])
        #     else:
        #         y += self.basis(x[..., d])

        # x = torch.mean(x, dim=-1)
        # y = self.basis(x)

        return y


class ChebRandBasis(nn.Module):
    def __init__(self, n : int) -> None:
        super().__init__()

        assert n > 0
        self.n = n

        self.cheb = ChebBasis(n=n)
        self.affine = nn.Linear(1, 1, bias=True)


    def forward(self, x):

        x = self.affine(x)
        v = self.cheb(x)

        return v


class LegendreBasis(nn.Module):
    def __init__(self, n : int) -> None:
        super().__init__()

        assert n > 0
        self.n = n

        # self.basis = [ChebPoly(i) for i in range(n)]
        self.basis = [None for _ in range(n)]

        self.basis[0] = Poly(n=1, coefs=[1,])

        if n > 1:
            self.basis[1] = Poly(n=2, coefs=[0, 1])

        if n > 2:
            for i in range(2, n):
                coef_0 = self.basis[i - 2].coefs
                coef_1 = self.basis[i - 1].coefs
                coefs = [0] * (i + 1)

                coefs[0] = - (i - 1) / i * coef_0[0]

                for j in range(1, i - 1):
                    coefs[j] = (2 * i - 1) / i * coef_1[j - 1] - (i - 1) / i * coef_0[j]

                coefs[i - 1] = (2 * i - 1) / i * coef_1[i - 2]
                coefs[i] = (2 * i - 1) / i * coef_1[i - 1]

                self.basis[i] = Poly(n=i + 1, coefs=coefs)

        # for b in self.basis:
        #     print(b)


    def forward(self, x):

        v = [self.basis[i](x) for i in range(self.n)]

        v = torch.cat(v, dim=-1)

        return v


class LegendreRandBasis(nn.Module):
    def __init__(self, n : int) -> None:
        super().__init__()

        assert n > 0
        self.n = n

        self.cheb = LegendreBasis(n=n)
        self.affine = nn.Linear(1, 1, bias=True)


    def forward(self, x):

        x = self.affine(x)
        v = self.cheb(x)

        return v

class local_rep(nn.Module):
    def __init__(self, in_features, out_features, hidden_layers, M, x_max, x_min, t_max, t_min, type="elm-t"):
        super(local_rep, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = M
        self.hidden_layers  = hidden_layers
        self.M = M
        self.a = torch.tensor([2.0/(x_max - x_min),2.0/(t_max - t_min)])
        self.x_0 = torch.tensor([(x_max + x_min)/2,(t_max + t_min)/2])

        if type == "elm-t":
            self.hidden_layer = nn.Sequential(nn.Linear(self.in_features, self.hidden_features, bias=True),nn.Tanh())
        elif type == "elm-s":
            self.hidden_layer = nn.Sequential(nn.Linear(self.in_features, self.hidden_features, bias=True),SinAct())
        elif type == "cheb":
            self.hidden_layer = MultiDChebBasis(n=self.hidden_features, ndim=self.in_features)

    def forward(self,x):
        x = self.a * (x - self.x_0)
        x = self.hidden_layer(x)
        return x


def pre_define_rfm(Nx,Nt,M,Qx,Qt,x0,x1,t0,t1,type="elm-t"):
    models = []
    points = []
    L = x1 - x0
    tf = t1 - t0
    for k in range(Nx):
        model_for_x = []
        point_for_x = []
        x_min = L/Nx * k + x0
        x_max = L/Nx * (k+1) + x0
        x_devide = np.linspace(x_min, x_max, Qx + 1)
        for n in range(Nt):
            t_min = tf/Nt * n + t0
            t_max = tf/Nt * (n+1) + t0
            model = local_rep(in_features = 2, out_features = 1, hidden_layers = 1, M = M, x_min = x_min, 
                            x_max = x_max, t_min = t_min, t_max = t_max, type = type)
            model = model.apply(weights_init)
            model = model.double()
            for param in model.parameters():
                param.requires_grad = False
            model_for_x.append(model)
            t_devide = np.linspace(t_min, t_max, Qt + 1)
            grid = np.array(list(itertools.product(x_devide,t_devide))).reshape(Qx+1,Qt+1,2)
            point_for_x.append(torch.tensor(grid,requires_grad=True))
        models.append(model_for_x)
        points.append(point_for_x)
    return(models,points)


class local_mul_rep(nn.Module):
    def __init__(self, space_features, out_features, hidden_layers, Mx, Mt, x_max, x_min, t_max, t_min):
        super(local_mul_rep, self).__init__()
        self.x_features = space_features
        self.out_features = out_features
        self.hidden_features = Mx
        self.hidden_layers  = hidden_layers
        self.x_max = x_max
        self.x_min = x_min
        self.t_max = t_max
        self.t_min = t_min
        self.Mx = Mx
        self.Mt = Mt
        self.a_x = 2.0 / (x_max - x_min)
        self.a_t = 2.0 / (t_max - t_min)
        self.x_0 = (x_max + x_min) / 2.0
        self.t_0 = (t_max + t_min) / 2.0
        self.hx = nn.Sequential(nn.Linear(self.x_features, self.hidden_features, bias=True),SinAct(),)#,nn.Tanh())
        self.ht = nn.Sequential(nn.Linear(1, self.hidden_features * self.Mt, bias=True),SinAct(),)#,nn.Tanh())
        #print([x_min,x_max],[t_min,t_max])

    def forward(self,x):

        part = x.size()[:-1]
        
        # x-axis
        y = self.a_x * (x[..., :self.x_features] - self.x_0)
        y_x = self.hx(y)

        # t-axis
        y = self.a_t * (x[..., self.x_features:] - self.t_0)
        y_t_rfm = self.ht(y)

        c0 = (y >= -1) & (y <= 1)
        y0 = 1.0

        y_t_pou = c0 * y0

        y_t = y_t_pou * y_t_rfm
        
        y = y_x.view(*part, self.Mx, 1) * y_t.view(*part, -1, self.Mt)
        y = y.view(*part, -1)
        
        return y

def hook(m, gin, gout):
    print(gin[0].norm())

class local_mul_nn(nn.Module):
    def __init__(self, space_features, out_features, hidden_layers, Mx, Mt, x_max, x_min, t_max, t_min):
        super(local_mul_nn, self).__init__()
        self.x_features = space_features
        self.out_features = out_features
        self.hidden_features = Mx
        self.hidden_layers  = hidden_layers
        self.x_max = x_max
        self.x_min = x_min
        self.t_max = t_max
        self.t_min = t_min
        self.Mx = Mx
        self.Mt = Mt
        self.a_x = 2.0 / (x_max - x_min)
        self.a_t = 2.0 / (t_max - t_min)
        self.x_0 = (x_max + x_min) / 2.0
        self.t_0 = (t_max + t_min) / 2.0
        self.hx = nn.Sequential(nn.Linear(self.x_features, self.hidden_features, bias=True),SinAct(),)#nn.Tanh())
        self.ht = nn.Sequential(nn.Linear(1, self.hidden_features * self.Mt, bias=True),SinAct(),)#nn.Tanh())
        self.fc = nn.Linear(Mx, 1, bias=False)
        #print([x_min,x_max],[t_min,t_max])

        # self.hx[0].register_backward_hook(hook)
        # self.ht[0].register_backward_hook(hook)
        # self.fc.register_backward_hook(hook)

    def feature(self, x):

        part = x.size()[:-1]
        
        # x-axis
        y = self.a_x * (x[..., :self.x_features] - self.x_0)
        y_x = self.hx(y)

        # t-axis
        y = self.a_t * (x[..., self.x_features:] - self.t_0)
        y_t_rfm = self.ht(y)

        c0 = (y >= -1) & (y <= 1)
        y0 = 1.0

        y_t_pou = c0 * y0

        y_t = y_t_pou * y_t_rfm
        
        y = y_x.view(*part, self.Mx, 1) * y_t.view(*part, -1, self.Mt)
        y = y.view(*part, -1)

        return y

    def forward(self,x):
        
        y = self.feature(x)

        y = self.fc(y)
        
        return y


class local_mul_t_psib_rep(nn.Module):
    def __init__(self, space_features, out_features, hidden_layers, Mx, Mt, x_max, x_min, t_max, t_min, t0, t1):
        super(local_mul_t_psib_rep, self).__init__()
        self.x_features = space_features
        self.out_features = out_features
        self.hidden_features = Mx
        self.hidden_layers  = hidden_layers
        self.x_max = x_max
        self.x_min = x_min
        self.t_max = t_max
        self.t_min = t_min
        self.Mx = Mx
        self.Mt = Mt
        self.t0 = t0
        self.t1 = t1
        self.a_x = 2.0 / (x_max - x_min)
        self.a_t = 2.0 / (t_max - t_min)
        self.x_0 = (x_max + x_min) / 2.0
        self.t_0 = (t_max + t_min) / 2.0
        self.hx = nn.Sequential(nn.Linear(self.x_features, self.hidden_features, bias=True),nn.Sigmoid())
        self.ht = nn.Sequential(nn.Linear(1, self.hidden_features * self.Mt, bias=True),nn.Sigmoid())
        #print([x_min,x_max],[t_min,t_max])

    def forward(self,x):

        part = x.size()[:-1]
        
        # x-axis
        x_axis = self.a_x * (x[..., :self.x_features] - self.x_0)
        y_x = self.hx(x_axis)

        # t-axis
        t_axis = self.a_t * (x[..., self.x_features:] - self.t_0)

        y_rfm = self.ht(t_axis)
        
        c0 = (t_axis > -5 / 4) & (t_axis <= -3 / 4)
        c1 = (t_axis > -3 / 4) & (t_axis <= 3 / 4)
        c2 = (t_axis > 3 / 4) & (t_axis <= 5 / 4)

        if self.t_min == self.t0:
            y0 = 1.0
        else:
            y0 = (1 + torch.sin(2 * torch.pi * t_axis)) / 2
        
        y1 = 1.0
        
        if self.t_max == self.t1:
            y2 = 1.0
        else:
            y2 = (1 - torch.sin(2 * torch.pi * t_axis)) / 2

        y_pou = c0 * y0 + c1 * y1 + c2 * y2
        
        y_t = y_pou * y_rfm
        
        y = y_x.view(*part, self.Mx, 1) * y_t.view(*part, -1, self.Mt)
        y = y.view(*part, -1)
        
        return y


def generate_points(Nx: int, Nt: int, Qx: int, Qt: int, x0: float, x1: float, t0: float, t1: float):

    dx = (x1 - x0) / Nx
    dt = (t1 - t0) / Nt

    points = list()

    for kx in range(Nx):

        points_x = list()

        x = np.linspace(x0 + kx * dx, x0 + (kx + 1) * dx, num=Qx + 1, endpoint=True).reshape(Qx + 1, 1, 1).repeat(Qt + 1, axis=1)

        for kt in range(Nt):

            t = np.linspace(t0 + kt * dt, t0 + (kt + 1) * dt, num=Qt + 1, endpoint=True).reshape(1, Qt + 1, 1).repeat(Qx + 1, axis=0)

            xt = np.concatenate((x, t), axis=2)
            xt = torch.tensor(xt, requires_grad=True)
            points_x.append(xt)

        # points_x = np.concatenate(points_x, axis=1)
        points.append(points_x)

    # points = np.concatenate(points, axis=0)

    return points


def pre_define_mul(Nx,Nt,Mx,Mt,Qx,Qt,x0,x1,t0,t1,mtype="psia"):
    models = []
    points = []
    L = x1 - x0
    tf = t1 - t0
    x_adapter = Ada_Rescale_W(l=0, f=1.0, s=0.0)
    t_adapter = Ada_Rescale_W(l=0, f=2.0, s=0.0)
    for k in range(Nx):
        model_for_x = []
        point_for_x = []
        x_min = L/Nx * k + x0
        x_max = L/Nx * (k+1) + x0
        x_devide = np.linspace(x_min, x_max, Qx + 1)
        for n in range(Nt):
            t_min = tf/Nt * n
            t_max = tf/Nt * (n+1)
            if mtype == "psia":
                model = local_mul_rep(space_features = 1, out_features = 1, hidden_layers = 1, Mx = Mx, Mt = Mt, x_min = x_min, 
                                      x_max = x_max, t_min = t_min, t_max = t_max)
            elif mtype == "nn":
                model = local_mul_nn(space_features = 1, out_features = 1, hidden_layers = 1, Mx = Mx, Mt = Mt, x_min = x_min, 
                                      x_max = x_max, t_min = t_min, t_max = t_max)
            elif mtype == "psib":
                model = local_mul_t_psib_rep(space_features = 1, out_features = 1, hidden_layers = 1, Mx = Mx, Mt = Mt, x_min = x_min, x_max = x_max, t_min = t_min, t_max = t_max, t0=t0, t1=t1)
            else:
                raise NotImplementedError
            model = model.apply(weights_init)
            model.hx = model.hx.apply(x_adapter.rescale)
            model.ht = model.ht.apply(t_adapter.rescale)
            model = model.double()
            if not mtype == "nn":
                for param in model.parameters():
                    param.requires_grad = False
            model_for_x.append(model)
            t_devide = np.linspace(t_min, t_max, Qt + 1)
            # grid = np.array(list(itertools.product(x_devide,t_devide))).reshape(Qx+1,Qt+1,2)
            # point_for_x.append(torch.tensor(grid,requires_grad=True))
        models.append(model_for_x)
        # points.append(point_for_x)
        points = generate_points(Nx=Nx, Nt=Nt, Qx=Qx, Qt=Qt, x0=x0, x1=x1, t0=t0, t1=t1)
    return(models,points)


def get_anal_u(vanal_u, points, Nx, Qx, nt=None, qt=None, tshift=0):

    if nt is None:
        nt = 0
    if qt is None:
        qt = 0

    point = list()
    
    for k in range(Nx):
        point.append(points[k][nt][:Qx, qt, :].detach().numpy().reshape((Qx, 2)))
    point = np.concatenate(point, axis=0)
    
    u_value = vanal_u(point[:, 0], point[:, 1] + tshift).reshape((Nx * Qx, 1))

    return u_value


def get_anal_fx(vanal_f, points, Nx, Qx, nt=None, qt=None):

    if nt is None:
        nt = 0
    if qt is None:
        qt = 0

    point = list()
    
    for k in range(Nx):
        point.append(points[k][nt][:Qx, qt, :].detach().numpy().reshape((Qx, 2)))
    point = np.concatenate(point, axis=0)
    
    u_value = vanal_f(point[:, 0]).reshape((Nx * Qx, 1))

    return u_value


def get_numsol_1d(models, points, w, Nx, Nt, M, Qx, Qt):
    
    u_value_xt = list()
    u_value_x = list()

    for kx in range(Nx):

        u_value = list()

        for kt in range(Nt):
            model = models[kx][kt]
            point = points[kx][kt][:Qx, :Qt, :]
            base_values = model(point).detach().numpy()
            M0 = (kx * Nt + kt) * M
            value = np.dot(base_values, w[M0: M0 + M, :])
            u_value.append(value.reshape(Qx, Qt, 1))

        u_value_x.append(np.concatenate(u_value, axis=1))

    u_value_xt = np.concatenate(u_value_x, axis=0)

    return u_value_xt


def get_num_values(models, points, w, Nx, Ny, M, Qx, Qy):
    
    u_value = list()

    for kx in range(Nx):
        for ky in range(Ny):
            model = models[kx][ky]
            point = points[kx][ky][:Qx, :Qy, :]
            base_values = model(point).detach().numpy()
            M0 = (kx * Ny + ky) * M
            u_value.append(np.dot(base_values, w[M0: M0 + M, :]))

    u_value = np.concatenate(u_value, axis=0)

    return u_value


def get_num_u(models, points, w, Nx, Nt, M, Qx, Qt, nt=None, qt=None):
    
    if nt is None:
        nt = Nt - 1
    if qt is None:
        qt = Qt
    
    u_value = list()

    for k in range(Nx):
        model = models[k][nt]
        point = points[k][nt][:Qx, qt, :]
        base_values = model(point).detach().numpy()
        M0 = (k * Nt + nt) * M
        u_value.append(np.dot(base_values, w[M0: M0 + M, :]))

    u_value = np.concatenate(u_value, axis=0)

    return u_value


def get_num_ut(models, points, w, Nx, Nt, M, Qx, Qt, nt=None, qt=None):

    if nt is None:
        nt = Nt - 1
    if qt is None:
        qt = Qt
    
    ut_value = list()

    for k in range(Nx):
        model = models[k][nt]
        point = points[k][nt][:Qx, qt, :]
        base_values = model(point)
        base_grads = list()
        for i in range(M):
            grad = torch.autograd.grad(outputs=base_values[:, i], inputs=point, \
                grad_outputs=torch.ones_like(base_values[:,i]), \
                create_graph = True, retain_graph = True)[0]
            base_grads.append(grad.detach().numpy())
        base_grads = np.array(base_grads).swapaxes(0, 2)
        base_t_grads = base_grads[1, :, :]
        M0 = (k * Nt + nt) * M
        ut_value.append(np.dot(base_t_grads, w[M0: M0 + M, :]))

    ut_value = np.concatenate(ut_value, axis=0)

    return ut_value


def sget_num_values(models, points, w, Nx, Ny, M, Qx, Qy):
    
    u_value = list()

    for kx in range(Nx):
        for ky in range(Ny):
            model = models[0][0]
            xshift = points[kx][ky][0, 0, 0]
            tshift = points[kx][ky][0, 0, 1]
            inshift = torch.tensor([xshift, tshift]).view(1, 1, 2)
            point = points[kx][ky][:Qx, :Qy, :] - inshift
            base_values = model(point).detach().numpy()
            M0 = (kx * Ny + ky) * M
            u_value.append(np.dot(base_values, w[M0: M0 + M, :]))

    u_value = np.concatenate(u_value, axis=0)

    return u_value


def sget_num_u(models, points, w, Nx, Nt, M, Qx, Qt, nt=None, qt=None):
    
    if nt is None:
        nt = Nt - 1
    if qt is None:
        qt = Qt
    
    u_value = list()

    for k in range(Nx):
        model = models[0][0]
        _xshift = points[k][nt][0, 0, 0]
        _tshift = points[k][nt][0, 0, 1]
        inshift = torch.tensor([_xshift, _tshift]).view(1, 2)
        point = points[k][nt][:Qx, qt, :] - inshift
        base_values = model(point).detach().numpy()
        M0 = (k * Nt + nt) * M
        u_value.append(np.dot(base_values, w[M0: M0 + M, :]))

    u_value = np.concatenate(u_value, axis=0)

    return u_value


def sget_num_ut(models, points, w, Nx, Nt, M, Qx, Qt, nt=None, qt=None):

    if nt is None:
        nt = Nt - 1
    if qt is None:
        qt = Qt
    
    ut_value = list()

    for k in range(Nx):
        model = models[0][0]
        _xshift = points[k][nt][0, 0, 0]
        _tshift = points[k][nt][0, 0, 1]
        inshift = torch.tensor([_xshift, _tshift]).view(1, 2)
        point = points[k][nt][:Qx, qt, :] - inshift
        base_values = model(point)
        base_grads = list()
        for i in range(M):
            grad = torch.autograd.grad(outputs=base_values[:, i], inputs=point, \
                grad_outputs=torch.ones_like(base_values[:,i]), \
                create_graph = True, retain_graph = True)[0]
            base_grads.append(grad.detach().numpy())
        base_grads = np.array(base_grads).swapaxes(0, 2)
        base_t_grads = base_grads[1, :, :]
        M0 = (k * Nt + nt) * M
        ut_value.append(np.dot(base_t_grads, w[M0: M0 + M, :]))

    ut_value = np.concatenate(ut_value, axis=0)

    return ut_value


def get_num_u_tpsib(models, points, w, Nx, Nt, M, Qx, Qt, nt=None, qt=None):
    
    if nt is None:
        nt = Nt - 1
    if qt is None:
        qt = Qt
    
    u_value = list()

    for k in range(Nx):

        if nt > 0:
            n_0 = nt - 1
        else:
            n_0 = nt
        if nt < Nt - 1:
            n_1 = nt + 2
        else:
            n_1 = nt + 1

        point = points[k][nt][:Qx, qt, :]
        u_value_ = np.zeros((Qx, 1))

        for n_s in range(n_0, n_1):
            M0 = (k * Nt + n_s) * M
            model = models[k][n_s]
            base_values = model(point).detach().numpy()
            u_value_ += np.dot(base_values, w[M0: M0 + M, :])

        u_value.append(u_value_)

    u_value = np.concatenate(u_value, axis=0)

    return u_value


def get_num_ut_tpsib(models, points, w, Nx, Nt, M, Qx, Qt, nt=None, qt=None):

    if nt is None:
        nt = Nt - 1
    if qt is None:
        qt = Qt
    
    ut_value = list()

    for k in range(Nx):
        
        if nt > 0:
            n_0 = nt - 1
        else:
            n_0 = nt
        if nt < Nt - 1:
            n_1 = nt + 2
        else:
            n_1 = nt + 1
        
        point = points[k][nt][:Qx, qt, :]
        ut_value_ = np.zeros((Qx, 1))

        for n_s in range(n_0, n_1):
            M0 = (k * Nt + n_s) * M
            model = models[k][n_s]
            base_values = model(point)
            base_grads = list()
            for i in range(M):
                grad = torch.autograd.grad(outputs=base_values[:, i], inputs=point, \
                    grad_outputs=torch.ones_like(base_values[:,i]), \
                    create_graph = True, retain_graph = True)[0]
                base_grads.append(grad.detach().numpy())
            base_grads = np.array(base_grads).swapaxes(0, 2)
            base_t_grads = base_grads[1, :, :]
            ut_value_ += np.dot(base_t_grads, w[M0: M0 + M, :])

        ut_value.append(ut_value_)

    ut_value = np.concatenate(ut_value, axis=0)

    return ut_value


def solve_lst_square(A, f, moore=False):

    # # rescaling
    # max_value = 10.0
    # for i in range(len(A)):
    #     if np.abs(A[i,:]).max()==0:
    #         print("error line : ",i)
    #         continue
    #     ratio = max_value/np.abs(A[i,:]).max()
    #     A[i,:] = A[i,:]*ratio
    #     f[i] = f[i]*ratio
    
    # solve
    if moore:
        inv_coeff_mat = pinv(A)  # moore-penrose inverse, shape: (n_units,n_colloc+2)
        w = np.matmul(inv_coeff_mat, f)
    else:
        w, res, rnk, s = lstsq(A,f, lapack_driver="gelss")
        print(s[0], s[-1])

    print(np.linalg.norm(np.matmul(A, w) - f), np.linalg.norm(np.matmul(A, w) - f) / np.linalg.norm(f))

    return w


def solve_lst_square_lowrank(A, f, moore=False):

    # rescaling
    max_value = 10.0
    for i in range(A.shape[0]):

        if np.abs(A[i,:]).max()==0:
            print("error line : ",i)
            continue
        ratio = max_value/np.abs(A[i,:]).max()
        A[i,:] = A[i,:]*ratio
        f[i] = f[i]*ratio
    
    # solve

    rs = list()
    rrs = list()

    w_ref = lstsq(A.toarray(), f)[0]
    r_ref = np.linalg.norm(A @ w_ref - f)
    rr_ref = np.linalg.norm(A @ w_ref - f) / np.linalg.norm(f)

    for i in range(10):

        q = A.shape[1] // 10 * (i + 1)

        print(A.shape, q)

        u, s, vh = svds(A=A, k=q-1)
        # u, s, vh = svd(a=A.toarray())

        quanzhong = ((u.T @ f) / s.reshape(-1, 1)) 

        w = vh.T @ quanzhong
        w = w.reshape(-1, 1)

        r = np.linalg.norm(A @ w - f)
        rr = np.linalg.norm(A @ w - f) / np.linalg.norm(f)

        rs.append(r)
        rrs.append(rr)

        fig, ax = plt.subplots()
        ax.plot(np.arange(q-1)+1, s[::-1], c="r")
        ax.set_yscale("log")
        ax.set_ylabel("Singularity Values", fontsize=16)
        ax.set_xlabel("Singularity Index", fontsize=16)

        ax2 = ax.twinx()
        ax2.plot(np.arange(q-1)+1, np.abs(quanzhong[::-1]), c="g")
        ax2.set_yscale("log")
        ax2.set_ylabel("Absolute Weights on Feature Vector", fontsize=16)

        ax.set_title(f"{q:d} Singularities", fontsize=20)

        fig.tight_layout()
        fig.savefig(f"singularities-{q:d}.png")
        plt.clf(); plt.cla(); plt.close()

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

    ax[0].plot((np.arange(10) + 1) * A.shape[1] // 10, rs)
    ax[0].plot([A.shape[1] // 10, A.shape[1]], [r_ref, r_ref], label="reference")
    ax[0].set_title("Absolute Error")
    ax[0].set_xlabel("Number of Singularities Used")
    ax[0].set_yscale("log")
    ax[0].legend(loc="best")

    ax[1].plot((np.arange(10) + 1) * A.shape[1] // 10, rrs)
    ax[1].plot([A.shape[1] // 10, A.shape[1]], [rr_ref, rr_ref], label="reference")
    ax[1].set_title("Relative Error")
    ax[1].set_xlabel("Number of Singularities Used")
    ax[1].set_yscale("log")
    ax[1].legend(loc="best")

    fig.tight_layout()
    fig.savefig(f"sing.png")
    plt.clf(); plt.cla(); plt.cla()

    # u, s, vh = torch.svd_lowrank(A=torch.from_numpy(A).cuda(), q=q)
    # w = vh @ (u.T @ torch.from_numpy(f).cuda() / s.view(-1, 1))
    # w = w.detach().cpu().numpy()

    print(np.linalg.norm(A @ w - f), np.linalg.norm(A @ w - f) / np.linalg.norm(f))

    return w


def solve_lst_square_richarderson(A : np.ndarray, f : np.ndarray, epochs=100):

    n_cond, n_param = A.shape

    w = np.zeros((n_param, 1), dtype=np.float64)# * 1e4

    a = 1e-6
    AT = A.transpose()

    residuals = list()

    for epoch in range(epochs):

        r = f - np.matmul(A, w)
        w = w + a * np.matmul(AT, r)

        residual = np.linalg.norm(np.matmul(A, w) - f)
        print(f"Epoch {epoch:d}/{epochs:d} error: {residual}")

        residuals.append(residual)

    np.save("Richard.npy", np.array(residuals))

    return w


def solve_lst_square_jacob(A : np.ndarray, f : np.ndarray, epochs=100):

    n_cond, n_param = A.shape

    w = np.zeros((n_param, 1), dtype=np.float64)# * 1e4

    AT = A.transpose()

    # print(np.matmul(AT[0, :], A[:, 0]))

    d = np.sum(np.power(A, 2.0), axis=0)
    # input(d[0])
    iDA = np.diag(1/d)
    m = np.matmul(iDA, AT)

    residuals = list()

    for epoch in range(epochs):

        r = f - np.matmul(A, w)
        w = w + np.matmul(m, r)

        residual = np.linalg.norm(np.matmul(A, w) - f)
        print(f"Epoch {epoch:d}/{epochs:d} error: {residual}")

        residuals.append(residual)

    np.save("Jacob.npy", np.array(residuals))

    return w


def solve_lst_square_gauss_seidel(A : np.ndarray, f : np.ndarray, epochs=100):

    n_cond, n_param = A.shape

    w = np.zeros((n_param, 1), dtype=np.float64)# * 1e4

    AT = A.transpose()

    d = np.sum(np.power(A, 2.0), axis=0)
    iDA = np.diag(1/d)
    m = np.matmul(iDA, AT)

    residuals = list()

    for epoch in range(epochs):

        r = f - np.matmul(A, w)
        w = w + np.matmul(m, r)

        residual = np.linalg.norm(np.matmul(A, w) - f)
        print(f"Epoch {epoch:d}/{epochs:d} error: {residual}")

        residuals.append(residual)

    np.save("Gauss-Seidel.npy", np.array(residuals))

    return w


def solve_lst_square_CGLS(A : np.ndarray, f : np.ndarray, epochs=100):

    n_cond, n_param = A.shape

    w = np.zeros((n_param, 1), dtype=np.float64)# * 1e4

    AT = A.transpose()

    r = f - np.matmul(A, w)
    p = np.matmul(AT, r)
    s = np.matmul(AT, r)
    gm = np.sum(np.power(s, 2.0))

    residuals = list()

    for epoch in range(epochs):

        q = np.matmul(A, p)
        al = gm / np.sum(np.power(q, 2.0))
        w = w + al * p
        r = r - al * q
        s = np.matmul(AT, r)
        gm_ = np.sum(np.power(s, 2.0))
        be = gm_ / gm
        p = s + be * p
        gm = gm_

        residual = np.linalg.norm(np.matmul(A, w) - f)

        print(f"Epoch {epoch:d}/{epochs:d} error: {residual}")

        residuals.append(residual)

    np.save("CGLS.npy", np.array(residuals))

    return w


def solve_lst_square_SGD(A : np.ndarray, f : np.ndarray, batch_size = 64, epochs = 10, lr=1e-1, w0=None):

    beta = 0.95

    n_cond, n_param = A.shape

    if w0 is None:
        w = np.zeros((n_param, 1), dtype=np.float128)
    else:
        w = w0

    v = np.zeros_like(w)

    residuals = list()

    for epoch in range(epochs):

        ind_cond = np.random.permutation(n_cond)

        st = 0
        it = 0

        while st < n_cond:

            end = min(st + batch_size, n_cond)

            A_ = A[ind_cond[st : end], :]
            f_ = f[ind_cond[st : end], :]

            dw = 2 * (np.matmul(np.matmul(A_.transpose(), A_), w) - np.matmul(A_.transpose(), f_))
            v = beta * v + (1 - beta) * dw
            w = w - lr * v

            st = end
            it += 1

            if it == 500:
                lr /= 2

        r = np.linalg.norm(np.matmul(A, w) - f)
        residuals.append(r)

        print(f"Epoch {epoch+1:d}/{epochs:d} error: {np.linalg.norm(np.matmul(A, w) - f)}")
        print(f"Epoch {epoch+1:d}/{epochs:d} end. ||v||={np.linalg.norm(v)}")

    np.save("SGD.npy", np.array(residuals))

    return w


def test_multidim_chebyshevbasis():

    print("test_multidim_chebyshevbasis")

    n = 5
    ndim = 2
    N = 100

    basis = MultiDChebBasis(n=n, ndim=ndim)

    x = np.linspace(-1, 1, N + 1)
    y = np.linspace(-1, 1, N + 1)
    grid = np.array(list(itertools.product(x, y))).reshape((N + 1, N + 1, ndim))
    grid = torch.from_numpy(grid)
    out = basis(grid)
    print(grid.shape, out.shape)



if __name__ == "__main__":
    # model = local_mul_t_psib_rep(space_features=1, out_features=1, hidden_layers=1, Mx=10, Mt=1, x_min=0.0, x_max=1.0, t_min=0.0, t_max=1.0)
    # x = torch.rand((20, 2))
    # x[:, 1] -= -1
    # out = model(x)

    # print(out.min(), out.max())

    test_multidim_chebyshevbasis()
