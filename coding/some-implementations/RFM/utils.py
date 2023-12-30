# -*- coding: utf-8 -*-

import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import itertools
import matplotlib.tri as mptri
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from scipy.spatial import Delaunay
import seaborn as sns
import imageio
import time

def set_seed(x):
    np.random.seed(x)
    torch.manual_seed(x)
    torch.cuda.manual_seed_all(x)
    torch.backends.cudnn.deterministic = True


def L_inf_error(v, axis=None):
    return np.max(np.abs(v), axis=axis)

def L_2_error(v, axis=None):
    return np.sqrt(np.sum(np.power(v, 2.0), axis=axis) / v.shape[0])


def visualize_single(values : np.ndarray, x0 : float, x1 : float, t0 : float, t1 : float, savename : str="visualize.png"):

    x_bins, t_bins = values.shape

    x = np.linspace(x0, x1, x_bins + 1)[:x_bins]
    y = np.linspace(t0, t1, t_bins + 1)[:t_bins]
    x,y = np.meshgrid(x,y)

    L = x1 - x0
    tf = t1 - t0

    font2 = {
    'weight' : 'normal',
    'size'   : 24,
    }

    fig, ax = plt.subplots(figsize=[6, 8])
    fig.set_tight_layout(True)
    # ax.set_aspect('equal')
    ax.set_xlim([x0, x1])
    ax.set_ylim([t0, t1])
    ax.tick_params(labelsize=20)
    ax.set_xlabel('x',font2)
    ax.set_ylabel('t',font2)
    im = ax.pcolor(x, y, values.T, cmap='jet')
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=20)
    cbar.ax.ticklabel_format(style="sci", scilimits=(-1, 2), axis='both')
    cbar.ax.yaxis.get_offset_text().set_fontsize(12)
    cbar.update_ticks()

    # fig.tight_layout()
    fig.savefig(savename, dpi=100)
    plt.clf(); plt.cla(); plt.close()

    print(f"{savename} saved.")


class visualize_stc_1d(object):

    def __init__(self, values : np.ndarray, x0 : float, x1 : float, xl="x", yl="u_r") -> None:

        self.x_bins = values.shape[0]

        self.x = np.linspace(x0, x1, self.x_bins + 1)[:self.x_bins]

        self.values = values
        self.x0 = x0
        self.x1 = x1

        font2 = {
        'weight' : 'normal',
        'size'   : 24,
        }

        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.fig.set_tight_layout(True)
        self.ax.set_xlim([x0, x1])
        self.ax.tick_params(labelsize=20)
        self.ax.set_xlabel(xl, font2)
        self.ax.set_ylabel(yl, font2)
        
        self.line, = self.ax.plot(self.x, self.values)

    def __call__(self, savename="g.png"):

        self.fig.savefig(savename)
        
        plt.clf(); plt.cla(); plt.close()


class visualize_dyn_space_1d(object):

    def __init__(self, values : np.ndarray, x0 : float, x1 : float, t0 : float, t1 : float, xl="x", yl="u_r") -> None:

        self.x_bins, self.t_bins = values.shape

        self.x = np.linspace(x0, x1, self.x_bins + 1)[:self.x_bins]

        self.values = values
        self.x0 = x0
        self.x1 = x1
        self.t0 = t0
        self.t1 = t1

        font2 = {
        'weight' : 'normal',
        'size'   : 24,
        }

        ymin, ymax = values.min(), values.max()
        ymin = min(ymin * 0.9, ymin * 1.1)
        ymax = max(ymax * 0.9, ymax * 1.1)

        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.fig.set_tight_layout(True)
        self.ax.set_xlim([x0, x1])
        self.ax.set_ylim([ymin, ymax])
        self.ax.tick_params(labelsize=20)
        self.ax.set_xlabel(xl, font2)
        self.ax.set_ylabel(yl, font2)
        
        self.line, = self.ax.plot(self.x, self.values[:, 0])
        
        

    def update(self, i):

        ydata = self.values[:, i]
        
        self.line.set_ydata(ydata)

        self.ax.set_title(f"t={i * (self.t1 - self.t0) / self.t_bins:.2f}, range=({ydata.max():.2f}, {ydata.min():.2f})")

        # return self.line, self.ax

    def __call__(self, savename="g.gif"):

        fps = int(self.t_bins / (self.t1 - self.t0))
        step = max(1, fps // 120)

        print(savename, fps, step)

        anim = FuncAnimation(self.fig, self.update, frames=np.arange(step, self.t_bins, step=step))
        anim.save(filename=savename, fps=fps // step)

        plt.clf(); plt.cla(); plt.close()


def visualize(true_values, numerical_values, x0, x1, t0, t1, savedir=None, eqname="diff",mename="rfm", img_format="eps"):

    epsilon = np.abs(true_values - numerical_values)

    if savedir is None:
        savedir = os.path.join("outputs", f"{eqname}_{mename}_tf={t1 - t0:.2f}")

    if not os.path.exists(savedir):
        os.makedirs(savedir, exist_ok=True)

    visualize_single(true_values, x0, x1, t0, t1, savename=os.path.join(savedir, f"{eqname}-{mename}-trusol.{img_format}"))
    visualize_single(numerical_values, x0, x1, t0, t1, savename=os.path.join(savedir, f"{eqname}-{mename}-numsol.{img_format}"))
    visualize_single(epsilon, x0, x1, t0, t1, savename=os.path.join(savedir, f"{eqname}-{mename}-eps.{img_format}"))


def visualize_2d_single(values : np.ndarray, x0 : float, x1 : float, y0 : float, y1 : float, t0 : float, t1 : float, savedir="visualization"):

    x_bins, y_bins, _ = values.shape

    x = np.linspace(x0, x1, x_bins + 1)[:x_bins]
    y = np.linspace(y0, y1, y_bins + 1)[:y_bins]

    x,y = np.meshgrid(x,y)

    Lx = x1 - x0
    Ly = y1 - y0
    tf = t1 - t0

    tstep = 1

    t_bins = values.shape[2]

    font2 = {
    'weight' : 'normal',
    'size'   : 20,
    }

    if not os.path.exists(savedir):
        os.makedirs(savedir, exist_ok=True)

    for t in range(0, t_bins, tstep):
        fig, ax = plt.subplots(figsize=[1.2 * Lx, 1.2 * Ly])
        ax.set_aspect('equal')
        ax.set_xlim([x0, x1])
        ax.set_ylim([y0, y1])
        ax.tick_params(labelsize=15)
        ax.set_title(f"Time = {t / t_bins * tf + t0:.2f}")
        ax.set_xlabel('x',font2)
        ax.set_ylabel('y',font2)
        im = ax.pcolor(x, y, values[:, :, t].T, cmap='jet')
        cbar = fig.colorbar(im, ax=ax)
        cbar.ax.tick_params(labelsize=15)
        cbar.ax.yaxis.get_offset_text().set_fontsize(15)
        cbar.update_ticks()
        fig.savefig(os.path.join(savedir, f"t={t:d}.png"))
        plt.clf(); plt.cla(); plt.close()

    gif_imgs = list()
    img_paths = [os.path.join(savedir, f"t={t:d}.png") for t in range(0, t_bins, tstep)]

    for path in img_paths:
        gif_imgs.append(imageio.imread(path))

    imageio.mimsave(os.path.join(savedir, f"tf={tf:.2f}.gif"), gif_imgs, fps=30)


def visualize_2d_wo(eqn, values : np.ndarray, x0 : float, x1 : float, y0 : float, y1 : float, savename="visualization.png"):

    x_bins, y_bins, _ = values.shape

    x = np.linspace(x0, x1, x_bins + 1)[:x_bins]
    y = np.linspace(y0, y1, y_bins + 1)[:y_bins]

    x,y = np.meshgrid(x,y)
    tri = Delaunay(np.stack([x.flatten(), y.flatten()], axis=1))

    is_inter = eqn.indicator(x, y)

    mask = np.empty(tri.nsimplex, dtype=np.bool8)
    mask.fill(True)
    
    # print(mask.sum())

    for i in range(tri.nsimplex):

        vs = tri.simplices[i]
        
        rs = vs // x.shape[1]
        cs = vs % x.shape[1]
        
        pt = np.stack([rs, cs], axis=1)
        
        if not np.all(is_inter[pt[:, 0], pt[:, 1]]):
            mask[i] = False
        
    # print(mask.sum())

    Lx = x1 - x0
    Ly = y1 - y0

    font2 = {
    'weight' : 'normal',
    'size'   : 20,
    }

    fig = plt.figure(figsize=[1.2 * Lx, 1.2 * Ly])
    ax = plt.axes(projection="3d")
    ax.set_aspect('equal')
    ax.set_xlim([x0, x1])
    ax.set_ylim([y0, y1])
    ax.set_zlim([-4, 4])
    ax.tick_params(labelsize=15)
    # ax.set_title(f"Time = {t / t_bins * tf + t0:.2f}")
    ax.set_xlabel('x',font2)
    ax.set_ylabel('y',font2)
    ax.set_zlabel('u',font2)
    im = ax.plot_trisurf(x.flatten(), y.flatten(), values.flatten(), triangles=tri.simplices[mask], cmap=plt.cm.coolwarm, alpha=0.7)
    for ob in eqn.geometry.obstacles:
        pt = ob.sampler(50)
        ax.plot3D(pt[:, 0], pt[:, 1], np.zeros(50,), color="black")
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=15)
    cbar.ax.yaxis.get_offset_text().set_fontsize(15)
    cbar.update_ticks()
    fig.savefig(savename)
    plt.clf(); plt.cla(); plt.close()


def visualize_3d_wo(eqn, values : np.ndarray, x0 : float, x1 : float, y0 : float, y1 : float, t0 : float, t1 : float, savedir="visualization"):

    x_bins, y_bins, t_bins = values.shape

    x = np.linspace(x0, x1, x_bins + 1)[:x_bins]
    y = np.linspace(y0, y1, y_bins + 1)[:y_bins]

    zmin, zmax = np.min(values), np.max(values)

    if zmin < 0:
        zmin *= 1.2
    else:
        zmin *= 0.8

    if zmax > 0:
        zmax *= 1.2
    else:
        zmax *= 0.8

    x,y = np.meshgrid(x,y)
    tri = Delaunay(np.stack([x.flatten(), y.flatten()], axis=1))

    is_inter = eqn.indicator(x, y)

    mask = np.empty(tri.nsimplex, dtype=np.bool8)
    mask.fill(True)
    
    # print(mask.sum())

    for i in range(tri.nsimplex):

        vs = tri.simplices[i]
        
        rs = vs // x.shape[1]
        cs = vs % x.shape[1]
        
        pt = np.stack([rs, cs], axis=1)
        
        if not np.all(is_inter[pt[:, 0], pt[:, 1]]):
            mask[i] = False
        
    
    # print(mask.sum())

    Lx = x1 - x0
    Ly = y1 - y0
    tf = t1 - t0

    tstep = 1

    font2 = {
    'weight' : 'normal',
    'size'   : 20,
    }

    if not os.path.exists(savedir):
        os.makedirs(savedir, exist_ok=True)

    for t in range(0, t_bins, tstep):
        fig = plt.figure(figsize=[1.2 * Lx, 1.2 * Ly])
        ax = plt.axes(projection="3d")
        ax.set_aspect('equal')
        ax.set_xlim([x0, x1])
        ax.set_ylim([y0, y1])
        ax.set_zlim([zmin, zmax])
        ax.tick_params(labelsize=15)
        ax.set_title(f"Time = {t / t_bins * tf + t0:.2f}")
        ax.set_xlabel('x',font2)
        ax.set_ylabel('y',font2)
        ax.set_zlabel('u',font2)
        im = ax.plot_trisurf(x.flatten(), y.flatten(), values[:, :, t].flatten(), triangles=tri.simplices[mask], cmap=plt.cm.coolwarm, alpha=0.7, vmin=zmin, vmax=zmax)
        for ob in eqn.geometry.obstacles:
            pt = ob.sampler(50)
            ax.plot3D(pt[:, 0], pt[:, 1], np.zeros(50,), color="black")
        cbar = fig.colorbar(im, ax=ax)
        cbar.ax.tick_params(labelsize=15)
        cbar.ax.yaxis.get_offset_text().set_fontsize(15)
        cbar.update_ticks()
        fig.savefig(os.path.join(savedir, f"t={t:d}.png"), dpi=100)
        plt.clf(); plt.cla(); plt.close()

    gif_imgs = list()
    img_paths = [os.path.join(savedir, f"t={t:d}.png") for t in range(0, t_bins, tstep)]

    for path in img_paths:
        gif_imgs.append(imageio.imread(path))
        os.remove(path)

    fps = t_bins // tf // 2
    imageio.mimwrite(os.path.join(savedir, f"tf={tf:.2f}.gif"), ims=gif_imgs, format="GIF", fps=fps)


def visualize_2d(true_values, numerical_values, x0, x1, y0, y1, t0, t1, eqname="membrane",mename="mul", vis_type="2d"):

    epsilon = np.abs(true_values - numerical_values)

    visualize_2d_single(true_values, x0, x1, y0, y1, t0, t1, savedir=f'outputs/{eqname}-trusol-{mename}')
    visualize_2d_single(numerical_values, x0, x1, y0, y1, t0, t1, savedir=f'outputs/{eqname}-numsol-{mename}')
    visualize_2d_single(epsilon, x0, x1, y0, y1, t0, t1, savedir=f'outputs/{eqname}-eps-{mename}')


def visualize_time_stats(tf, datas, labels, savename=None):

    fig, ax = plt.subplots(figsize=(5, 5))
    
    for data, label in zip(datas, labels):

        n, t = data.shape

        x = np.arange(tf, step=tf / t)

        mean, std = np.mean(data, axis=0), np.std(data, axis=0)

        ax.plot(x, mean, label=label)
        ax.fill_between(x, y1=np.min(data, axis=0), y2=np.max(data, axis=0), alpha=0.3)
        ax.set_yscale("log")

    ax.legend(loc="best")
    ax.set_xlabel("Time", fontsize=16)
    ax.set_ylabel("Error", fontsize=16)

    if savename is not None:
        plt.tight_layout()
        fig.savefig(savename)

    plt.clf(); plt.cla(); plt.close()


def stest(vanal_u,models,Nx=1,Nt=1,M=1,Mx=None,Mt=None,Qx=1,Qt=1,w=None,x0=0,x1=1,t0=0,t1=1,test_Qx=None,test_Qt=None,block=0):
    L = x1 - x0
    tf = t1 - t0

    if Mx is not None and Mt is not None:
        M = Mx * Mt

    epsilon = []
    true_values = []
    numerical_values = []

    if test_Qx is None:
        test_Qx = 10 * Qx
    if test_Qt is None:
        test_Qt = 10 * Qt

    tshift = block * tf
    
    for k in range(Nx):
        epsilon_x = []
        true_value_x = []
        numerical_value_x = []
        for n in range(Nt):
            # forward and grad
            x_min = L/Nx * k + x0
            x_max = L/Nx * (k+1) + x0
            x_devide = np.linspace(x_min, x_max, test_Qx + 1)[:test_Qx]
            t_min = tf/Nt * n
            t_max = tf/Nt * (n+1)
            t_devide = np.linspace(t_min, t_max, test_Qt + 1)[:test_Qt]
            grid = np.array(list(itertools.product(x_devide,t_devide))).reshape(test_Qx,test_Qt,2)
            test_point = torch.tensor(grid,requires_grad=True)
            in_ = test_point.detach().numpy()
            true_value = vanal_u(in_[:,:,0],in_[:,:,1] + tshift)
            model = models[0][0]
            _xshift = test_point[0, 0, 0]
            _tshift = test_point[0, 0, 1]
            inshift = torch.tensor([_xshift, _tshift]).view(1, 1, 2)
            values = model(test_point - inshift).detach().numpy()
            numerical_value = np.dot(values, w[k*Nt*M + n*M : k*Nt*M + n*M + M,:]).reshape(test_Qx,test_Qt)
            e = np.abs(true_value - numerical_value)
            true_value_x.append(true_value)
            numerical_value_x.append(numerical_value)
            epsilon_x.append(e)
        epsilon_x = np.concatenate(epsilon_x, axis=1)
        epsilon.append(epsilon_x)
        true_value_x = np.concatenate(true_value_x, axis=1)
        true_values.append(true_value_x)
        numerical_value_x = np.concatenate(numerical_value_x, axis=1)
        numerical_values.append(numerical_value_x)
    epsilon = np.concatenate(epsilon, axis=0)
    true_values = np.concatenate(true_values, axis=0)
    numerical_values = np.concatenate(numerical_values, axis=0)
    e = epsilon.reshape((-1,1))
    L_i = L_inf_error(e)
    L_2 = L_2_error(e)
    print('********************* ERROR *********************')
    print(f"Block={block:d},t0={tshift:.2f},t1={tshift + tf:.2f}")
    if Mx is not None and Mt is not None:
        print('Nx={:d},Nt={:d},Mx={:d},Mt={:d},Qx={:d},Qt={:d}'.format(Nx,Nt,Mx,Mt,Qx,Qt))
    else:
        print('Nx={:d},Nt={:d},M={:d},Qx={:d},Qt={:d}'.format(Nx,Nt,M,Qx,Qt))
    print('L_inf={:.2e}'.format(L_i),'L_2={:.2e}'.format(L_2))
    print("边值条件误差")
    print("{:.2e} {:.2e}".format(max(epsilon[0,:]),max(epsilon[-1,:])))
    print("初值、终值误差")
    print("{:.2e} {:.2e}".format(max(epsilon[:,0]),max(epsilon[:,-1])))
    # np.save('./epsilon_psi2.npy',epsilon)

    return true_values, numerical_values, L_i, L_2


def test(vanal_u,models,Nx=1,Nt=1,M=1,Mx=None,Mt=None,Qx=1,Qt=1,w=None,x0=0,x1=1,t0=0,t1=1,test_Qx=None,test_Qt=None,block=0):
    L = x1 - x0
    tf = t1 - t0

    if Mx is not None and Mt is not None:
        M = Mx * Mt

    epsilon = []
    true_values = []
    numerical_values = []

    if test_Qx is None:
        test_Qx = 10 * Qx
    if test_Qt is None:
        test_Qt = 10 * Qt

    tshift = block * tf
    
    for k in range(Nx):
        epsilon_x = []
        true_value_x = []
        numerical_value_x = []
        for n in range(Nt):
            # forward and grad
            x_min = L/Nx * k + x0
            x_max = L/Nx * (k+1) + x0
            x_devide = np.linspace(x_min, x_max, test_Qx + 1)[:test_Qx]
            t_min = tf/Nt * n
            t_max = tf/Nt * (n+1)
            t_devide = np.linspace(t_min, t_max, test_Qt + 1)[:test_Qt]
            grid = np.array(list(itertools.product(x_devide,t_devide))).reshape(test_Qx,test_Qt,2)
            test_point = torch.tensor(grid,requires_grad=True)
            in_ = test_point.detach().numpy()
            true_value = vanal_u(in_[:,:,0],in_[:,:,1] + tshift)
            values = models[k][n](test_point).detach().numpy()
            numerical_value = np.dot(values, w[k*Nt*M + n*M : k*Nt*M + n*M + M,:]).reshape(test_Qx,test_Qt)
            e = np.abs(true_value - numerical_value)
            true_value_x.append(true_value)
            numerical_value_x.append(numerical_value)
            epsilon_x.append(e)
        epsilon_x = np.concatenate(epsilon_x, axis=1)
        epsilon.append(epsilon_x)
        true_value_x = np.concatenate(true_value_x, axis=1)
        true_values.append(true_value_x)
        numerical_value_x = np.concatenate(numerical_value_x, axis=1)
        numerical_values.append(numerical_value_x)
    epsilon = np.concatenate(epsilon, axis=0)
    true_values = np.concatenate(true_values, axis=0)
    numerical_values = np.concatenate(numerical_values, axis=0)
    e = epsilon.reshape((-1,1))
    L_i = L_inf_error(e)
    L_2 = L_2_error(e)
    print('********************* ERROR *********************')
    print(f"Block={block:d},t0={tshift:.2f},t1={tshift + tf:.2f}")
    if Mx is not None and Mt is not None:
        print('Nx={:d},Nt={:d},Mx={:d},Mt={:d},Qx={:d},Qt={:d}'.format(Nx,Nt,Mx,Mt,Qx,Qt))
    else:
        print('Nx={:d},Nt={:d},M={:d},Qx={:d},Qt={:d}'.format(Nx,Nt,M,Qx,Qt))
    print('L_inf={:.2e}'.format(L_i),'L_2={:.2e}'.format(L_2))
    print("边值条件误差")
    print("{:.2e} {:.2e}".format(max(epsilon[0,:]),max(epsilon[-1,:])))
    print("初值、终值误差")
    print("{:.2e} {:.2e}".format(max(epsilon[:,0]),max(epsilon[:,-1])))
    # np.save('./epsilon_psi2.npy',epsilon)

    return true_values, numerical_values, L_i, L_2


def test_psib(vanal_u,models,Nx=1,Nt=1,M=1,Mx=None,Mt=None,Qx=1,Qt=1,w=None,x0=0,x1=1,t0=0,t1=1,test_Qx=None,test_Qt=None,block=0):
    
    L = x1 - x0
    tf = t1 - t0

    if Mx is not None and Mt is not None:
        M = Mx * Mt

    epsilon = []
    true_values = []
    numerical_values = []

    if test_Qx is None:
        test_Qx = 10 * Qx
    if test_Qt is None:
        test_Qt = 10 * Qt

    tshift = block * tf
    
    for k in range(Nx):
        epsilon_x = []
        true_value_x = []
        numerical_value_x = []
        for n in range(Nt):
            # forward and grad
            x_min = L/Nx * k + x0
            x_max = L/Nx * (k+1) + x0
            x_devide = np.linspace(x_min, x_max, test_Qx + 1)[:test_Qx]
            t_min = tf/Nt * n
            t_max = tf/Nt * (n+1)
            t_devide = np.linspace(t_min, t_max, test_Qt + 1)[:test_Qt]
            grid = np.array(list(itertools.product(x_devide,t_devide))).reshape(test_Qx,test_Qt,2)
            test_point = torch.tensor(grid,requires_grad=True)
            in_ = test_point.detach().numpy()
            true_value = vanal_u(in_[:,:,0],in_[:,:,1] + tshift)
            numerical_value = np.zeros_like(true_value)

            if n > 0:
                n_0 = n - 1
            else:
                n_0 = n

            if n < Nt - 1:
                n_1 = n + 2
            else:
                n_1 = n + 1

            for n_s in range(n_0, n_1):
                values = models[k][n_s](test_point).detach().numpy()
                M0 = (k * Nt + n_s) * M
                numerical_value += np.dot(values, w[M0 : M0 + M,:]).reshape(test_Qx,test_Qt)

            e = np.abs(true_value - numerical_value)
            true_value_x.append(true_value)
            numerical_value_x.append(numerical_value)
            epsilon_x.append(e)
        epsilon_x = np.concatenate(epsilon_x, axis=1)
        epsilon.append(epsilon_x)
        true_value_x = np.concatenate(true_value_x, axis=1)
        true_values.append(true_value_x)
        numerical_value_x = np.concatenate(numerical_value_x, axis=1)
        numerical_values.append(numerical_value_x)
    epsilon = np.concatenate(epsilon, axis=0)
    true_values = np.concatenate(true_values, axis=0)
    numerical_values = np.concatenate(numerical_values, axis=0)
    e = epsilon.reshape((-1,1))
    L_i = L_inf_error(e)
    L_2 = L_2_error(e)
    print('********************* ERROR *********************')
    print(f"Block={block:d},t0={tshift:.2f},t1={tshift + tf:.2f}")
    if Mx is not None and Mt is not None:
        print('Nx={:d},Nt={:d},Mx={:d},Mt={:d},Qx={:d},Qt={:d}'.format(Nx,Nt,Mx,Mt,Qx,Qt))
    else:
        print('Nx={:d},Nt={:d},M={:d},Qx={:d},Qt={:d}'.format(Nx,Nt,M,Qx,Qt))
    print('L_inf={:.2e}'.format(L_i),'L_2={:.2e}'.format(L_2))
    print("边值条件误差")
    print("{:.2e} {:.2e}".format(max(epsilon[0,:]),max(epsilon[-1,:])))
    print("初值、终值误差")
    print("{:.2e} {:.2e}".format(max(epsilon[:,0]),max(epsilon[:,-1])))
    # np.save('./epsilon_psi2.npy',epsilon)

    return true_values, numerical_values, L_i, L_2

def record(main, eqname, recursive_times=1, tf=1.0, time_blocks=1, Nxs=1, Nts=1, Ms=1, Mxs=1, Mts=1, Qxs=1, Qts=1, eqn="advect", method="rfm"):

    if method == "rfm":
        loop = zip(Nxs, Nts, Ms, Qxs, Qts, time_blocks)
    elif method == "mul" or method == "smul":
        loop = zip(Nxs, Nts, Mxs, Mts, Qxs, Qts, time_blocks)
    else:
        raise NotImplementedError

    for i, item in enumerate(loop):

        if method == "rfm":
            Nx, Nt, M, Qx, Qt, time_block = item
        elif method == "mul" or method == "smul":
            Nx, Nt, Mx, Mt, Qx, Qt, time_block = item
        
        L_i_res = list()
        L_2_res = list()
        e_i_res = list()
        e_2_res = list()
        result = []

        for k in range(recursive_times):

            plot = (k == 0)

            if method == "rfm":
                
                print(f"[Iter={k+1:d}/{recursive_times:d}] Nx={Nx:d}, Nt={Nt:d}, M={M:d}, Qx={Qx:d}, Qt={Qt:d}, time_block={time_block:d}")

                img_save_dir = f"outputs/{eqname}_rfm_tf={tf:.2f}-time_block={time_block:d}_Nx={Nx:d}_Nt={Nt:d}_M={M:d}_Qx={Qx:d}_Qt={Qt:d}/"

            elif method == "mul":

                print(f"[Iter={k+1:d}/{recursive_times:d}] Nx={Nx:d}, Nt={Nt:d}, Mx={Mx:d}, Mt={Mt:d}, Qx={Qx:d}, Qt={Qt:d}, time_block={time_block:d}")

                img_save_dir = f"outputs/{eqname}_mul_tf={tf:.2f}-time_block={time_block:d}_Nx={Nx:d}_Nt={Nt:d}_Mx={Mx:d}_Mt={Mt:d}_Qx={Qx:d}_Qt={Qt:d}/"

            elif method == "smul":

                print(f"[Iter={k+1:d}/{recursive_times:d}] Nx={Nx:d}, Nt={Nt:d}, Mx={Mx:d}, Mt={Mt:d}, Qx={Qx:d}, Qt={Qt:d}, time_block={time_block:d}")

                img_save_dir = f"outputs/{eqname}_smul_tf={tf:.2f}-time_block={time_block:d}_Nx={Nx:d}_Nt={Nt:d}_Mx={Mx:d}_Mt={Mt:d}_Qx={Qx:d}_Qt={Qt:d}/"

            if not os.path.exists(img_save_dir):
                os.makedirs(img_save_dir, exist_ok=True)

            if method == "rfm":
                L_is, L_2s, e_i, e_2 = main(Nx=Nx,Nt=Nt,M=M,tf=tf,time_block=time_block,Qx=Qx,Qt=Qt,plot=plot,img_save_dir=img_save_dir)
            elif method == "mul":
                L_is, L_2s, e_i, e_2 = main(Nx=Nx,Nt=Nt,Mx=Mx,Mt=Mt,tf=tf,time_block=time_block,Qx=Qx,Qt=Qt,plot=plot,img_save_dir=img_save_dir)
            elif method == "smul":
                L_is, L_2s, e_i, e_2 = main(Nx=Nx,Nt=Nt,Mx=Mx,Mt=Mt,tf=tf,time_block=time_block,Qx=Qx,Qt=Qt,plot=plot,img_save_dir=img_save_dir)

            L_i_res.append(L_is)
            L_2_res.append(L_2s)
            e_i_res.append(np.stack(e_i, axis=0))
            e_2_res.append(np.stack(e_2, axis=0))

            L_i = L_inf_error(L_is)
            L_2 = L_2_error(L_2s)

            result.append([L_i, L_2])

        L_i_res = np.array(L_i_res)
        L_2_res = np.array(L_2_res)
        e_i_res = np.array(e_i_res).reshape(recursive_times, time_block, Nt, -1)
        e_2_res = np.array(e_2_res).reshape(recursive_times, time_block, Nt, -1)

        result = np.array(result)

        fm = "a"

        with open(f"logs/{eqn}_{method}_tf={tf:.2f}.log", fm) as f:

            print("time: ", time.strftime("%m-%d %H:%M"), file=f)

            print("-" * 20, file=f)

            if method == "rfm":
                print(f"Nx={Nx:d}, Nt={Nt:d}, M={M:d}, Qx={Qx:d}, Qt={Qt:d}, t_block={time_block:d}", file=f)
            elif method == "mul":
                print(f"Nx={Nx:d}, Nt={Nt:d}, Mx={Mx:d}, Mt={Mt:d}, Qx={Qx:d}, Qt={Qt:d}, t_block={time_block:d}", file=f)
            elif method == "smul":
                print(f"Nx={Nx:d}, Nt={Nt:d}, Mx={Mx:d}, Mt={Mt:d}, Qx={Qx:d}, Qt={Qt:d}, t_block={time_block:d}", file=f)
            else:
                raise ValueError

            print("Mean L_i: {:.2e}, Std L_i: {:.2e}".format(result[:, 0].mean(), result[:, 0].std()), file=f)
            print("Mean L_2: {:.2e}, Std L_2: {:.2e}".format(result[:, 1].mean(), result[:, 1].std()), file=f)
            
            print("-" * 20, file=f)

            np.set_printoptions(formatter={'float': '{:.2e}'.format})

            print(f"Mean L_i per block: {L_i_res.mean(0)}", file=f)
            print(f"Std L_i per block: {L_i_res.std(0)}", file=f)
            print(f"Mean L_2 per block: {L_2_res.mean(0)}", file=f)
            print(f"Std L_2 per block: {L_2_res.std(0)}", file=f)
            
            print("-" * 20, file=f)

            e_i_initial = e_i_res[:, :, :, 0]
            e_i_final = e_i_res[:, :, :, -1]
            print(f"Mean initial L_i per block: {e_i_initial.mean(0)}", file=f)
            print(f"Mean final L_i per block: {e_i_final.mean(0)}", file=f)

            e_2_initial = e_2_res[:, :, :, 0]
            e_2_final = e_2_res[:, :, :, -1]
            print(f"Mean initial L_2 per block: {e_2_initial.mean(0)}", file=f)
            print(f"Mean final L_2 per block: {e_2_final.mean(0)}", file=f)
            
            print("#" * 40, file=f)
            print(file=f)
            print(file=f)
            
        visualize_time_stats(tf=tf, datas=[e_i_res.reshape(recursive_times, -1), e_2_res.reshape(recursive_times, -1)], labels=[r"$L_i$", r"$L_2$"], savename=os.path.join(img_save_dir, f"error_tb={time_block:d}.png"))

        if method == "rfm":
            np.save(f"outputs/{eqname}_rfm_tf={tf:.2f}-time_block={time_block:d}_Nx={Nx:d}_Nt={Nt:d}_M={M:d}_Qx={Qx:d}_Qt={Qt:d}/e_i.npy", e_i)
            np.save(f"outputs/{eqname}_rfm_tf={tf:.2f}-time_block={time_block:d}_Nx={Nx:d}_Nt={Nt:d}_M={M:d}_Qx={Qx:d}_Qt={Qt:d}/e_2.npy", e_2)
    
        elif method == "mul" or method == "smul":
            np.save(f"outputs/{eqname}_{method}_tf={tf:.2f}-time_block={time_block:d}_Nx={Nx:d}_Nt={Nt:d}_Mx={Mx:d}_Mt={Mt:d}_Qx={Qx:d}_Qt={Qt:d}/e_i.npy", e_i)
            np.save(f"outputs/{eqname}_{method}_tf={tf:.2f}-time_block={time_block:d}_Nx={Nx:d}_Nt={Nt:d}_Mx={Mx:d}_Mt={Mt:d}_Qx={Qx:d}_Qt={Qt:d}/e_2.npy", e_2)