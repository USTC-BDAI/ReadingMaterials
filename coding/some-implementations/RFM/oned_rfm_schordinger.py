# -*- coding: utf-8 -*-
"""
Created on Wed Dec 8 2021

@author: Askeladd
"""
import os
import sys
import time
import numpy as np
import torch

import utils
import rfm

from equations.schordinger import Schordinger

torch.set_default_dtype(torch.float64)

eqn = Schordinger()

vanal_ur = eqn.vanal_ur
vanal_ui = eqn.vanal_ui
vanal_gr = eqn.vanal_gr
vanal_gi = eqn.vanal_gi
vanal_fr = eqn.vanal_fr
vanal_fi = eqn.vanal_fi


def cal_matrix(models,points,Nx,Nt,M,Qx,Qt,initial_gr=None,initial_gi=None,tshift=0):

    # matrix define (Aw=b)
    Ae_i = np.zeros([Nx*Nt*Qx*Qt,2*Nx*Nt*M]) # i ur_t + a i ui_xx = 0
    fe_i = np.zeros([Nx*Nt*Qx*Qt,1])

    Ae_r = np.zeros([Nx*Nt*Qx*Qt,2*Nx*Nt*M]) # i i ui_t + a ur_xx = 0
    fe_r = np.zeros([Nx*Nt*Qx*Qt,1])

    Ai_ur = np.zeros([Nx*Qx,2*Nx*Nt*M]) # ur(x, 0) = gr(x)
    fi_ur = np.zeros([Nx*Qx,1])

    Ai_ui = np.zeros([Nx*Qx,2*Nx*Nt*M]) # ui(x, 0) = gi(x)
    fi_ui = np.zeros([Nx*Qx,1])
    
    Ab_ur_1 = np.zeros([Nt*Qt,2*Nx*Nt*M]) # ur(a_1, t) = ur(b_1, t)
    fb_ur_1 = np.zeros([Nt*Qt,1])
    
    Ab_ui_1 = np.zeros([Nt*Qt,2*Nx*Nt*M]) # ui(a_1, t) = ui(b_1, t)
    fb_ui_1 = np.zeros([Nt*Qt,1])
    
    Ab_ur_2 = np.zeros([Nt*Qt,2*Nx*Nt*M]) # ur_x(a_1, t) = ur_x(b_1, t)
    fb_ur_2 = np.zeros([Nt*Qt,1])
    
    Ab_ui_2 = np.zeros([Nt*Qt,2*Nx*Nt*M]) # ui_x(a_1, t) = ui_x(b_1, t)
    fb_ui_2 = np.zeros([Nt*Qt,1])

    Ac_ur_0t = np.zeros([(Nt - 1)*Qx*Nx, 2*Nx*Nt*M]) # C^0 continuity on t of ur
    fc_ur_0t = np.zeros([(Nt - 1)*Qx*Nx, 1])

    Ac_ui_0t = np.zeros([(Nt - 1)*Qx*Nx, 2*Nx*Nt*M]) # C^0 continuity on t of ui
    fc_ui_0t = np.zeros([(Nt - 1)*Qx*Nx, 1])

    Ac_ur_0x = np.zeros([(Nx - 1)*Qt*Nt, 2*Nx*Nt*M]) # C^0 continuity on x of ur
    fc_ur_0x = np.zeros([(Nx - 1)*Qt*Nt, 1])

    Ac_ui_0x = np.zeros([(Nx - 1)*Qt*Nt, 2*Nx*Nt*M]) # C^0 continuity on x of ui
    fc_ui_0x = np.zeros([(Nx - 1)*Qt*Nt, 1])

    Ac_ur_1x = np.zeros([(Nx - 1)*Qt*Nt, 2*Nx*Nt*M]) # C^1 continuity on x of ur
    fc_ur_1x = np.zeros([(Nx - 1)*Qt*Nt, 1])

    Ac_ui_1x = np.zeros([(Nx - 1)*Qt*Nt, 2*Nx*Nt*M]) # C^1 continuity on x of ui
    fc_ui_1x = np.zeros([(Nx - 1)*Qt*Nt, 1])

    M_gap = Nx * Nt * M
    
    for k in range(Nx):
        for n in range(Nt):
            
            in_ = points[k][n].detach().numpy()
            out = models[k][n](points[k][n])
            values = out.detach().numpy()
            M_begin = (k * Nt + n) * M

            if n == 0:
                if initial_gr is None:
                    fi_ur[k*Qx : (k+1)*Qx, :] = vanal_gr(in_[:Qx, 0, 0]).reshape((Qx,1))
                else:
                    fi_ur[k*Qx : (k+1)*Qx, :] = initial_gr[k*Qx: (k+1)*Qx, :]

                if initial_gi is None:
                    fi_ui[k*Qx : (k+1)*Qx, :] = vanal_gi(in_[:Qx, 0, 0]).reshape((Qx,1))
                else:
                    fi_ui[k*Qx : (k+1)*Qx, :] = initial_gi[k*Qx: (k+1)*Qx, :]
            
            f_in = in_[:Qx,:Qt,:].reshape((-1,2))
            fe_r[k*Nt*Qx*Qt + n*Qx*Qt: k*Nt*Qx*Qt + n*Qx*Qt + Qx*Qt, :] = vanal_fr(f_in[:,0],f_in[:,1] + tshift).reshape(-1,1)
            fe_i[k*Nt*Qx*Qt + n*Qx*Qt: k*Nt*Qx*Qt + n*Qx*Qt + Qx*Qt, :] = vanal_fi(f_in[:,0],f_in[:,1] + tshift).reshape(-1,1)
            
            # partial differential equation
            grads = []
            grads_2_xx = []
            for i in range(M):
                g_1 = torch.autograd.grad(outputs=out[:,:,i], inputs=points[k][n],
                                        grad_outputs=torch.ones_like(out[:,:,i]),
                                        create_graph = True, retain_graph = True)[0]
                grads.append(g_1.squeeze().detach().numpy())
                
                g_2_x = torch.autograd.grad(outputs=g_1[:,:,0], inputs=points[k][n],
                                    grad_outputs=torch.ones_like(out[:,:,i]),
                                    create_graph = True, retain_graph = True)[0]
                grads_2_xx.append(g_2_x[:,:,0].squeeze().detach().numpy())
                
            grads = np.array(grads).swapaxes(0,3)
            
            # print(values.shape,grads.shape)

            grads_t = grads[1, :Qx, :Qt, :].reshape(-1, M)
            # grads_t = grads_t.reshape((-1, M))

            grads_2_xx = np.array(grads_2_xx)
            grads_2_xx = grads_2_xx[:,:Qx,:Qt]
            grads_2_xx = grads_2_xx.transpose(1,2,0).reshape(-1,M)
            
            # i ur_t + a i ui_xx = 0
            Ae_i[k*Nt*Qx*Qt + n*Qx*Qt : k*Nt*Qx*Qt + n*Qx*Qt + Qx*Qt, M_begin : M_begin + M] = grads_t
            Ae_i[k*Nt*Qx*Qt + n*Qx*Qt : k*Nt*Qx*Qt + n*Qx*Qt + Qx*Qt, M_begin + M_gap : M_begin + M_gap + M] = eqn.coef_a * grads_2_xx
            
            # i i ui_t + a ur_xx = 0
            Ae_r[k*Nt*Qx*Qt + n*Qx*Qt : k*Nt*Qx*Qt + n*Qx*Qt + Qx*Qt, M_begin + M_gap : M_begin + M_gap + M] = - grads_t
            Ae_r[k*Nt*Qx*Qt + n*Qx*Qt : k*Nt*Qx*Qt + n*Qx*Qt + Qx*Qt, M_begin : M_begin + M] = eqn.coef_a * grads_2_xx
            
            # initial conditions
            if n == 0:
                # ur(x, 0) = gr(x)
                Ai_ur[k*Qx : k*Qx+Qx, M_begin : M_begin + M] = values[:Qx,0,:]

                # ui(x, 0) = gi(x)
                Ai_ui[k*Qx : k*Qx+Qx, M_begin + M_gap : M_begin + M_gap + M] = values[:Qx,0,:]

            # periodical boundary conditions
            if k == 0:
                # ur(a_1, t) = ur(b_1, t)
                Ab_ur_1[n*Qt : (n+1)*Qt, M_begin : M_begin + M] += values[0, :Qt, :]

                # ui(a_1, t) = ui(b_1, t)
                Ab_ui_1[n*Qt : (n+1)*Qt, M_begin + M_gap : M_begin + M_gap + M] += values[0, :Qt, :]

                # ur_x(a_1, t) = ur_x(b_1, t)
                Ab_ur_2[n*Qt : (n+1)*Qt, M_begin : M_begin + M] += grads[0, 0, :Qt, :]

                # ui_x(a_1, t) = ui_x(b_1, t)
                Ab_ui_2[n*Qt : (n+1)*Qt, M_begin + M_gap : M_begin + M_gap + M] += grads[0, 0, :Qt, :]

            if k == Nx - 1:
                # ur(a_1, t) = ur(b_1, t)
                Ab_ur_1[n*Qt : (n+1)*Qt, M_begin : M_begin + M] -= values[-1, :Qt, :]

                # ui(a_1, t) = ui(b_1, t)
                Ab_ui_1[n*Qt : (n+1)*Qt, M_begin + M_gap : M_begin + M_gap + M] -= values[-1, :Qt, :]

                # ur_x(a_1, t) = ur_x(b_1, t)
                Ab_ur_2[n*Qt : (n+1)*Qt, M_begin : M_begin + M] -= grads[0, -1, :Qt, :]

                # ui_x(a_1, t) = ui_x(b_1, t)
                Ab_ui_2[n*Qt : (n+1)*Qt, M_begin + M_gap : M_begin + M_gap + M] -= grads[0, -1, :Qt, :]

            # C0 continuity on t for ur, ui
            if n > 0:
                Ac_ur_0t[(n - 1)*Qx*Nx + k*Qx: (n - 1)*Qx*Nx + k*Qx+Qx, M_begin: M_begin + M] = values[:Qx, 0, :]
                Ac_ui_0t[(n - 1)*Qx*Nx + k*Qx: (n - 1)*Qx*Nx + k*Qx+Qx, M_begin + M_gap: M_begin + M_gap + M] = values[:Qx, 0, :]
            if n < Nt - 1:
                Ac_ur_0t[n*Qx*Nx + k*Qx: n*Qx*Nx + k*Qx+Qx, M_begin: M_begin + M] = -values[:Qx, -1, :]
                Ac_ui_0t[n*Qx*Nx + k*Qx: n*Qx*Nx + k*Qx+Qx, M_begin + M_gap: M_begin + M_gap + M] = -values[:Qx, -1, :]

            # C0 continuity on x for ur, ui
            if k > 0:
                Ac_ur_0x[(k-1)*Qt*Nt + n*Qt: (k-1)*Qt*Nt + (n+1)*Qt, M_begin: M_begin + M] = values[0, :Qt, :]
                Ac_ui_0x[(k-1)*Qt*Nt + n*Qt: (k-1)*Qt*Nt + (n+1)*Qt, M_begin + M_gap: M_begin + M_gap + M] = values[0, :Qt, :]
            if k < Nx - 1:
                Ac_ur_0x[k*Nt*Qt + n*Qt: k*Nt*Qt + (n+1)*Qt, M_begin: M_begin + M] = -values[-1, :Qt, :]
                Ac_ui_0x[k*Nt*Qt + n*Qt: k*Nt*Qt + (n+1)*Qt, M_begin + M_gap: M_begin + M_gap + M] = -values[-1, :Qt, :]

            # C1 continuity on x for ur, ui
            if k > 0:
                Ac_ur_1x[(k-1)*Qt*Nt + n*Qt: (k-1)*Qt*Nt + (n+1)*Qt, M_begin: M_begin + M] = grads[0, 0, :Qt, :]
                Ac_ui_1x[(k-1)*Qt*Nt + n*Qt: (k-1)*Qt*Nt + (n+1)*Qt, M_begin + M_gap: M_begin + M_gap + M] = grads[0, 0, :Qt, :]
            if k < Nx - 1:
                Ac_ur_1x[k*Nt*Qt + n*Qt: k*Nt*Qt + (n+1)*Qt, M_begin: M_begin + M] = -grads[0, -1, :Qt, :]
                Ac_ui_1x[k*Nt*Qt + n*Qt: k*Nt*Qt + (n+1)*Qt, M_begin + M_gap: M_begin + M_gap + M] = -grads[0, -1, :Qt, :]
            

    A = np.concatenate((Ae_r, Ae_i, Ai_ur, Ai_ui, Ab_ur_1, Ab_ui_1, Ab_ur_2, Ab_ui_2, Ac_ur_0t, Ac_ui_0t, Ac_ur_0x, Ac_ui_0x, Ac_ur_1x, Ac_ui_1x), axis=0)
    f = np.concatenate((fe_r, fe_i, fi_ur, fi_ui, fb_ur_1, fb_ui_1, fb_ur_2, fb_ui_2, fc_ur_0t, fc_ui_0t, fc_ur_0x, fc_ui_0x, fc_ur_1x, fc_ui_1x), axis=0)

    # input((A.shape, f.shape))
    
    return(A,f)


def main(Nx,Nt,M,Qx,Qt,tf,time_block=1,plot = False,moore = False, img_save_dir=None):
    # prepare models and collocation pointss

    tlen = tf / time_block

    models, points = rfm.pre_define_rfm(Nx=Nx, Nt=Nt, M=M, Qx=Qx, Qt=Qt, x0=eqn.a1, x1=eqn.b1, t0=0, t1=tlen)

    initial_gr = rfm.get_anal_u(vanal_u=vanal_gr, points=points, Nx=Nx, Qx=Qx, nt=0, qt=0)
    initial_gi = rfm.get_anal_u(vanal_u=vanal_gi, points=points, Nx=Nx, Qx=Qx, nt=0, qt=0)

    ur_true_values = list()
    ur_numerical_values = list()
    ui_true_values = list()
    ui_numerical_values = list()
    ur_L_is = list()
    ur_L_2s = list()
    ur_e_i = list()
    ur_e_2 = list()
    ui_L_is = list()
    ui_L_2s = list()
    ui_e_i = list()
    ui_e_2 = list()

    M_unit = Nx * Nt * M

    for t in range(time_block):
    
        # matrix define (Aw=b)
        A,f = cal_matrix(models,points,Nx,Nt,M,Qx,Qt,initial_gr=initial_gr,initial_gi=initial_gi,tshift=t * tlen)

        w = rfm.solve_lst_square(A, f, moore=moore)

        final_ur = rfm.get_num_u(models=models, points=points, w=w[:M_unit], Nx=Nx, Nt=Nt, M=M, Qx=Qx, Qt=Qt, nt=Nt-1, qt=Qt)
        final_ui = rfm.get_num_u(models=models, points=points, w=w[M_unit:], Nx=Nx, Nt=Nt, M=M, Qx=Qx, Qt=Qt, nt=Nt-1, qt=Qt)

        # anal_u = rfm.get_anal_u(vanal_u=vanal_u, points=points, Nx=Nx, Qx=Qx, nt=Nt-1, qt=Qt, tshift=t*tlen)
        # anal_ut = rfm.get_anal_u(vanal_u=vanal_psi, points=points, Nx=Nx, Qx=Qx, nt=Nt-1, qt=Qt, tshift=t*tlen)

        # print(utils.L_inf_error(final_u - anal_u))
        # print(utils.L_inf_error(final_ut - anal_ut))

        initial_gr = final_ur
        initial_gi = final_ui

        # initial_u = anal_u
        # initial_ut = anal_ut

        # evaluate ur
        true_values_, numerical_values_, L_i, L_2 = utils.test(vanal_u=vanal_ur, models=models, \
            Nx=Nx, Nt=Nt, M=M, Qx=Qx, Qt=Qt, \
            w=w[:M_unit], \
            x0=eqn.a1, x1=eqn.b1, t0=0, t1=tlen, block=t)

        ur_true_values.append(true_values_)
        ur_numerical_values.append(numerical_values_)
        ur_L_is.append(L_i)
        ur_L_2s.append(L_2)

        t_unit = true_values_.shape[1] // Nt

        true_slices = np.concatenate([true_values_[:, ::t_unit], true_values_[:, -1:]], axis=1)
        nume_slices = np.concatenate([numerical_values_[:, ::t_unit], numerical_values_[:, -1:]], axis=1)

        ur_e_i.append(utils.L_inf_error(true_slices - nume_slices, axis=0))
        ur_e_2.append(utils.L_2_error(true_slices - nume_slices, axis=0))

        # evaluate ui
        true_values_, numerical_values_, L_i, L_2 = utils.test(vanal_u=vanal_ui, models=models, \
            Nx=Nx, Nt=Nt, M=M, Qx=Qx, Qt=Qt, \
            w=w[M_unit:], \
            x0=eqn.a1, x1=eqn.b1, t0=0, t1=tlen, block=t)

        ui_true_values.append(true_values_)
        ui_numerical_values.append(numerical_values_)
        ui_L_is.append(L_i)
        ui_L_2s.append(L_2)

    ur_true_values = np.concatenate(ur_true_values, axis=1)[:, ::time_block]
    ur_numerical_values = np.concatenate(ur_numerical_values, axis=1)[:, ::time_block]
    ur_L_is = np.array(ur_L_is)
    ur_L_2s = np.array(ur_L_2s)
    ur_e_i = utils.L_inf_error(ur_true_values - ur_numerical_values, axis=0).reshape(time_block, Nt, -1)
    ur_e_2 = utils.L_2_error(ur_true_values - ur_numerical_values, axis=0).reshape(time_block, Nt, -1)

    ui_true_values = np.concatenate(ui_true_values, axis=1)[:, ::time_block]
    ui_numerical_values = np.concatenate(ui_numerical_values, axis=1)[:, ::time_block]
    ui_L_is = np.array(ui_L_is)
    ui_L_2s = np.array(ui_L_2s)
    ui_e_i = utils.L_inf_error(ui_true_values - ui_numerical_values, axis=0).reshape(time_block, Nt, -1)
    ui_e_2 = utils.L_2_error(ui_true_values - ui_numerical_values, axis=0).reshape(time_block, Nt, -1)

    np.save(os.path.join(img_save_dir, "ur_sol.npy"), ur_true_values)
    np.save(os.path.join(img_save_dir, "ur_num.npy"), ur_numerical_values)
    np.save(os.path.join(img_save_dir, "ui_sol.npy"), ui_true_values)
    np.save(os.path.join(img_save_dir, "ui_num.npy"), ui_numerical_values)
    
    # visualize
    if plot:

        utils.visualize(ur_true_values, ur_numerical_values, x0=eqn.a1, x1=eqn.b1, t0=0, t1=tf, savedir=img_save_dir, eqname=f"{eqn.name}-ur", mename="rfm", img_format="png")
        utils.visualize(ui_true_values, ui_numerical_values, x0=eqn.a1, x1=eqn.b1, t0=0, t1=tf, savedir=img_save_dir, eqname=f"{eqn.name}-ui", mename="rfm", img_format="png")

    return ur_L_is, ur_L_2s, ur_e_i, ur_e_2, ui_L_is, ui_L_2s, ui_e_i, ui_e_2



if __name__ == '__main__':
    # set_seed(100)

    if len(sys.argv) > 1:
        n = int(sys.argv[1])
    else:
        n = 1

    if len(sys.argv) > 2:
        tf = float(sys.argv[2])
    else:
        tf = 10.0


    time_blocks = [3,]
    Nxs = [5,]
    Nts = [1,]
    Ms = [100,]
    Qxs = [30,]
    Qts = [30,]

    for i, (Nx, Nt, M, Qx, Qt, time_block) in enumerate(zip(Nxs, Nts, Ms, Qxs, Qts, time_blocks)):

        img_save_dir = f"outputs/{eqn.name}_rfm_tf={tf:.2f}-time_block={time_block:d}_Nx={Nx:d}_Nt={Nt:d}_M={M:d}_Qx={Qx:d}_Qt={Qt:d}/"

        if not os.path.exists(img_save_dir):
            os.makedirs(img_save_dir, exist_ok=True)
        
        for k in range(n):
            print(f"[Iter={k+1:d}/{n:d}] Nx={Nx:d}, Nt={Nt:d}, M={M:d}, Qx={Qx:d}, Qt={Qt:d}, time_block={time_block:d}")
            plot = (k == 0)
            _ = main(Nx=Nx,Nt=Nt,M=M,Qx=Qx,Qt=Qt,tf=tf,time_block=time_block,plot=plot,img_save_dir=img_save_dir)
