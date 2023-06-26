import sys, os
sys.path.append(os.path.join(sys.path[0], "code"))

import pytorch_models as models
import pytorch_samplers as samplers
import pytorch_losses as losses
import pytorch_training as training
import pytorch_utils as utils
import sinkhorn_cnf as cnf
from geomloss import SamplesLoss
from torchdiffeq import odeint as odeint

import importlib
import time
import json
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import torch
import random
import numpy as np
import statistics as stats

importlib.reload(models)
importlib.reload(losses)
importlib.reload(samplers)
importlib.reload(training)
importlib.reload(utils)

torch.set_default_dtype(torch.float64)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def run_experiment(rho_list, json_fname, k, A_model, use_A):
    """Runs scRNA trajectory inference experiment."""
    targets = rho_list[::k]
    W1_vals = {}
    sinkhorn_w1_loss = SamplesLoss(loss="sinkhorn", p=1, blur=1e-6)
    # Define hyperparams
    n_samples = 1000
    step_size = 1/60
    space_dims = 2
    if use_A:
        lambd = 1e-1 # NOTE CHANGE TO 1e-1 FOR IDENTITY MATRIX EXPERIMENT, 1e1 for old A experiment
    else:
        lambd = 0
    lr = 1e-3
    weight_decay = 1e-3
    n_epochs = 10000
    A = A_model
    time_varying = True
    hidden_dims = 64
    for i in range(len(targets) - 1):
        torch.cuda.empty_cache()
        base_sampler = targets[i]
        target_sampler_list = [targets[i+1]]
        advected_samples, trained_v, losses = cnf.sinkhorn_cnf(base_sampler, # should have a .sample method
                 target_sampler_list, # each element should have a .sample method
                 n_samples, # number of samples drawn from each sampler in each epoch of training
                 step_size, # size of time step between consecutive advected samples -- 1/step_size should be an integer
                 space_dims,
                 lambd, # controls strength of kinetic energy loss
                 lr,
                 weight_decay,
                 n_epochs,
                 A, # metric tensor as Pytorch nn.Module
                 time_varying, # is the velocity field time-varying?
                 hidden_dims,
                 None
                )
        # Save the trained model
        if use_A:
            fname = "trained_v_with_A_lambd_1em1_use_every_" + str(k) + "_i_is_" + str(i) + ".pt"
            fpath = "trained_models/scrna_vel_fields/with_A/" + fname
            fpath = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), fpath))
            torch.save(trained_v.state_dict(), fpath)
        else:
            fname = "trained_v_no_A_use_every_" + str(k) + "_i_is_" + str(i) + ".pt"
            fpath = "trained_models/scrna_vel_fields/no_A/" + fname
            fpath = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), fpath))
            torch.save(trained_v.state_dict(), fpath)
        # Compute W1 distances between advected and left out samples
        for j in range(1,k):
            model_samples = advected_samples[int((j/k)*60)]
            true_sampler = rho_list[k*i + j]
            true_samples = true_sampler.sample(n_samples)
            w1 = sinkhorn_w1_loss(model_samples, true_samples)
            key = "tp " + str(k*i + j)
            W1_vals[key] = float(w1.detach().cpu().numpy())
        print(W1_vals)
    # Write W1_vals to json before terminating
    with open(json_fname, 'w') as f:
        json.dump(W1_vals, f)
    return W1_vals

def run_final_tp_experiment(rho_list, json_fname, k, A_model, use_A):
    l = [i for i in range(len(rho_list))]
    old_target_tps = l[::k]
    old_targets = rho_list[::k]
    W1_vals = {}
    sinkhorn_w1_loss = SamplesLoss(loss="sinkhorn", p=1, blur=1e-6)
    # Define hyperparams
    n_samples = 1000
    step_size = 1/60
    space_dims = 2
    if use_A:
        lambd = 1e-1
    else:
        lambd = 0
    lr = 1e-3
    weight_decay = 1e-3
    n_epochs = 10000
    A = A_model
    time_varying = True
    hidden_dims = 64
    torch.cuda.empty_cache()
    base_sampler = old_targets[-1]
    target_sampler_list = [rho_list[-1]]
    advected_samples, trained_v, losses = cnf.sinkhorn_cnf(base_sampler, # should have a .sample method
             target_sampler_list, # each element should have a .sample method
             n_samples, # number of samples drawn from each sampler in each epoch of training
             step_size, # size of time step between consecutive advected samples -- 1/step_size should be an integer
             space_dims,
             lambd, # controls strength of kinetic energy loss
             lr,
             weight_decay,
             n_epochs,
             A, # metric tensor as Pytorch nn.Module
             time_varying, # is the velocity field time-varying?
             hidden_dims,
             None
            )
    # Save the trained model
    if use_A:
        fname = "trained_v_with_A_lambd_1em1_use_every_" + str(k) + "_final_tps.pt"
        fpath = "trained_models/scrna_vel_fields/with_A/" + fname
        fpath = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), fpath))
        torch.save(trained_v.state_dict(), fpath)
    else:
        fname = "trained_v_no_A_use_every_" + str(k) + "_final_tps.pt"
        fpath = "trained_models/scrna_vel_fields/no_A/" + fname
        fpath = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), fpath))
        torch.save(trained_v.state_dict(), fpath)
    # Compute W1 distances between advected and left out samples
    n_tps = len(rho_list) - old_target_tps[-1]
    for j in range(1, n_tps-1):
        model_samples = advected_samples[int((j/n_tps)*60)]
        true_sampler = rho_list[old_target_tps[-1] + j]
        true_samples = true_sampler.sample(n_samples)
        w1 = sinkhorn_w1_loss(model_samples, true_samples)
        key = "tp " + str(old_target_tps[-1] + j)
        W1_vals[key] = float(w1.detach().cpu().numpy())
    print(W1_vals)
    # Write W1_vals to json before terminating
    with open(json_fname, 'w') as f:
        json.dump(W1_vals, f)
    return W1_vals

def compute_avg_W1_samples_all_times(rho_list, k, use_A):
    """Computes avg W1 between ground truth and advected samples across all tp for specified k."""
    sinkhorn_w1_loss = SamplesLoss(loss="sinkhorn", p=1, blur=1e-6)
    targets = rho_list[::k]
    l = [i for i in range(len(rho_list))]
    old_target_tps = l[::k]
    W1_vals = []
    for i in range(len(targets) - 1):
        # Load the velocity field
        if use_A:
            fname = "trained_v_with_A_lambd_1em1_use_every_" + str(k) + "_i_is_" + str(i) + ".pt"
            fpath = "trained_models/scrna_vel_fields/with_A/" + fname
            fpath = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), fpath))
        else:
            fname = "trained_v_no_A_use_every_" + str(k) + "_i_is_" + str(i) + ".pt"
            fpath = "trained_models/scrna_vel_fields/no_A/" + fname
            fpath = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), fpath))
        space_dims = 2
        time_varying = True
        trained_v = cnf.VelocityField(space_dims, time_varying).to(device)
        trained_v.load_state_dict(torch.load(fpath))
        
        # Select base sampler
        base_sampler = targets[i]
        
        # Compute advected samples
        n_samples = 1000 # was 1k
        step_size = 1/60
        method = "midpoint"
        T = 1
        n_times = int(T/step_size) + 1
        times = torch.linspace(0, T, n_times, device=device)
        base_samples = base_sampler.sample(n_samples).to(device)
        base_samples += 0.5*torch.randn(n_samples, space_dims, dtype=torch.float64, device=device)
        advected_samples = odeint(trained_v, base_samples, times, method=method) # advected samples
        
        # Compute W1 vals
        for j in range(1,k):
            model_samples = advected_samples[int((j/k)*60)].detach()
            true_sampler = rho_list[k*i + j]
            true_samples = true_sampler.sample(n_samples)
            w1 = sinkhorn_w1_loss(model_samples, true_samples)
            W1_vals.append(float(w1.detach().cpu().numpy()))
    
    # Now handle final tps
    # Load the velocity field
    # Skip if k=2 and k=19 since there are no final tps in this case
    if k > 2 and k < 19:
        if use_A:
            fname = "trained_v_with_A_lambd_1em1_use_every_" + str(k) + "_final_tps.pt"
            fpath = "trained_models/scrna_vel_fields/with_A/" + fname
            fpath = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), fpath))
        else:
            fname = "trained_v_no_A_use_every_" + str(k) + "_final_tps.pt"
            fpath = "trained_models/scrna_vel_fields/no_A/" + fname
            fpath = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), fpath))
        space_dims = 2
        time_varying = True
        trained_v = cnf.VelocityField(space_dims, time_varying).to(device)
        trained_v.load_state_dict(torch.load(fpath))

        # Select base sampler
        base_sampler = targets[-1]

        # Compute advected samples
        n_samples = 1000 # was 1k
        step_size = 1/60
        method = "midpoint"
        T = 1
        n_times = int(T/step_size) + 1
        times = torch.linspace(0, T, n_times, device=device)
        base_samples = base_sampler.sample(n_samples).to(device)
        base_samples += 0.5*torch.randn(n_samples, space_dims, dtype=torch.float64, device=device)
        advected_samples = odeint(trained_v, base_samples, times, method=method) # advected samples

        # Compute W1 distances between advected and left out samples
        n_tps = len(rho_list) - old_target_tps[-1]
        for j in range(1, n_tps-1):
            model_samples = advected_samples[int((j/n_tps)*60)].detach()
            true_sampler = rho_list[old_target_tps[-1] + j]
            true_samples = true_sampler.sample(n_samples)
            w1 = sinkhorn_w1_loss(model_samples, true_samples)
            W1_vals.append(float(w1.detach().cpu().numpy()))
        
    # Return mean of W1 vals
    return sum(W1_vals)/len(W1_vals)

def compute_multiple_runs(rho_list, k, use_A, n_runs):
    run_list = []
    for i in range(n_runs):
        avg_w1 = compute_avg_W1_samples_all_times(rho_list, k, use_A)
        run_list.append(avg_w1)
    mean = stats.mean(run_list)
    std = stats.stdev(run_list)
    return mean, std

# Specify k value and whether we use A
# Plots advected samples across all time points

def visualize_advected_samples_all_times(ax, rho_list, k, use_A):
    """Plots advected samples across all tp for specified k."""
    targets = rho_list[::k]
    l = [i for i in range(len(rho_list))]
    old_target_tps = l[::k]
    #plt.figure(figsize=(20,20))
    #fig, ax = plt.subplots(figsize=(20,20))
    for i in range(len(targets) - 1):
        # Load the velocity field
        if use_A:
            fname = "trained_v_with_A_lambd_1em1_use_every_" + str(k) + "_i_is_" + str(i) + ".pt"
            fpath = "trained_models/scrna_vel_fields/with_A/" + fname
            fpath = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), fpath))
        else:
            fname = "trained_v_no_A_use_every_" + str(k) + "_i_is_" + str(i) + ".pt"
            fpath = "trained_models/scrna_vel_fields/no_A/" + fname
            fpath = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), fpath))
        space_dims = 2
        time_varying = True
        trained_v = cnf.VelocityField(space_dims, time_varying).to(device)
        trained_v.load_state_dict(torch.load(fpath))
        
        # Select base sampler
        base_sampler = targets[i]
        
        # Compute advected samples
        n_samples = 1000 # was 1k
        step_size = 1/60
        method = "midpoint"
        T = 1
        n_times = int(T/step_size) + 1
        times = torch.linspace(0, T, n_times, device=device)
        base_samples = base_sampler.sample(n_samples).to(device)
        base_samples += 0.5*torch.randn(n_samples, space_dims, dtype=torch.float64, device=device)
        advected_samples = odeint(trained_v, base_samples, times, method=method) # advected samples
        
        # Compute key times
        key_times = []
        for j in range(1,k+1):
            key_times.append(int((j/k)*60))
        
        # Plot samples
        norm = colors.Normalize(vmin=0, vmax=k*(len(targets)-2) + k + 1)
        cmap = cm.viridis
        for j, t in enumerate(key_times):
            data = advected_samples[t].detach().cpu().numpy()
            time_index = k*i + (j+1)
            #sns.kdeplot(x=data[:,0], y=data[:,1], fill=False, levels=100, ax=ax, color=cmap(norm(time_index)))
            ax.scatter(data[:,0], data[:,1], color=cmap(norm(time_index)), s=0.1)
        
    # Now handle final tps
    # Load the velocity field
    # Skip if k=2 and k=19 since there are no final tps in this case
    if k > 2 and k < 19:
        if use_A:
            fname = "trained_v_with_A_lambd_1em1_use_every_" + str(k) + "_final_tps.pt"
            fpath = "trained_models/scrna_vel_fields/with_A/" + fname
            fpath = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), fpath))
        else:
            fname = "trained_v_no_A_use_every_" + str(k) + "_final_tps.pt"
            fpath = "trained_models/scrna_vel_fields/no_A/" + fname
            fpath = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), fpath))
        space_dims = 2
        time_varying = True
        trained_v = cnf.VelocityField(space_dims, time_varying).to(device)
        trained_v.load_state_dict(torch.load(fpath))

        # Select base sampler
        base_sampler = targets[-1]

        # Compute advected samples
        n_samples = 1000 # was 1k
        step_size = 1/60
        method = "midpoint"
        T = 1
        n_times = int(T/step_size) + 1
        times = torch.linspace(0, T, n_times, device=device)
        base_samples = base_sampler.sample(n_samples).to(device)
        base_samples += 0.5*torch.randn(n_samples, space_dims, dtype=torch.float64, device=device)
        advected_samples = odeint(trained_v, base_samples, times, method=method) # advected samples

        # Compute new key times
        n_tps = len(rho_list) - old_target_tps[-1]
        key_times = []
        for j in range(1, n_tps):
            key_times.append(int((j/n_tps)*60))
        
        # Plot samples
        for j, t in enumerate(key_times):
            data = advected_samples[t].detach().cpu().numpy()
            time_index = old_target_tps[-1] + j
            ax.scatter(data[:,0], data[:,1], color=cmap(norm(time_index)), s=0.1)