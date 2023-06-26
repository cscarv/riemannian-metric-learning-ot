import importlib
import numpy as np
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint

import pytorch_utils as utils

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
torch.set_default_dtype(torch.float64)

class CircularA(nn.Module):
    """Hard-coded spatially-varying anisotropy tensor A(x,y) = I - v(x,y)v(x,y)' for v(x) = [y/sqrt(x^2 + y^2), -x/sqrt(x^2 + y^2)]."""

    def __init__(self):
        super(CircularA, self).__init__()
        self.d = 2

    def forward(self, x):
        # x should have dimensions (B, 2)
        v_0 = -x[:,1]
        v_1 = x[:,0]
        v = torch.stack((v_0, v_1), dim=1)
        # Now normalize v
        v = F.normalize(v)
        v = torch.unsqueeze(v, 2)
        # Construct A
        vT = torch.transpose(v, 1, 2)
        A = torch.eye(2, device=device) - torch.bmm(v, vT)
        return A
    
class MassSplittingA(nn.Module):
    """Hard-coded spatially-varying anisotropy tensor A(x,y) that causes mass about the origin to split."""

    def __init__(self):
        super(MassSplittingA, self).__init__()
        self.d = 2

    def forward(self, x):
        # x should have dimensions (B, 2)
        v_0 = torch.ones_like(x[:,0])
        v_1 = torch.sign(x[:,1])
        v = torch.stack((v_0, v_1), dim=1)
        # Now normalize v
        v = F.normalize(v)
        v = torch.unsqueeze(v, 2)
        # Construct A
        vT = torch.transpose(v, 1, 2)
        A = torch.eye(2, device=device) - torch.bmm(v, vT)
        return A
    
class XPathsA(nn.Module):
    """Hard-coded spatially-varying anisotropy tensor A(x,y) that causes mass to move in x-paths."""

    def __init__(self):
        super(XPathsA, self).__init__()
        self.d = 2

    def forward(self, x):
        # x should have dimensions (B, 2)
        v_1 = F.normalize(torch.ones_like(x))
        v_2 = torch.ones_like(x)
        v_2[:,1] *= -1
        v_2 = F.normalize(v_2)
        
        # Construct weights for combining v_1 and v_2
        alpha = 1.25*torch.tanh(F.relu(x[:,0]*x[:,1])) # large in quadrants 1,3
        beta = -1.25*torch.tanh(F.relu(-x[:,0]*x[:,1])) # large in quadrants 2,4
        
        alpha = torch.unsqueeze(alpha, dim=1)
        beta = torch.unsqueeze(beta, dim=1)
        
        v = alpha*v_1 + beta*v_2
        
        v = torch.unsqueeze(v, 2)
        # Construct A
        vT = torch.transpose(v, 1, 2)
        A = torch.eye(2, device=device) - torch.bmm(v, vT)
        return A
    
class VelocityField(nn.Module):
    """Spatially-varying velocity field v(x). Same parametrization as TrajectoryNet."""
    def __init__(self, space_dims):
        super(VelocityField, self).__init__()
        self.d = space_dims
        self.linear_elu_stack = nn.Sequential( # torchdiffeq docs say to avoid non-differentiable nonlinearities
            nn.Linear(self.d, 64),
            nn.ELU(),
            nn.Linear(64, 64),
            nn.ELU(),
            nn.Linear(64, self.d),
        )
        
    def forward(self, t, x): # t is dummy variable for compatibility with torchdiffeq
        return self.linear_elu_stack(x)
    
def compute_A_geodesic(x_0,
                       x_1,
                       A,
                       t,
                       lambd=1e0 # Strength of penalty on missing x_1
                      ):
    """Compute an A-geodesic x(t) between x_0 and x_1 parametrized over t \in [0,1].
    Report points x(t) at user-supplied times t.
    The last time t must be 1."""
    
    # Initialize the model for velocity field v
    space_dims = 2
    v = VelocityField(space_dims).to(device)
    
    # Initialize optimizer
    lr = 1e-3
    weight_decay = 1e-3
    opt = torch.optim.AdamW(v.parameters(), lr=lr, weight_decay=weight_decay)
    
    n_epochs = 100
    losses = []
    # Train the velocity field v to push x_0 to x_1 along an A-geodesic
    for epoch in range(n_epochs):
        opt.zero_grad()
        x_t = odeint(v, x_0, t, method='explicit_adams')
        target_penalty = lambd*torch.linalg.norm(x_t[-1] - x_1)
        # Compute kinetic energy loss
        A_at_x = A(x_t)
        v_at_x = torch.unsqueeze(v(t, x_t), dim=2)
        kinetic_energy = torch.mean(torch.sum(torch.squeeze(v_at_x * (A_at_x @ v_at_x)), dim=1))
        # Compute total loss and backprop
        loss = kinetic_energy + target_penalty
        losses.append(loss.detach().cpu())
        loss.backward()
        opt.step()
    
    # Report points x(t) for final geodesic
    return odeint(v, x_0, t), losses

def generate_Gaussians_in_circle_data(circle_radius,
                                      gaussian_std,
                                      n_samples,
                                      n_gaussians
                                     ):
    """Generates 2D Gaussians-in-circle training data using compute_A_geodesic.
    n_gaussians should be divisible by 4."""
    
    # Generate Tensor that will store the training data
    space_dims = 2
    data = torch.empty((n_gaussians, 0, space_dims), dtype=torch.float64, device=device)
    data_list = list(data) # each element of data_list is a Tensor of shape (0, space_dims)
    
    angles = torch.linspace(0, 2 * np.pi, 5).to(device) # divide the circle into 4 arcs, each anchored by two Gaussians
    angles = angles[:-1]
    x_coords = torch.t(torch.unsqueeze(torch.cos(angles), dim=0))
    y_coords = torch.t(torch.unsqueeze(torch.sin(angles), dim=0))
    means = circle_radius * torch.hstack((x_coords, y_coords))
    
    # Generate the true metric tensor A
    A = CircularA()
    
    # Generate training data
    for i in range(4):
        start_sampler = torch.distributions.Normal(means[i], gaussian_std)
        end_sampler = torch.distributions.Normal(means[(i+1)%4], gaussian_std)
        start_samples = start_sampler.sample((n_samples,)).to(device)
        end_samples = end_sampler.sample((n_samples,)).to(device)
        
        # Now compute geodesics
        for j in range(n_samples):
            x_0 = start_samples[j]
            x_1 = end_samples[j]
            n_times = int(n_gaussians/4) + 1
            t = torch.linspace(0, 1, n_times, device=device)
            x_t, losses = compute_A_geodesic(x_0, x_1, A, t)
            x_t = x_t[:-1] # exclude final time point
            for r in range(int(x_t.shape[0])):
                idx = int(n_gaussians/4)*i + r
                data_list[idx] = torch.cat((data_list[idx], torch.unsqueeze(x_t[r], dim=0)), dim=0)
        
    return torch.stack(data_list)

def generate_mass_splitting_data(gaussian_std,
                                 n_samples,
                                 n_intermediate
                                ):
    """Generates mass splitting training data using compute_A_geodesic."""
    
    # Generate Tensor that will store the training data
    space_dims = 2
    data = torch.empty((n_intermediate+2, 0, space_dims), dtype=torch.float64, device=device)
    data_list = list(data) # each element of data_list is a Tensor of shape (0, space_dims)
    
    initial_mean = torch.tensor([0,0], dtype=torch.float64, device=device)
    top_mean = torch.tensor([1,1], dtype=torch.float64, device=device)
    bottom_mean = torch.tensor([1,-1], dtype=torch.float64, device=device)
    
    # Generate the true metric tensor A
    A = MassSplittingA()
    
    # Generate training data
    start_sampler = torch.distributions.Normal(initial_mean, gaussian_std)
    top_end_sampler = torch.distributions.Normal(top_mean, gaussian_std)
    bottom_end_sampler = torch.distributions.Normal(bottom_mean, gaussian_std)
    start_samples = start_sampler.sample((n_samples,)).to(device)
    top_end_samples = top_end_sampler.sample((int(n_samples/2),)).to(device)
    bottom_end_samples = bottom_end_sampler.sample((int(n_samples/2),)).to(device)
    end_samples = torch.vstack((top_end_samples, bottom_end_samples))
    end_samples = end_samples[torch.randperm(end_samples.size()[0])] # shuffle end_samples        
        
    # Now compute geodesics
    for j in range(n_samples):
        x_0 = start_samples[j]
        x_1 = end_samples[j]
        n_times = n_intermediate + 2
        t = torch.linspace(0, 1, n_times, device=device)
        x_t, losses = compute_A_geodesic(x_0, x_1, A, t)
        x_t[-1] = x_1 # Fix last time point to its exact position
        for t_point in range(int(x_t.shape[0])):
            idx = t_point
            data_list[idx] = torch.cat((data_list[idx], torch.unsqueeze(x_t[t_point], dim=0)), dim=0)
        
    return torch.stack(data_list)

def generate_x_path_data(gaussian_std,
                         n_samples,
                         n_intermediate
                        ):
    """Generates X-path training data using compute_A_geodesic."""
    
    # Generate Tensor that will store the training data
    space_dims = 2
    data_0 = torch.empty((n_intermediate+2, 0, space_dims), dtype=torch.float64, device=device) # stores BL to TR trajectory
    data_0_list = list(data_0) # each element of data_list is a Tensor of shape (0, space_dims)
    data_1 = torch.empty((n_intermediate+2, 0, space_dims), dtype=torch.float64, device=device) # stores TL to BR trajectory
    data_1_list = list(data_1) # each element of data_list is a Tensor of shape (0, space_dims)
    
    tl_mean = torch.tensor([-1,1], dtype=torch.float64, device=device)
    br_mean = torch.tensor([1,-1], dtype=torch.float64, device=device)
    bl_mean = torch.tensor([-1,-1], dtype=torch.float64, device=device)
    tr_mean = torch.tensor([1,1], dtype=torch.float64, device=device)
    
    # Generate the true metric tensor A
    A = XPathsA()
    
    # Generate training data
    tl_sampler = torch.distributions.Normal(tl_mean, gaussian_std)
    br_sampler = torch.distributions.Normal(br_mean, gaussian_std)
    bl_sampler = torch.distributions.Normal(bl_mean, gaussian_std)
    tr_sampler = torch.distributions.Normal(tr_mean, gaussian_std)
    tl_samples = tl_sampler.sample((n_samples,)).to(device)
    br_samples = br_sampler.sample((n_samples,)).to(device)
    bl_samples = bl_sampler.sample((n_samples,)).to(device)
    tr_samples = tr_sampler.sample((n_samples,)).to(device)
        
    # Now compute geodesics for the BL to TR trajectory
    for j in range(n_samples):
        x_0 = bl_samples[j]
        x_1 = tr_samples[j]
        n_times = n_intermediate + 2
        t = torch.linspace(0, 1, n_times, device=device)
        x_t, losses = compute_A_geodesic(x_0, x_1, A, t)
        for t_point in range(int(x_t.shape[0])):
            idx = t_point
            data_0_list[idx] = torch.cat((data_0_list[idx], torch.unsqueeze(x_t[t_point], dim=0)), dim=0)
    data_bl_tr = torch.stack(data_0_list)
            
    # Now compute geodesics for the TL to BR trajectory
    for j in range(n_samples):
        x_0 = tl_samples[j]
        x_1 = br_samples[j]
        n_times = n_intermediate + 2
        t = torch.linspace(0, 1, n_times, device=device)
        x_t, losses = compute_A_geodesic(x_0, x_1, A, t, lambd=1e1)
        x_t[-1] = x_1 # Fix last time point to its exact position
        for t_point in range(int(x_t.shape[0])):
            idx = t_point
            data_1_list[idx] = torch.cat((data_1_list[idx], torch.unsqueeze(x_t[t_point], dim=0)), dim=0)
    data_tl_br = torch.stack(data_1_list)
        
    return torch.stack((data_bl_tr, data_tl_br))