import pykeops
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint as odeint
from geomloss import SamplesLoss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
torch.set_default_dtype(torch.float64)

# EmpiricalDist is a utility class that generates a sampler for N cross-sectional data points of dim D stored in an (N,D) Tensor

class EmpiricalDist:
    """Empirical distribution object.

    Has sample() method that samples with replacement from cross-sectional data stored in an (N,D) Tensor.
    """

    def __init__(self, data, device):
        self.data = data
        self.n_points = data.shape[0]
        self.dims = data.shape[1]
        self.device = device

    def sample(self, n_samples):
        """Generate n_samples drawn with replacement from self.data."""
        indices = torch.randint(high=self.n_points, size=n_samples).to(self.device)
        return self.data[indices]

class VelocityField(nn.Module):
    """Spatially-varying and optionally time-varying velocity field v(x). Same parametrization as TrajectoryNet."""
    def __init__(self, space_dims, time_varying=False, hidden_dims=64):
        super(VelocityField, self).__init__()
        self.time_varying = time_varying
        if self.time_varying:
            self.in_dims = space_dims + 1
        else:
            self.in_dims = space_dims
        self.input_layer = nn.Linear(self.in_dims, hidden_dims)
        self.hidden_layer = nn.Sequential(nn.Linear(hidden_dims + 1, hidden_dims), nn.Softplus())
        self.out_layer = nn.Linear(hidden_dims, space_dims)
        self.hidden_stack = nn.ModuleList( # torchdiffeq docs say to avoid non-differentiable nonlinearities
            [self.hidden_layer for i in range(3)]
        )
        
    def forward(self, t, x): # t will always be a scalar during calls from torchdiffeq.odeint, but may be (T,) for energy loss computation
        if self.time_varying:
            if t.ndim == 0: # called by odeint
                time = t*torch.ones_like(x[:,0], dtype=torch.float64, device=device)
                x_and_t = torch.hstack((x, torch.unsqueeze(time, dim=1)))
                hidden = self.input_layer(x_and_t)
                for hidden_layer in self.hidden_stack:
                    hidden = torch.hstack((hidden, torch.unsqueeze(time, dim=1)))
                    hidden = hidden_layer(hidden)
                return self.out_layer(hidden)
            else: # forward pass for kinetic energy computation
                B = int(x.shape[0]/t.shape[0]) # assumes that x_t has already been reshaped to (T*B, space_dims)
                time = torch.unsqueeze(t.repeat_interleave(B), dim=1)
                x_and_t = torch.hstack((x, time))
                hidden = self.input_layer(x_and_t)
                for hidden_layer in self.hidden_stack:
                    hidden = torch.hstack((hidden, time))
                    hidden = hidden_layer(hidden)
                return self.out_layer(hidden)
        else: # will need to fix this
            return self.linear_elu_stack(x)
        
def v_quiver(v, n, box_radius, t):
    """Visualize velocity field v(t,x) for fixed t on n x n grid. 
    Grid is overlaid on [-box_radius, box_radius]^2."""
    x = torch.linspace(-box_radius, box_radius, n)
    y = torch.linspace(-box_radius, box_radius, n)
    X, Y = torch.meshgrid(x, y)
    U1 = torch.zeros((n, n))  # x-component of vector field
    V1 = torch.zeros((n, n))  # y-component of vector field
    for i in range(n):
        for j in range(n):
            pt = torch.tensor([[X[i, j], Y[i, j]]], device=device).double()
            vel = v(t, pt).detach().cpu()
            U1[i, j] = vel[0, 0]
            V1[i, j] = vel[0, 1]
    plt.figure(figsize=(20,20))
    plt.quiver(X, Y, U1, V1, color="r")
    
def visualize_advected_samples(advected_samples, key_times=None):
    """Visualize advected samples outputted by sinkhorn_cnf."""
    plt.figure(figsize=(20,20))
    if key_times is not None:
        for t in key_times:
            plt.scatter(advected_samples[t,:,0].detach().cpu(), advected_samples[t,:,1].detach().cpu())
    else:
        for t in range(len(advected_samples)):
            plt.scatter(advected_samples[t,:,0].detach().cpu(), advected_samples[t,:,1].detach().cpu())
    
        
def sinkhorn_cnf(base_sampler, # Should have a sample() method that behaves like torch.Distributions.Normal.sample()
                 target_sampler_list, # Each element should have a sample() method
                 n_samples, # Number of samples drawn from each sampler in each epoch of training
                 step_size, # Size of time step between consecutive advected samples. 1/step_size should be an integer
                 space_dims,
                 lambd, # Controls strength of kinetic energy loss
                 lr,
                 weight_decay,
                 n_epochs,
                 A=None, # Metric tensor as Pytorch nn.Module. Defaults to A(x)=I if not user-specified.
                 time_varying=False, # Is the velocity field time-varying?
                 hidden_dims=64, # Hidden dims for velocity field neural net
                 v_model=None # Can input a pre-trained model for further training
                ):
    """Trains a Sinkhorn CNF that advects samples from start_sampler to samples drawn
    from each sampler in target_sampler_list. Assumes that the targets are given in 
    consecutive order and are equispaced in time."""
    
    # Initialize the model for velocity field v
    if v_model is None:
        v = VelocityField(space_dims, time_varying, hidden_dims).to(device)
    else:
        v = v_model
    
    # Initialize optimizer
    opt = torch.optim.AdamW(v.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Define times at which advected particle positions are computed
    T = len(target_sampler_list)
    n_times = int(T/step_size) + 1
    times = torch.linspace(0, T, n_times, device=device)
    
    # Construct loss function
    sinkhorn_losses = []
    kinetic_energies = []
    sinkhorn_loss = SamplesLoss(loss="sinkhorn", p=2, blur=5e-2) #0.05 = 5e-2 is good start point
    
    # Select ODE solver -- see torchdiffeq docs for options
    method = "midpoint"
    
    # Train the velocity field v to push base_samples to targets
    for epoch in range(n_epochs):
        opt.zero_grad()
        
        # If base sample is EmpiricalDist, then argument to .sample() should NOT be a tuple
        base_samples = base_sampler.sample(n_samples).to(device)
        
        # Add some Gaussian noise to the base samples
        base_samples += 0.5*torch.randn(n_samples, space_dims, dtype=torch.float64, device=device)
        
        # Advect base_samples through velocity field v
        x_t = odeint(v, base_samples, times, method=method) # advected samples
        
        # Compute kinetic energy loss
        # First unroll x_t to convert first two batch dimensions (T,B) into one larger batch dim (T*B)
        if lambd > 0:
            T = x_t.shape[0]
            B = x_t.shape[1]
            x_t_unrolled = torch.reshape(x_t, (T*B, space_dims))
            if A is None: # set A(x) to the identity map if not user-specified
                eye = torch.eye(2, dtype=torch.float64, device=device)
                A_at_x = eye.repeat(T*B, 1, 1)
            else:
                A_at_x = A(x_t_unrolled)
            v_at_x = torch.unsqueeze(v(times, x_t_unrolled), dim=2)
            kinetic_energy = torch.mean(torch.sum(torch.squeeze(v_at_x * (A_at_x @ v_at_x)), dim=1)) # v(t,x(t))^T @ A(x(t)) @ v(t,x(t))
        else:
            kinetic_energy = torch.tensor(0, dtype=torch.float64, device=device)
        
        # Compute Sinkhorn losses
        target_penalty = 0
        for t, target_sampler in enumerate(target_sampler_list):
            t_index = int((t+1)/step_size)
            advected_samples = x_t[t_index]
            target_samples = target_sampler.sample(n_samples).to(device) # argument is not tuple
            target_penalty += sinkhorn_loss(advected_samples, target_samples)
            
        # Compute total loss and backprop
        loss = lambd*kinetic_energy + target_penalty
        if epoch%100 == 0:
            print("iter " + str(epoch) + " loss: " + str(target_penalty))
        sinkhorn_losses.append(target_penalty.detach().cpu())
        kinetic_energies.append(lambd*kinetic_energy.detach().cpu())
        loss.backward()
        opt.step()
    
    loss_vals = [sinkhorn_losses, kinetic_energies]
    
    return odeint(v, base_samples, times, method=method), v, loss_vals