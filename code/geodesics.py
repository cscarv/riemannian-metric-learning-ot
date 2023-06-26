import torch
import torch.nn as nn
from torchdiffeq import odeint

torch.set_default_dtype(torch.float64)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
                       lambd # Strength of penalty on missing x_1
                      ):
    """Compute an A-geodesic x(t) between x_0 and x_1 parametrized over t \in [0,1].
    Report points x(t) at user-supplied times t.
    The last time t must be 1."""
    
    # Initialize the model for velocity field v
    space_dims = 2
    v = VelocityField(space_dims).to(device)
    
    # Initialize optimizer
    lr = 1e-3
    weight_decay = 0
    opt = torch.optim.AdamW(v.parameters(), lr=lr, weight_decay=weight_decay)
    
    n_epochs = 500
    losses = []
    # Train the velocity field v to push x_0 to x_1 along an A-geodesic
    for epoch in range(n_epochs):
        opt.zero_grad()
        x_t = odeint(v, x_0, t, method='dopri5')
        target_penalty = lambd*torch.linalg.norm(x_t[-1] - x_1)
        # Compute kinetic energy loss
        A_at_x = A(x_t)
        v_at_x = torch.unsqueeze(v(t, x_t), dim=2)
        #kinetic_energy = torch.mean(torch.sum(torch.squeeze(v_at_x * (A_at_x @ v_at_x)), dim=1))
        kinetic_energy = torch.mean(torch.sum(torch.squeeze(v_at_x * torch.linalg.solve(A_at_x, v_at_x)), dim=1)) # v(t,x(t))^T @ A^-1(x(t)) @ v(t,x(t))
        #print(kinetic_energy)
        # Compute total loss and backprop
        loss = kinetic_energy + target_penalty
        losses.append(loss.detach().cpu())
        loss.backward()
        opt.step()
    
    # Report points x(t) for final geodesic
    return odeint(v, x_0, t), losses