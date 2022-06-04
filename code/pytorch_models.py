import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)


class PSDMatrix(nn.Module):
    """PSD matrix-valued function."""

    def __init__(self, space_dims, hidden_dims):
        super(PSDMatrix, self).__init__()
        self.d = space_dims
        self.layer1 = nn.Linear(space_dims, hidden_dims)
        self.layer2 = nn.Linear(
            hidden_dims, space_dims ** 2
        )  # Output should be nxn if n:=space_dims

    def forward(self, x):
        R_vec = self.layer2(F.softplus(self.layer1(x)))  # shallow NN -- note softplus activation
        R_mat = torch.reshape(R_vec, (x.shape[0], self.d, self.d))
        # Note that we add 1e-3 to the diagonal of R -- this ensures that it's actually PD
        return torch.matmul(torch.transpose(R_mat, 1, 2), R_mat) + 1e-3 * torch.eye(
            2, device=device
        )
    
class PSDMatrixMultiLayer(nn.Module):
    """PSD matrix-valued function."""

    def __init__(self, space_dims, hidden_dims):
        super(PSDMatrixMultiLayer, self).__init__()
        self.d = space_dims
        self.input_layer = nn.Sequential(nn.Linear(space_dims, hidden_dims), nn.Softplus())
        self.hidden_layer1 = nn.Sequential(nn.Linear(hidden_dims, hidden_dims), nn.Softplus())
        self.hidden_layer2 = nn.Sequential(nn.Linear(hidden_dims, hidden_dims), nn.Softplus())
        self.hidden_layer3 = nn.Sequential(nn.Linear(hidden_dims, hidden_dims), nn.Softplus())
        #self.hidden_stack = nn.Sequential(self.hidden_layer1, self.hidden_layer2, self.hidden_layer3)
        self.hidden_stack = nn.Sequential(self.hidden_layer1)
        self.out_layer = nn.Linear(hidden_dims, space_dims ** 2)  # Output should be nxn if n:=space_dims

    def forward(self, x):
        R_vec = self.out_layer(self.hidden_stack(self.input_layer(x)))  # multilayer NN
        R_mat = torch.reshape(R_vec, (x.shape[0], self.d, self.d))
        # Note that we add 1e-3 to the diagonal of R -- this ensures that it's actually PD
        return torch.matmul(torch.transpose(R_mat, 1, 2), R_mat) + 1e-3 * torch.eye(2, device=device)


class ScalarFn(nn.Module):
    """Scalar-valued function."""

    def __init__(self, space_dims, hidden_dims):
        super(ScalarFn, self).__init__()
        self.d = space_dims
        self.layer1 = nn.Linear(space_dims, hidden_dims)
        self.layer2 = nn.Linear(hidden_dims, 1)  # Output should be real value

    def forward(self, x):
        phi = self.layer2(F.softplus(self.layer1(x)))  # shallow NN -- was softplus
        return phi


class SpectralNormScalarFn(nn.Module):
    """Scalar-valued function with spectral normalization of its parameters."""

    def __init__(self, space_dims, hidden_dims):
        super(SpectralNormScalarFn, self).__init__()
        self.d = space_dims
        self.layer1 = spectral_norm(nn.Linear(space_dims, hidden_dims))
        self.layer2 = spectral_norm(nn.Linear(hidden_dims, 1))  # Output should be real value

    def forward(self, x):
        phi = self.layer2(F.softplus(self.layer1(x)))  # shallow NN
        return phi
