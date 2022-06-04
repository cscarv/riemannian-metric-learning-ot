import numpy as np
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)


class EmpiricalDist:
    """Empirical distribution object.

    Stores empirical data as Pytorch Tensor and has sample() method.
    """

    def __init__(self, data, device):
        self.data = data
        self.n_points = data.shape[0]
        self.dims = data.shape[1]
        self.device = device

    def sample(self, n_samples):
        """Generate n_samples drawn with replacement from self.data."""
        indices = torch.randint(high=self.n_points, size=(n_samples,)).to(self.device)
        return self.data[indices].to(self.device)


def generate_gaussian_samplers(n_means, circle_radius, std):
    """Generates lists of n_means isotropic Gaussian distributions rho_0 and rho_1 with fixed std
    and means drawn uniformly from circle of radius circle_radius."""

    angles = torch.linspace(0, 2 * np.pi, n_means)  # consider just excluding the endpoint here

    rho_0_list = []
    rho_1_list = []
    shift = 0.2

    for i in range(angles.shape[0] - 1):
        center = circle_radius * torch.tensor(
            [torch.cos(angles[i] + shift), torch.sin(angles[i] + shift)], device=device
        )
        rho_0_list.append(torch.distributions.Normal(center, torch.tensor([std]).to(device)))

        rho_1_list = rho_0_list[1::] + rho_0_list[:1:]

    return rho_0_list, rho_1_list


def generate_box_sampler(ell, r, b, u):
    """Generates a pair of samplers that sample from the x-coord and y-coord of a uniform
    distribution over [l,r] x [b,u]."""

    x_sampler = torch.distributions.uniform.Uniform(ell, r)
    y_sampler = torch.distributions.uniform.Uniform(b, u)

    return x_sampler, y_sampler


def generate_eb_samplers(data_list):
    """Returns rho_0_list and rho_1_list, where the rho_0 and rho_1 are now EmpiricalDist objects
    that draw samples with replacement from TrajectoryNet's EB data."""

    rho_0_list = []

    for i in range(len(data_list)):
        data = torch.from_numpy(data_list[i]).double()
        rho_0_list.append(EmpiricalDist(data, device))

    rho_1_list = rho_0_list[1::] # rho_1_list ranges from t=1 to t=T
    rho_0_list = rho_0_list[:-1] # rho_0_list ranges from t=0 to t=T-1

    return rho_0_list, rho_1_list

def generate_all_eb_samplers(data_list):
    """Returns rho_list, whose elements are EmpiricalDist objects
    that draw samples with replacement from TrajectoryNet's EB data."""

    rho_list = []

    for i in range(len(data_list)):
        data = torch.from_numpy(data_list[i]).double()
        rho_list.append(EmpiricalDist(data, device))

    return rho_list

def generate_synthetic_samplers(data):
    """Returns rho_0_list and rho_1_list, where the rho_0 and rho_1 are now EmpiricalDist objects
    that draw samples with replacement from synthetic data."""

    rho_0_list = []

    for t in range(data.shape[0]):
        data_t = data[t]
        rho_0_list.append(EmpiricalDist(data_t, device))

    rho_1_list = rho_0_list[1::] # rho_1_list ranges from t=1 to t=T
    rho_0_list = rho_0_list[:-1] # rho_0_list ranges from t=0 to t=T-1

    return rho_0_list, rho_1_list
