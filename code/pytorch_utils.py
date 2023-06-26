import matplotlib.pyplot as plt
import numpy as np
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def eigs_quiver(R, n, x_lims, y_lims, color="black"):
    x = np.linspace(x_lims[0], x_lims[1], n)
    print(x)
    y = np.linspace(y_lims[0], y_lims[1], n)
    X, Y = np.meshgrid(x, y)
    U1 = np.zeros((n, n))  # x-component of vector field
    V1 = np.zeros((n, n))  # y-component of vector field
    U2 = np.zeros((n, n))
    V2 = np.zeros((n, n))
    log_ratios = np.zeros((n, n))
    ratios = np.zeros((n, n))
    small_eigs = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            pt = torch.tensor([[X[i, j], Y[i, j]]], device=device).double()
            M = R(pt).squeeze().detach()
            sorted_evals, sorted_evecs = torch.linalg.eigh(M)
            small_eigs[i, j] = sorted_evals[0]
            if sorted_evals[0] == 0:
                log_ratios[i,j] = -1
                ratios[i,j] = -1
            else:
                log_ratios[i,j] = torch.log10(sorted_evals[1]/sorted_evals[0])
                ratios[i,j] = sorted_evals[1]/sorted_evals[0]
            U1[i, j] = sorted_evecs[0, 0]
            V1[i, j] = sorted_evecs[1, 0]
            U2[i, j] = sorted_evecs[0, 1]
            V2[i, j] = sorted_evecs[1, 1]
    plt.pcolor(X, Y, log_ratios)
    plt.colorbar()
    plt.quiver(X, Y, U2, V2, color=color) # was U1, V1
    plt.quiver(X, Y, -U2, -V2, color=color) # was -U1, -V1
    
def eigs_quiver_comparison(ax, A_true, A_learned, n, x_lims, y_lims):
    x = np.linspace(x_lims[0], x_lims[1], n)
    y = np.linspace(y_lims[0], y_lims[1], n)
    X, Y = np.meshgrid(x, y)
    U1_true = np.zeros((n, n))  # x-component of vector field
    V1_true = np.zeros((n, n))  # y-component of vector field
    U1_learned = np.zeros((n, n))  # x-component of vector field
    V1_learned = np.zeros((n, n))  # y-component of vector field
    U2_learned = np.zeros((n, n))  # x-component of vector field
    V2_learned = np.zeros((n, n))  # y-component of vector field
    for i in range(n):
        for j in range(n):
            pt = torch.tensor([[X[i, j], Y[i, j]]], device=device).double()
            M_true = A_true(pt).squeeze().detach()
            sorted_evals_true, sorted_evecs_true = torch.linalg.eigh(M_true)
            U1_true[i, j] = sorted_evecs_true[0, 0]
            V1_true[i, j] = sorted_evecs_true[1, 0]
            M_learned = A_learned(pt).squeeze().detach()
            sorted_evals_learned, sorted_evecs_learned = torch.linalg.eigh(M_learned)
            U1_learned[i, j] = sorted_evecs_learned[0, 0]
            V1_learned[i, j] = sorted_evecs_learned[1, 0]
            U2_learned[i, j] = sorted_evecs_learned[0, 1]
            V2_learned[i, j] = sorted_evecs_learned[1, 1]
    ax.quiver(X, Y, U1_true, V1_true, color="#5D3A9B")
    ax.quiver(X, Y, -U1_true, -V1_true, color="#5D3A9B")
    ax.quiver(X, Y, U2_learned, V2_learned, color="#E66100") # was U1_learned, V1_learned
    ax.quiver(X, Y, -U2_learned, -V2_learned, color="#E66100") # was -U1_learned, -V1_learned
    
def eigs_quiver_with_conds(ax, A_learned, n, x_lims, y_lims):
    x = np.linspace(x_lims[0], x_lims[1], n)
    y = np.linspace(y_lims[0], y_lims[1], n)
    X, Y = np.meshgrid(x, y)
    U1_learned = np.zeros((n, n))  # x-component of vector field
    V1_learned = np.zeros((n, n))  # y-component of vector field
    log_ratios = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            pt = torch.tensor([[X[i, j], Y[i, j]]], device=device).double()
            M_learned = A_learned(pt).squeeze().detach()
            sorted_evals_learned, sorted_evecs_learned = torch.linalg.eigh(M_learned)
            U1_learned[i, j] = sorted_evecs_learned[0, 0]
            V1_learned[i, j] = sorted_evecs_learned[1, 0]
            log_ratios[i,j] = torch.log10(sorted_evals_learned[1]/sorted_evals_learned[0])
    im = ax.pcolor(X, Y, log_ratios, cmap="magma")
    return im
    #ax.quiver(X, Y, U1_learned, V1_learned, color="#E66100")
    #ax.quiver(X, Y, -U1_learned, -V1_learned, color="#E66100")
    
def eigs_quiver_rescaled(R, n, x_lims, y_lims, scale_factor):
    x = np.linspace(x_lims[0], x_lims[1], n)
    y = np.linspace(y_lims[0], y_lims[1], n)
    X, Y = np.meshgrid(x, y)
    U1 = np.zeros((n, n))  # x-component of vector field
    V1 = np.zeros((n, n))  # y-component of vector field
    U2 = np.zeros((n, n))
    V2 = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            pt = torch.tensor([[X[i, j], Y[i, j]]], device=device).float()
            M = R(pt).squeeze().detach()
            L, Q = torch.linalg.eigh(M)
            # Rescale L
            L = torch.tensor([scale_factor, 1]).float().to(device)
            # Reconstruct M with rescaled L
            M = Q @ torch.diag(L) @ Q.T
            sorted_evals, sorted_evecs = torch.linalg.eigh(M)
            U1[i, j] = (1 / sorted_evals[0]) * sorted_evecs[0, 0]
            V1[i, j] = (1 / sorted_evals[0]) * sorted_evecs[1, 0]
            U2[i, j] = (1 / sorted_evals[1]) * sorted_evecs[0, 1]
            V2[i, j] = (1 / sorted_evals[1]) * sorted_evecs[1, 1]
    plt.quiver(X, Y, U1, V1, scale=1e6)
    plt.quiver(X, Y, -U1, -V1, scale=1e6)
    plt.quiver(X, Y, U2, V2, scale=1e6)
    plt.quiver(X, Y, -U2, -V2, scale=1e6)
    
def eigs_similarity_metric(A_true, A_learned, n, x_lims, y_lims, space_dims): # x_lims, y_lims was box_radius
    #xs = torch.linspace(-box_radius, box_radius, n).to(device)
    #grid_pts = torch.cartesian_prod(*[xs]*space_dims)
    x = torch.linspace(x_lims[0], x_lims[1], n).to(device)
    y = torch.linspace(y_lims[0], y_lims[1], n).to(device)
    grid_pts = torch.cartesian_prod(x, y)
    true_A_at_pts = A_true(grid_pts)
    learned_A_at_pts = A_learned(grid_pts)
    true_A_evals, true_A_evecs = torch.linalg.eigh(true_A_at_pts)
    learned_A_evals, learned_A_evecs = torch.linalg.eigh(learned_A_at_pts)
    true_A_evecs_T = torch.transpose(true_A_evecs, 1, 2)
    inner_prods = torch.bmm(true_A_evecs_T, torch.flip(learned_A_evecs, dims=(2,))) # reverse order of learned_A_evecs
    diags = torch.diagonal(inner_prods, dim1=1, dim2=2)
    similarity = torch.linalg.vector_norm(diags, ord=1)
    similarity = similarity/(diags.shape[0]*diags.shape[1])
    return similarity


def phi_heatmap(phi, n, box_radius):
    x = np.linspace(-box_radius, box_radius, n)
    y = np.linspace(-box_radius, box_radius, n)
    X, Y = np.meshgrid(x, y)
    phi_field = np.zeros((n, n))  # values of phi
    for i in range(n):
        for j in range(n):
            pt = torch.tensor([[X[i, j], Y[i, j]]], device=device).double()
            phi_field[i, j] = phi(pt).detach().cpu()
    plt.pcolor(X, Y, phi_field)
    plt.colorbar()


def phi_quiver(phi, n, box_radius):
    x = np.linspace(-box_radius, box_radius, n)
    y = np.linspace(-box_radius, box_radius, n)
    X, Y = np.meshgrid(x, y)
    U1 = np.zeros((n, n))  # x-component of vector field
    V1 = np.zeros((n, n))  # y-component of vector field
    for i in range(n):
        for j in range(n):
            pt = torch.tensor([[X[i, j], Y[i, j]]], device=device).double()
            pt.requires_grad = True
            grad = torch.autograd.grad(
                outputs=phi(pt), inputs=pt, grad_outputs=torch.ones_like(phi(pt))
            )[0]
            U1[i, j] = grad[0, 0]
            V1[i, j] = grad[0, 1]
    plt.quiver(X, Y, U1, V1, color="r")


def R_norm_heatmap(R, n, box_radius):
    x = np.linspace(-box_radius, box_radius, n)
    y = np.linspace(-box_radius, box_radius, n)
    X, Y = np.meshgrid(x, y)
    R_field = np.zeros((n, n))  # values of phi
    for i in range(n):
        for j in range(n):
            pt = torch.tensor([[X[i, j], Y[i, j]]], device=device).float()
            R_field[i, j] = torch.norm(R(pt) ** 2).detach().cpu()
    plt.pcolor(X, Y, R_field)
    plt.colorbar()


def load_eb_data(data_file):
    """Loads EB data from TrajectoryNet paper."""
    data_dict = np.load(data_file, allow_pickle=True)

    num_principal_components = 2
    all_data = data_dict["pcs"][:, 0:num_principal_components]

    timestamps = data_dict["sample_labels"]

    data_list = []
    data_list.append(all_data[timestamps == 0])  # data_list[0] contains datapoints at time 0
    data_list.append(all_data[timestamps == 1])  # data_list[1] contains datapoints at time 1, etc
    data_list.append(all_data[timestamps == 2])
    data_list.append(all_data[timestamps == 3])
    data_list.append(all_data[timestamps == 4])

    return data_list

def load_rescaled_eb_data(data_file):
    """Loads rescaled EB data from TrajectoryNet paper."""
    data_dict = np.load(data_file, allow_pickle=True)

    num_principal_components = 2
    #all_data = data_dict["pcs"][:, 0:num_principal_components]
    all_data = np.load("../labb/data/rescaled_eb_pca.npy")[:, 0:num_principal_components]

    timestamps = data_dict["sample_labels"]

    data_list = []
    data_list.append(all_data[timestamps == 0])  # data_list[0] contains datapoints at time 0
    data_list.append(all_data[timestamps == 1])  # data_list[1] contains datapoints at time 1, etc
    data_list.append(all_data[timestamps == 2])
    data_list.append(all_data[timestamps == 3])
    data_list.append(all_data[timestamps == 4])

    return data_list


def load_eb_phate_data(data_file, label_file):
    """Loads EB-PHATE data from TrajectoryNet paper."""
    data_file = "../labb/data/phate_data.npy"
    all_phate = 100 * np.load(data_file, allow_pickle=True)

    label_file = "../labb/data/phate_labels.npy"
    phate_labels = np.load(label_file, allow_pickle=True)

    phate_data_list = []
    phate_data_list.append(all_phate[phate_labels == "Day 00-03"])
    phate_data_list.append(all_phate[phate_labels == "Day 06-09"])
    phate_data_list.append(all_phate[phate_labels == "Day 12-15"])
    phate_data_list.append(all_phate[phate_labels == "Day 18-21"])
    phate_data_list.append(all_phate[phate_labels == "Day 24-27"])

    return phate_data_list
