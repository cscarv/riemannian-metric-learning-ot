import os
import torch
import pytorch_losses as losses
import pytorch_models as models
import pytorch_samplers as samplers
import pytorch_utils as utils

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_gaussian_circle_model(
    n_means,
    circle_radius,
    std,
    n_samples,
    space_dims,
    scalar_hidden_dims,
    matrix_hidden_dims,
    fro_reg_strength,
    gp_strength,
    lr,
    weight_decay,
    n_epochs,
    seed,
):
    """Trains the model on the Gaussians-in-circle toy example."""

    torch.manual_seed(seed)  # Fix random seed

    # Generate samplers for the Gaussian dists
    rho_0_list, rho_1_list = samplers.generate_gaussian_samplers(n_means, circle_radius, std)

    # Initialize models
    R = models.PSDMatrix(space_dims, matrix_hidden_dims).to(device)
    phi_list = []
    for i in range(n_means - 1):
        phi_list.append(models.ScalarFn(space_dims, scalar_hidden_dims).to(device))

    params = (
        list(R.parameters())
        + list(phi_list[0].parameters())
        + list(phi_list[1].parameters())
        + list(phi_list[2].parameters())
        + list(phi_list[3].parameters())
        + list(phi_list[4].parameters())
        + list(phi_list[5].parameters())
        + list(phi_list[6].parameters())
        + list(phi_list[7].parameters())
        + list(phi_list[8].parameters())
    )

    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    for epoch in range(n_epochs):
        opt.zero_grad()
        loss = 0
        for i in range(n_means - 1):
            rho_0 = rho_0_list[i]
            rho_1 = rho_1_list[i]
            phi = phi_list[i]
            loss += losses.loss_fn(phi, R, rho_0, rho_1, n_samples, fro_reg_strength, gp_strength)
        loss.backward()
        opt.step()

    return R, phi_list, rho_0_list


def train_gaussian_circle_phis(
    n_means,
    circle_radius,
    std,
    n_samples,
    space_dims,
    scalar_hidden_dims,
    lr,
    weight_decay,
    n_epochs,
    seed,
):
    """Trains the phis alone on the Gaussians-in-circle toy example.

    Implements the first step of the two-step training procedure.
    """

    torch.manual_seed(seed)  # Fix random seed

    # Generate samplers for the Gaussian dists
    rho_0_list, rho_1_list = samplers.generate_gaussian_samplers(n_means, circle_radius, std)

    # Initialize models
    phi_list = []
    for i in range(n_means - 1):
        phi_list.append(models.ScalarFn(space_dims, scalar_hidden_dims).to(device))

    params = (
        list(phi_list[0].parameters())
        + list(phi_list[1].parameters())
        + list(phi_list[2].parameters())
        + list(phi_list[3].parameters())
        + list(phi_list[4].parameters())
        + list(phi_list[5].parameters())
        + list(phi_list[6].parameters())
        + list(phi_list[7].parameters())
        + list(phi_list[8].parameters())
    )

    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    for epoch in range(n_epochs):
        opt.zero_grad()
        loss = 0
        for i in range(n_means - 1):
            rho_0 = rho_0_list[i]
            rho_1 = rho_1_list[i]
            rho_0_samples = rho_0.sample((n_samples, 1)).squeeze()
            rho_1_samples = rho_1.sample((n_samples, 1)).squeeze()
            phi = phi_list[i]
            # Compute BB loss alone
            # Rely on weight decay to approximate spectral normalization
            loss += losses.bb_loss(phi, rho_0_samples, rho_1_samples, n_samples)
        loss.backward()
        opt.step()

    return phi_list, rho_0_list, rho_1_list


def train_gaussian_circle_R(
    phi_list,
    n_means,
    circle_radius,
    std,
    n_samples,
    space_dims,
    matrix_hidden_dims,
    fro_reg_strength,
    gp_strength,
    lr,
    weight_decay,
    n_epochs,
    seed,
):
    """Trains R alone on the Gaussians-in-circle toy example.

    Implements the second step of the two-step training procedure.
    """

    torch.manual_seed(seed)  # Fix random seed

    # Generate samplers for the Gaussian dists
    rho_0_list, rho_1_list = samplers.generate_gaussian_samplers(n_means, circle_radius, std)

    # Initialize model
    R = models.PSDMatrix(space_dims, matrix_hidden_dims).to(device)

    params = list(R.parameters())

    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    for epoch in range(n_epochs):
        opt.zero_grad()
        loss = 0
        for i in range(n_means - 1):
            rho_0 = rho_0_list[i]
            rho_1 = rho_1_list[i]
            rho_0_samples = rho_0.sample((n_samples, 1)).squeeze()
            rho_1_samples = rho_1.sample((n_samples, 1)).squeeze()
            phi = phi_list[i]
            loss += fro_reg_strength * losses.fro_norm_regularizer(
                R, rho_0_samples, rho_1_samples, n_samples
            )
            loss += gp_strength * losses.gradient_penalty(
                phi, R, rho_0_samples, rho_1_samples, n_samples
            )
        loss.backward()
        opt.step()

    return R, rho_0_list, rho_1_list


def twostep_train_gaussian_circle_model(
    n_means,
    circle_radius,
    std,
    n_samples,
    space_dims,
    scalar_hidden_dims,
    matrix_hidden_dims,
    fro_reg_strength,
    gp_strength,
    lr,
    weight_decay,
    n_epochs_phi,
    n_epochs_R,
    seed,
):
    """Trains the model on the Gaussians-in-circle toy example using the two-step procedure."""

    # First train the phi models
    phi_list, rho_0_list, rho_1_list = train_gaussian_circle_phis(
        n_means,
        circle_radius,
        std,
        n_samples,
        space_dims,
        scalar_hidden_dims,
        lr,
        weight_decay,
        n_epochs_phi,
        seed,
    )

    # Then train the R model given the learned phis
    R, rho_0_list, rho_1_list = train_gaussian_circle_R(
        phi_list,
        n_means,
        circle_radius,
        std,
        n_samples,
        space_dims,
        matrix_hidden_dims,
        fro_reg_strength,
        gp_strength,
        lr,
        weight_decay,
        n_epochs_R,
        seed,
    )

    return R, phi_list, rho_0_list, rho_1_list


def train_three_boxes_phis(
    l_tensor,
    r_tensor,
    b_tensor,
    u_tensor,
    n_samples,
    scalar_hidden_dims,
    lr,
    weight_decay,
    n_epochs,
    seed,
):
    """Trains the phis alone on the three-boxes toy example.

    Implements the first step of the two-step training procedure.
    """

    torch.manual_seed(seed)  # Fix random seed

    # Generate samplers for the uniform dists on the three boxes
    x_samplers = []
    y_samplers = []
    box_0_x, box_0_y = samplers.generate_box_sampler(
        l_tensor[0], r_tensor[0], b_tensor[0], u_tensor[0]
    )
    x_samplers.append(box_0_x)
    y_samplers.append(box_0_y)

    box_1_x, box_1_y = samplers.generate_box_sampler(
        l_tensor[1], r_tensor[1], b_tensor[1], u_tensor[1]
    )
    x_samplers.append(box_1_x)
    y_samplers.append(box_1_y)

    box_2_x, box_2_y = samplers.generate_box_sampler(
        l_tensor[2], r_tensor[2], b_tensor[2], u_tensor[2]
    )
    x_samplers.append(box_2_x)
    y_samplers.append(box_2_y)

    # Initialize models
    phi_list = []
    space_dims = 2
    for i in range(2):
        phi_list.append(models.ScalarFn(space_dims, scalar_hidden_dims).to(device))

    params = list(phi_list[0].parameters()) + list(phi_list[1].parameters())

    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    for epoch in range(n_epochs):
        opt.zero_grad()
        loss = 0

        # loss for phi_0
        x_0 = x_samplers[0]
        y_0 = y_samplers[0]
        x_1 = x_samplers[1]
        y_1 = y_samplers[1]
        phi = phi_list[0]
        # Draw rho_0_samples
        x0_samples = x_0.sample((n_samples, 1))
        y0_samples = y_0.sample((n_samples, 1))
        rho_0_samples = torch.hstack((x0_samples, y0_samples))
        # Draw rho_1_samples
        x1_samples = x_1.sample((n_samples, 1))
        y1_samples = y_1.sample((n_samples, 1))
        rho_1_samples = torch.hstack((x1_samples, y1_samples))
        # Compute BB loss alone
        # Rely on weight decay to approximate spectral normalization
        loss += losses.bb_loss(phi, rho_0_samples, rho_1_samples, n_samples)

        # loss for phi_1
        x_0 = x_samplers[1]
        y_0 = y_samplers[1]
        x_1 = x_samplers[2]
        y_1 = y_samplers[2]
        phi = phi_list[1]
        # Draw rho_0_samples
        x0_samples = x_0.sample((n_samples, 1))
        y0_samples = y_0.sample((n_samples, 1))
        rho_0_samples = torch.hstack((x0_samples, y0_samples))
        # Draw rho_1_samples
        x1_samples = x_1.sample((n_samples, 1))
        y1_samples = y_1.sample((n_samples, 1))
        rho_1_samples = torch.hstack((x1_samples, y1_samples))
        # Compute BB loss alone
        # Rely on weight decay to approximate spectral normalization
        loss += losses.bb_loss(phi, rho_0_samples, rho_1_samples, n_samples)

        loss.backward()
        opt.step()

    return phi_list, x_samplers, y_samplers


def train_three_boxes_R(
    phi_list,
    l_tensor,
    r_tensor,
    b_tensor,
    u_tensor,
    n_samples,
    matrix_hidden_dims,
    fro_reg_strength,
    gp_strength,
    lr,
    weight_decay,
    n_epochs,
    seed,
):
    """Trains R alone on the three-boxes toy example given trained phis.

    Implements the second step of the two-step training procedure.
    """

    torch.manual_seed(seed)  # Fix random seed

    # Generate samplers for the uniform dists on the three boxes
    x_samplers = []
    y_samplers = []
    box_0_x, box_0_y = samplers.generate_box_sampler(
        l_tensor[0], r_tensor[0], b_tensor[0], u_tensor[0]
    )
    x_samplers.append(box_0_x)
    y_samplers.append(box_0_y)

    box_1_x, box_1_y = samplers.generate_box_sampler(
        l_tensor[1], r_tensor[1], b_tensor[1], u_tensor[1]
    )
    x_samplers.append(box_1_x)
    y_samplers.append(box_1_y)

    box_2_x, box_2_y = samplers.generate_box_sampler(
        l_tensor[2], r_tensor[2], b_tensor[2], u_tensor[2]
    )
    x_samplers.append(box_2_x)
    y_samplers.append(box_2_y)

    # Initialize model
    space_dims = 2
    R = models.PSDMatrix(space_dims, matrix_hidden_dims).to(device)

    params = list(R.parameters())

    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    for epoch in range(n_epochs):
        opt.zero_grad()
        loss = 0

        # loss for first pair
        x_0 = x_samplers[0]
        y_0 = y_samplers[0]
        x_1 = x_samplers[1]
        y_1 = y_samplers[1]
        phi = phi_list[0]
        # Draw rho_0_samples
        x0_samples = x_0.sample((n_samples, 1))
        y0_samples = y_0.sample((n_samples, 1))
        rho_0_samples = torch.hstack((x0_samples, y0_samples))
        # Draw rho_1_samples
        x1_samples = x_1.sample((n_samples, 1))
        y1_samples = y_1.sample((n_samples, 1))
        rho_1_samples = torch.hstack((x1_samples, y1_samples))
        # Compute loss
        loss += fro_reg_strength * losses.fro_norm_regularizer(
            R, rho_0_samples, rho_1_samples, n_samples
        )
        loss += gp_strength * losses.gradient_penalty(
            phi, R, rho_0_samples, rho_1_samples, n_samples
        )

        # loss for phi_1
        x_0 = x_samplers[1]
        y_0 = y_samplers[1]
        x_1 = x_samplers[2]
        y_1 = y_samplers[2]
        phi = phi_list[1]
        # Draw rho_0_samples
        x0_samples = x_0.sample((n_samples, 1))
        y0_samples = y_0.sample((n_samples, 1))
        rho_0_samples = torch.hstack((x0_samples, y0_samples))
        # Draw rho_1_samples
        x1_samples = x_1.sample((n_samples, 1))
        y1_samples = y_1.sample((n_samples, 1))
        rho_1_samples = torch.hstack((x1_samples, y1_samples))
        # Compute loss
        loss += fro_reg_strength * losses.fro_norm_regularizer(
            R, rho_0_samples, rho_1_samples, n_samples
        )
        loss += gp_strength * losses.gradient_penalty(
            phi, R, rho_0_samples, rho_1_samples, n_samples
        )

        loss.backward()
        opt.step()

    return R, x_samplers, y_samplers


def twostep_train_three_boxes(
    l_tensor,
    r_tensor,
    b_tensor,
    u_tensor,
    n_samples,
    space_dims,
    scalar_hidden_dims,
    matrix_hidden_dims,
    fro_reg_strength,
    gp_strength,
    lr,
    weight_decay_phi,
    weight_decay_R,
    n_epochs_phi,
    n_epochs_R,
    seed,
):
    """Trains the model on the three-boxes toy example using the two-step procedure."""

    # First train the phi models
    phi_list, x_samplers, y_samplers = train_three_boxes_phis(
        l_tensor,
        r_tensor,
        b_tensor,
        u_tensor,
        n_samples,
        scalar_hidden_dims,
        lr,
        weight_decay_phi,
        n_epochs_phi,
        seed,
    )

    # Then train the R model given the learned phis
    R, x_samplers, y_samplers = train_three_boxes_R(
        phi_list,
        l_tensor,
        r_tensor,
        b_tensor,
        u_tensor,
        n_samples,
        matrix_hidden_dims,
        fro_reg_strength,
        gp_strength,
        lr,
        weight_decay_R,
        n_epochs_R,
        seed,
    )

    return R, phi_list, x_samplers, y_samplers


def train_eb_phis(data_type, n_samples, scalar_hidden_dims, lr, weight_decay, n_epochs, seed, gp_strength=0):
    """Trains the phis alone on the EB dataset from TrajectoryNet.

    Implements the first step of the two-step training procedure.
    """

    torch.manual_seed(seed)  # Fix random seed

    # Import eb5 data
    if data_type == "pca":
        data_file = "../labb/data/eb_velocity_v5.npz"
        data_list = utils.load_eb_data(data_file)
    elif data_type == "rescaled_pca":
        data_file = "../labb/data/eb_velocity_v5.npz"
        data_list = utils.load_rescaled_eb_data(data_file)
    elif data_type == "phate":
        data_file = "../labb/data/phate_data.npy"
        label_file = "../labb/data/phate_labels.npy"
        data_list = utils.load_eb_phate_data(data_file, label_file)

    # Generate samplers for the empirical distributions
    rho_0_list, rho_1_list = samplers.generate_eb_samplers(data_list)

    # Initialize models
    phi_list = []
    space_dims = 2
    for i in range(len(rho_0_list)):
        phi_list.append(models.ScalarFn(space_dims, scalar_hidden_dims).to(device))
    
    # Define params for optimization
    params = (list(phi_list[0].parameters()))
    for t in range(1, len(phi_list)):
        params = (params + list(phi_list[t].parameters()))

    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    for epoch in range(n_epochs):
        opt.zero_grad()
        loss = 0
        for i in range(len(rho_0_list)):
            rho_0 = rho_0_list[i]
            rho_1 = rho_1_list[i]
            rho_0_samples = rho_0.sample(n_samples)
            rho_1_samples = rho_1.sample(n_samples)
            phi = phi_list[i]
            # Compute BB loss alone
            # Rely on weight decay to approximate spectral normalization
            loss += losses.bb_loss(phi, rho_0_samples, rho_1_samples, n_samples)
            loss += gp_strength * losses.gradient_penalty(
                    phi, None, rho_0_samples, rho_1_samples, n_samples
                )
        loss.backward()
        opt.step()

    return phi_list, rho_0_list, rho_1_list


def train_eb_R(
    data_type,
    phi_list,
    n_samples,
    matrix_hidden_dims,
    fro_reg_strength,
    gp_strength,
    lr,
    weight_decay,
    n_epochs,
    seed,
):
    """Trains R alone on the EB dataset from TrajectoryNet.

    Implements the second step of the two-step training procedure.
    """

    torch.manual_seed(seed)  # Fix random seed

    # Import eb5 data
    if data_type == "pca":
        data_file = "../labb/data/eb_velocity_v5.npz"
        data_list = utils.load_eb_data(data_file)
    elif data_type == "rescaled_pca":
        data_file = "../labb/data/eb_velocity_v5.npz"
        data_list = utils.load_rescaled_eb_data(data_file)
    elif data_type == "phate":
        data_file = "../labb/data/phate_data.npy"
        label_file = "../labb/data/phate_labels.npy"
        data_list = utils.load_eb_phate_data(data_file, label_file)

    # Generate samplers for the empirical distributions
    rho_0_list, rho_1_list = samplers.generate_eb_samplers(data_list)

    # Initialize model
    space_dims = 2
    R = models.PSDMatrix(space_dims, matrix_hidden_dims).to(device)

    params = list(R.parameters())

    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    for epoch in range(n_epochs):
        opt.zero_grad()
        loss = 0
        for i in range(len(rho_0_list)):
            rho_0 = rho_0_list[i]
            rho_1 = rho_1_list[i]
            rho_0_samples = rho_0.sample(n_samples)
            rho_1_samples = rho_1.sample(n_samples)
            phi = phi_list[i]
            loss += fro_reg_strength * losses.fro_norm_regularizer(
                R, rho_0_samples, rho_1_samples, n_samples
            )
            loss += gp_strength * losses.gradient_penalty(
                phi, R, rho_0_samples, rho_1_samples, n_samples
            )
        loss.backward()
        opt.step()

    return R, rho_0_list, rho_1_list


def twostep_train_eb_model(
    data_type,
    n_samples,
    scalar_hidden_dims,
    matrix_hidden_dims,
    fro_reg_strength,
    gp_strength_phi,
    gp_strength_R,
    lr,
    weight_decay,
    n_epochs_phi,
    n_epochs_R,
    seed,
):
    """Trains the model on the EB data from TrajectoryNet using the two-step procedure."""

    # First train the phi models
    phi_list, rho_0_list, rho_1_list = train_eb_phis(
        data_type, n_samples, scalar_hidden_dims, lr, weight_decay, n_epochs_phi, seed
    )

    # Then train the R model given the learned phis
    R, rho_0_list, rho_1_list = train_eb_R(
        data_type,
        phi_list,
        n_samples,
        matrix_hidden_dims,
        fro_reg_strength,
        gp_strength_R,
        lr,
        weight_decay,
        n_epochs_R,
        seed,
    )

    return R, phi_list, rho_0_list, rho_1_list

def train_synthetic_phis(fpath, time_skip, n_samples, samples_per_batch, scalar_hidden_dims, lr, weight_decay, n_epochs, seed, gp_strength=0, R_model=None, scale_factor=1):
    """Trains the phis alone on synthetic data.

    Implements the first step of the two-step training procedure.
    """

    torch.manual_seed(seed)  # Fix random seed

    # Import synthetic data
    path = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), fpath))
    data = torch.load(path)
    data.requires_grad = False
    data = data.to(device)
    
    # Rescale data
    data *= scale_factor
    
    # If data contains just one trajectory and has only 3 dims, unsqueeze to obtain 4 dims
    if len(data.shape) == 3:
        data = torch.unsqueeze(data, dim=0)
    
    # Subsample across time
    # time_skip should be 1 or 3 for mass_splitting and x_paths
    # time_skip should be 1, 2, or 3 for gaussians_in_circles
    data = data[:, ::time_skip, :, :]
    
    # Subsample within each time point
    data = data[:, :, :n_samples, :]

    # Generate samplers for the empirical distributions
    rho_0_lists = [] # one rho_0_list for each trajectory
    rho_1_lists = [] # one rho_1_list for each trajectory
    for traj in range(data.shape[0]):
        rho_0_list, rho_1_list = samplers.generate_synthetic_samplers(data[traj])
        rho_0_lists.append(rho_0_list)
        rho_1_lists.append(rho_1_list)

    # Initialize models
    phi_list = []
    space_dims = 2
    for rho_0_list in rho_0_lists: # just pool all the phis together
        for i in range(len(rho_0_list)):
            phi_list.append(models.ScalarFn(space_dims, scalar_hidden_dims).to(device))

    # Define params for optimization
    params = (list(phi_list[0].parameters()))
    for t in range(1, len(phi_list)):
        params = (params + list(phi_list[t].parameters()))

    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    loss_vals = []

    for epoch in range(n_epochs):
        opt.zero_grad()
        loss = 0
        counter = 0
        for traj in range(len(rho_0_lists)):
            rho_0_list = rho_0_lists[traj]
            rho_1_list = rho_1_lists[traj]
            for i in range(len(rho_0_list)):
                rho_0 = rho_0_list[i]
                rho_1 = rho_1_list[i]
                rho_0_samples = rho_0.sample(samples_per_batch)
                #print("rho_0 x coord mean: " + str(torch.mean(rho_0_samples[:,0])))
                rho_1_samples = rho_1.sample(samples_per_batch)
                #print("rho_1 x coord mean: " + str(torch.mean(rho_1_samples[:,0])))
                phi = phi_list[counter]
                loss += losses.bb_loss(phi, rho_0_samples, rho_1_samples, samples_per_batch)
                loss += gp_strength * losses.gradient_penalty(
                    phi, R_model, rho_0_samples, rho_1_samples, samples_per_batch
                )
                counter += 1
        loss_vals.append(loss.detach().cpu())
        loss.backward()
        opt.step()
     
    final_loss = -1*loss_vals[-1] # multiply by -1 since we care about the sup(-f) = -inf(f)

    return phi_list, rho_0_lists, rho_1_lists, final_loss

def train_synthetic_R(
    fpath,
    phi_list,
    time_skip,
    n_samples,
    samples_per_batch,
    matrix_hidden_dims,
    fro_reg_strength,
    gp_strength,
    lr,
    weight_decay,
    n_epochs,
    seed,
    scale_factor=1
):
    """Trains R alone on synthetic data.

    Implements the second step of the two-step training procedure.
    """

    torch.manual_seed(seed)  # Fix random seed

    # Import synthetic data
    path = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), fpath))
    data = torch.load(path)
    data.requires_grad = False
    data = data.to(device)
    
    # Rescale data
    data *= scale_factor
    
    # If data contains just one trajectory and has only 3 dims, unsqueeze to obtain 4 dims
    if len(data.shape) == 3:
        data = torch.unsqueeze(data, dim=0)
    
    # Subsample across time
    # time_skip should be 1 or 3 for mass_splitting and x_paths
    # time_skip should be 1, 2, or 3 for gaussians_in_circles
    data = data[:, ::time_skip, :, :]
    
    # Subsample within each time point
    data = data[:, :, :n_samples, :]

    # Generate samplers for the empirical distributions
    rho_0_lists = [] # one rho_0_list for each trajectory
    rho_1_lists = [] # one rho_1_list for each trajectory
    for traj in range(data.shape[0]):
        rho_0_list, rho_1_list = samplers.generate_synthetic_samplers(data[traj])
        rho_0_lists.append(rho_0_list)
        rho_1_lists.append(rho_1_list)

    # Initialize model
    space_dims = 2
    R = models.PSDMatrixMultiLayer(space_dims, matrix_hidden_dims).to(device)

    params = R.parameters()

    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    loss_vals = []

    for epoch in range(n_epochs):
        opt.zero_grad()
        loss = 0
        counter = 0
        for traj in range(len(rho_0_lists)):
            rho_0_list = rho_0_lists[traj]
            rho_1_list = rho_1_lists[traj]
            for i in range(len(rho_0_list)):
                rho_0 = rho_0_list[i]
                rho_1 = rho_1_list[i]
                rho_0_samples = rho_0.sample(samples_per_batch)
                rho_1_samples = rho_1.sample(samples_per_batch)
                phi = phi_list[counter]
                loss += fro_reg_strength * losses.fro_norm_regularizer(
                    R, rho_0_samples, rho_1_samples, n_samples
                )
                loss += gp_strength * losses.gradient_penalty(
                    phi, R, rho_0_samples, rho_1_samples, n_samples
                )
                counter += 1
        loss_vals.append(loss.detach().cpu())
        loss.backward()
        opt.step()
    
    final_loss = loss_vals[-1]

    return R, rho_0_lists, rho_1_lists, final_loss

def twostep_train_synthetic(
    fpath,
    time_skip,
    n_samples,
    samples_per_batch,
    scalar_hidden_dims,
    matrix_hidden_dims,
    fro_reg_strength,
    gp_strength_phi,
    gp_strength_R,
    lr,
    weight_decay,
    n_epochs_phi,
    n_epochs_R,
    seed,
    scale_factor=1
):
    """Trains the model on synthetic data using the two-step procedure."""

    # First train the phi models
    phi_list, rho_0_lists, rho_1_lists, final_phi_loss = train_synthetic_phis(
        fpath, time_skip, n_samples, samples_per_batch, scalar_hidden_dims, lr, weight_decay, n_epochs_phi, seed, gp_strength_phi,
        scale_factor=scale_factor
    )

    # Then train the R model given the learned phis
    R, rho_0_lists, rho_1_lists, final_R_loss = train_synthetic_R(
        fpath,
        phi_list,
        time_skip,
        n_samples,
        samples_per_batch,
        matrix_hidden_dims,
        fro_reg_strength,
        gp_strength_R,
        lr,
        weight_decay,
        n_epochs_R,
        seed,
        scale_factor=scale_factor
    )

    return R, phi_list, rho_0_lists, rho_1_lists

def multistep_train_synthetic(
    fpath,
    time_skip,
    n_samples,
    samples_per_batch,
    scalar_hidden_dims,
    matrix_hidden_dims,
    fro_reg_strength,
    gp_strength_phi_initial,
    gp_strength_phi_later,
    gp_strength_R,
    lr,
    weight_decay,
    n_epochs_phi,
    n_epochs_R,
    n_steps,
    seed,
    R_init=None,
    scale_factor=1
):
    """Trains the model on synthetic data using the multi-step procedure."""

    phi_lists = []
    losses = []
    
    # First train the phi models
    phi_list, rho_0_lists, rho_1_lists, final_phi_loss = train_synthetic_phis(
        fpath, time_skip, n_samples, samples_per_batch, scalar_hidden_dims, lr, weight_decay, n_epochs_phi, seed,
        gp_strength_phi_initial, R_init, scale_factor=scale_factor
    )
    phi_lists.append(phi_list)
    losses.append(final_phi_loss)

    # Then train the R model given the learned phis
    R, rho_0_lists, rho_1_lists, final_R_loss = train_synthetic_R(
        fpath,
        phi_list,
        time_skip,
        n_samples,
        samples_per_batch,
        matrix_hidden_dims,
        fro_reg_strength,
        gp_strength_R,
        lr,
        weight_decay,
        n_epochs_R,
        seed,
        scale_factor=scale_factor
    )
    losses.append(final_R_loss)
    
    if n_steps > 1:
        for i in range(n_steps - 1):
            # phi step with current R
            phi_list, rho_0_lists, rho_1_lists, final_phi_loss = train_synthetic_phis(
                fpath, time_skip, n_samples, samples_per_batch, scalar_hidden_dims, lr, weight_decay, n_epochs_phi, seed,
                gp_strength_phi_later, R, scale_factor=scale_factor)
            phi_lists.append(phi_list)
            losses.append(final_phi_loss)
            
            # R step
            R, rho_0_lists, rho_1_lists, final_R_loss = train_synthetic_R(fpath,
                                                                          phi_list,
                                                                          time_skip,
                                                                          n_samples,
                                                                          samples_per_batch,
                                                                          matrix_hidden_dims,
                                                                          fro_reg_strength,
                                                                          gp_strength_R,
                                                                          lr,
                                                                          weight_decay,
                                                                          n_epochs_R,
                                                                          seed,
                                                                          scale_factor=scale_factor
                                                                         )
            losses.append(final_R_loss)  

    return R, phi_lists, rho_0_lists, rho_1_lists, losses

def train_wot_phis(data_list, n_samples, scalar_hidden_dims, lr, weight_decay, n_epochs, seed, gp_strength=0):
    """Trains the phis alone on the WOT dataset.

    Implements the first step of the two-step training procedure.
    """

    torch.manual_seed(seed)  # Fix random seed

    # Generate samplers for the empirical distributions
    rho_0_list, rho_1_list = samplers.generate_eb_samplers(data_list)

    # Initialize models
    phi_list = []
    space_dims = 2
    for i in range(len(rho_0_list)):
        phi_list.append(models.ScalarFn(space_dims, scalar_hidden_dims).to(device))
    
    # Define params for optimization
    params = (list(phi_list[0].parameters()))
    for t in range(1, len(phi_list)):
        params = (params + list(phi_list[t].parameters()))

    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    for epoch in range(n_epochs):
        opt.zero_grad()
        loss = 0
        for i in range(len(rho_0_list)):
            rho_0 = rho_0_list[i]
            rho_1 = rho_1_list[i]
            rho_0_samples = rho_0.sample(n_samples)
            rho_1_samples = rho_1.sample(n_samples)
            phi = phi_list[i]
            loss += losses.bb_loss(phi, rho_0_samples, rho_1_samples, n_samples)
            loss += gp_strength * losses.gradient_penalty(
                    phi, None, rho_0_samples, rho_1_samples, n_samples
                )
        loss.backward()
        opt.step()

    return phi_list, rho_0_list, rho_1_list


def train_wot_R(
    data_list,
    phi_list,
    n_samples,
    matrix_hidden_dims,
    fro_reg_strength,
    identity_reg_strength,
    gp_strength,
    lr,
    weight_decay,
    n_epochs,
    seed,
    R_model_type
):
    """Trains R alone on the EB dataset from TrajectoryNet.

    Implements the second step of the two-step training procedure.
    """

    torch.manual_seed(seed)  # Fix random seed

    # Generate samplers for the empirical distributions
    rho_0_list, rho_1_list = samplers.generate_eb_samplers(data_list)

    # Initialize model
    space_dims = 2
    radius = 15
    if R_model_type == "multilayer":
        R = models.PSDMatrixMultiLayer(space_dims, matrix_hidden_dims).to(device)
    elif R_model_type == "singlelayer":
        R = models.PSDMatrix(space_dims, matrix_hidden_dims).to(device)

    params = list(R.parameters())

    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    for epoch in range(n_epochs):
        opt.zero_grad()
        loss = 0
        for i in range(len(rho_0_list)):
            rho_0 = rho_0_list[i]
            rho_1 = rho_1_list[i]
            rho_0_samples = rho_0.sample(n_samples)
            rho_1_samples = rho_1.sample(n_samples)
            phi = phi_list[i]
            loss += identity_reg_strength * losses.identity_regularizer(
                R, radius, n_samples
            )
            loss += fro_reg_strength * losses.fro_norm_regularizer(
                R, rho_0_samples, rho_1_samples, n_samples
            )
            loss += gp_strength * losses.gradient_penalty(
                phi, R, rho_0_samples, rho_1_samples, n_samples
            )
        loss.backward()
        opt.step()

    return R, rho_0_list, rho_1_list


def twostep_train_wot_model(
    data_list,
    n_samples,
    scalar_hidden_dims,
    matrix_hidden_dims,
    fro_reg_strength,
    identity_reg_strength,
    gp_strength_phi,
    gp_strength_R,
    lr,
    weight_decay,
    n_epochs_phi,
    n_epochs_R,
    seed,
    R_model_type="multilayer"
):
    """Trains the model on the EB data from TrajectoryNet using the two-step procedure."""

    # First train the phi models
    phi_list, rho_0_list, rho_1_list = train_wot_phis(
        data_list, n_samples, scalar_hidden_dims, lr, weight_decay, n_epochs_phi, seed, gp_strength_phi
    )

    # Then train the R model given the learned phis
    R, rho_0_list, rho_1_list = train_wot_R(
        data_list,
        phi_list,
        n_samples,
        matrix_hidden_dims,
        fro_reg_strength,
        identity_reg_strength,
        gp_strength_R,
        lr,
        weight_decay,
        n_epochs_R,
        seed,
        R_model_type
    )

    return R, phi_list, rho_0_list, rho_1_list