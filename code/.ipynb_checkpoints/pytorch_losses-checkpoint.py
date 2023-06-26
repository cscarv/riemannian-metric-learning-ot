import torch
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def bb_loss(phi, rho_0_samples, rho_1_samples, n_samples):
    """Computes the Benamou-Brenier loss term given n_samples from distributions rho_0 and rho_1
    with scalar potential phi."""

    phi_0 = phi(rho_0_samples)
    phi_1 = phi(rho_1_samples)

    loss = torch.sum(phi_1 - phi_0)

    return loss


def fro_norm_regularizer(R, rho_0_samples, rho_1_samples, n_samples):
    """Computes the 1/||R||_F^2 regularizer given n_samples from distributions rho_0 and rho_1.

    We enforce the regularization at points drawn uniformly from lines connecting rho_0 and rho_1 as
    in the gradient penalty.
    """

    t = torch.rand(n_samples, 1, device=device)
    p = t * rho_0_samples + (1 - t) * rho_1_samples

    R_at_p = R(p)

    loss = torch.sum(1 / (torch.linalg.norm(R_at_p, dim=(1, 2)) ** 2))

    return loss

def non_inv_fro_norm_regularizer(R, rho_0_samples, rho_1_samples, n_samples):
    """Computes the ||R||_F^2 regularizer given n_samples from distributions rho_0 and rho_1.

    We enforce the regularization at points drawn uniformly from lines connecting rho_0 and rho_1 as
    in the gradient penalty.
    """

    t = torch.rand(n_samples, 1, device=device)
    p = t * rho_0_samples + (1 - t) * rho_1_samples

    R_at_p = R(p)

    loss = torch.sum((torch.linalg.norm(R_at_p, dim=(1, 2)) ** 2))

    return loss

def inv_nuc_norm_regularizer(R, rho_0_samples, rho_1_samples, n_samples):
    """Computes the ||R^-1||_nuc regularizer given n_samples from distributions rho_0 and rho_1.

    We enforce the regularization at points drawn uniformly from lines connecting rho_0 and rho_1 as
    in the gradient penalty.
    """

    t = torch.rand(n_samples, 1, device=device)
    p = t * rho_0_samples + (1 - t) * rho_1_samples

    inv_R_at_p = torch.linalg.inv(R(p)) # this might be a suboptimal way of computing the inverse

    loss = torch.sum(torch.linalg.norm(inv_R_at_p, dim=(1, 2), ord='nuc'))

    return loss

def identity_regularizer(R, radius, n_samples):
    """Computes the ||I - R||^2 regularizer given n_samples uniform samples from box of given radius.
    """

    p = 2*radius*(torch.rand(n_samples, 2, device=device) - 0.5)

    R_at_p = R(p)
    
    eye = torch.eye(2, dtype=torch.float64, device=device)
    eye = eye.repeat(n_samples, 1, 1)

    loss = -1*torch.sum(torch.linalg.norm(eye - R_at_p, dim=(1, 2)) ** 2)

    return loss


def gradient_penalty(phi, R, rho_0_samples, rho_1_samples, n_samples):
    """Evaluates the gradient penalty at points drawn uniformly from lines connecting rho_0 and
    rho_1 as in Gulrajani et al."""

    t = torch.rand(n_samples, 1, device=device)
    p = t * rho_0_samples + (1 - t) * rho_1_samples
    p.requires_grad = True
    
    if R is not None:
        grads = torch.autograd.grad(
            outputs=phi(p), inputs=p, grad_outputs=torch.ones_like(phi(p)), create_graph=True
        )[0].unsqueeze(dim=2)

        R_at_p = R(p)
        R_grad_phi = R_at_p @ grads

        loss = torch.sum(F.softplus(-1 + 0.5 * torch.sum(grads * R_grad_phi, dim=1)))
    
    else:
        grads = torch.autograd.grad(
            outputs=phi(p), inputs=p, grad_outputs=torch.ones_like(phi(p)), create_graph=True
        )[0]
        
        loss = torch.sum(F.softplus(-1 + 0.5 * torch.linalg.vector_norm(grads, dim=1)))

    return loss


def loss_fn(phi, R, rho_0_samples, rho_1_samples, n_samples, fro_reg_strength, gp_strength):
    """Computes total loss function."""

    bb_loss_val = bb_loss(phi, rho_0_samples, rho_1_samples, n_samples)
    fro_reg_val = fro_norm_regularizer(R, rho_0_samples, rho_1_samples, n_samples)
    gp_val = gradient_penalty(phi, R, rho_0_samples, rho_1_samples, n_samples)

    return bb_loss_val + fro_reg_strength * fro_reg_val + gp_strength * gp_val
