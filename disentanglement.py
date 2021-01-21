import numpy as np
import random
import torch

from torch.utils.data import SubsetRandomSampler
from torch.utils.data import Dataset, DataLoader

from ops import kl_divergence


def disentanglement_score(model, device, dataset, z_dim, L=100, n_votes=800, verbose=False):
    factors = dataset.latents_classes
    max_factors = dataset.latents_classes[-1]
    n_factors = len(max_factors)

    n_votes_per_factor = int(n_votes / n_factors)

    all_latents = []
    # Fix a factor k
    for k_fixed in range(n_factors):
        # Generate training examples for this factor
        for _ in range(n_votes_per_factor):
            # Fix a value for this factor
            fixed_factor = np.random.randint(0, max_factors[k_fixed]+1)
            sample_indices = np.random.choice(np.where(factors[:, k_fixed] == fixed_factor)[0], size=L)

            latents = model.encode(dataset[sample_indices][0].to(device)).detach().cpu().flatten(start_dim=1)
            all_latents.append(latents)

    # Concatenate every code
    all_latents = torch.cat(all_latents)

    # Now, lets compute the KL divergence of each dimension wrt the prior
    emp_mean_kl = kl_divergence(all_latents[:, :z_dim], all_latents[:, z_dim:], dim_wise=True)

    # Throw the dimensions that collapsed to the prior
    kl_tol = 1e-5
    useful_dims = np.where(emp_mean_kl.numpy() > kl_tol)[0]
    u_dim = len(useful_dims)

    print(useful_dims)

    if verbose:
        print(u_dim)

    # useful_dims = np.concatenate((useful_dims, useful_dims+z_dim), axis=None)
    print(useful_dims)

    # Compute scales for useful dims
    scales = torch.std(all_latents[:, useful_dims], axis=0)

    all_latents = all_latents[:, useful_dims]/scales

    if verbose:
        print("Empirical mean for kl dimension-wise:")
        print(list(emp_mean_kl))
        print("Useful dimensions:", list(useful_dims), " - Total:", useful_dims.shape[0])
        print("Empirical Scales:", list(scales))

    r1 = 0
    v_matrix = torch.zeros((u_dim, n_factors))

    # Fix a factor k
    for k_fixed in range(n_factors):
        # Generate training examples for this factor
        for i in range(n_votes_per_factor):
            r2 = r1 + L
            norm_latents = all_latents[r1:r2]
            # Take the empirical variance in each dimension of these normalised representations
            emp_var = torch.var(norm_latents, axis=0)  # dimension (z_dim,), variance for each dimension of code
            # Then the index of the dimension with the lowest variance...
            d_j = torch.argmin(emp_var)
            # ...and the target index k provide one training input/output example for the classifier majority vote
            v_matrix[d_j, k_fixed] += 1

            r1 = r2

    if verbose:
        print("Votes:")
        print(v_matrix.numpy(), torch.sum(v_matrix, dim=0), torch.sum(v_matrix, dim=1))

    # Since both inputs and outputs lie in a discrete space, the optimal classifier is the majority-vote classifier
    # and the metric is the error rate of the classifier (actually they show the accuracy in the paper)

    return torch.sum(torch.max(v_matrix, dim=1).values)/n_votes
