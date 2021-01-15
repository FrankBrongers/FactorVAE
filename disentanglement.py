import numpy as np
import random
import torch


class disentanglement_score:
    def __init__(self, model, dataloader, z_dim, subset_fraction=.1, subset_size=None):
        self.dataloader = dataloader
        self.classes = dataloader.dataset.latents_classes
        l = dataloader.len()

        self.max_classes = dataloader.dataset.latents_classes[-1]
        self.count_classes = len(self.max_classes)

        self.z_dim = z_dim

        if not subset_size:
            subset_size = numpy.round(l*subset_fraction)

        assert l >= subset_size

        subset = random.sample(range(0, l), subset_size)
        latents = torch.zeros(subset_size, z_dim*2)

        for c, idx in enumerate(subset):
            input, _ = dataloader[idx]
            z = self.model(input, no_dec=True).cpu().detach()
            latents[c] = z

        self.emp_std = torch.std(latents, dim=1, unbiased=True, keepdim=True)

    def fixed_factor_latents(model, fixed_factor_index=1, sample_size):
        fixed_factor = random.randrange(self.max_classes[fixed_factor_index]+1)
        sample_indices = numpy.random.choice(np.where(self.classes[:, fixed_factor_index] == fixed_factor)[0])

        latents = torch.zeros(sample_size, z_dim*2)
        for c, idx in enumerate(sample_indices):
            input, _ = dataloader[idx]
            z = self.model(input, no_dec=True).cpu().detach()
            latents[c] = z

        return latents

    def score(model, sample_size):
        factors_indices = np.arange(self.count_classes)[1:]

        for idx in factors_indices:
            latents = fixed_factor_latents(model, idx, sample_size)
            norm_latents = latents/self.emp_std
            emp_var = self.empirical_variance(norm_latents, sample_size)
            vote = torch.argmin(input, dim=1)

    def empirical_variance(latents, sample_size):
        denom = (2*sample_size*(sample_size-1))

        sum_d = torch.zeros(self.z_dim)
        for i in latents:
            d_i = torch.sum((latents - latents[i])**2, dim=1)
            d_i = d_i[:z_dim] + d_i[z_dim:]

            sum_d += d_i

        return sum_d/denom
