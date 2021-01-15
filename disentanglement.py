import numpy as np
import random
import torch

from torch.utils.data import SubsetRandomSampler


class disentanglement_score:
    def __init__(self, model, dataset, args, z_dim, subset_fraction=.1, subset_size=None):
        self.z_dim = args.z_dim
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.device = args.device

        self.dataset = dataset
        self.classes = dataset.latents_classes
        self.max_classes = dataset.latents_classes[-1]
        self.count_classes = len(self.max_classes)

        l = dataset.len()

        if not subset_size:
            subset_size = np.round(l*subset_fraction)

        assert l >= subset_size, 'subset cannot be bigger than the dataset'

        subset = random.sample(range(0, l), subset_size)

        dataloader = create_subsetloader(subset)

        latents = torch.zeros(sample_size, z_dim*2)
        for c, input in enumerate(dataloader):
            z = self.model(input.to(self.device), no_dec=True).cpu().detach()
            latents[c:c+z.size(0)] = z

        self.emp_std = torch.std(latents, dim=1, unbiased=True, keepdim=True)

    def fixed_factor_latents(self, model, fixed_factor_index=1, sample_size):
        fixed_factor = random.randrange(self.max_classes[fixed_factor_index]+1)
        sample_indices = numpy.random.choice(np.where(self.classes[:, fixed_factor_index] == fixed_factor)[0])

        dataloader = create_subsetloader(sample_indices)

        c = 0
        latents = torch.zeros(sample_size, z_dim*2)
        for input in enumerate(dataloader):
            z = self.model(input.to(self.device), no_dec=True).cpu().detach()
            c2 = c+z.size(0)
            latents[c:c2] = z
            c = c2

        return latents

    def score(self, model, sample_size):
        factors_indices = np.arange(self.count_classes)[1:]

        for idx in factors_indices:
            latents = fixed_factor_latents(model, idx, sample_size)
            norm_latents = latents/self.emp_std
            emp_var = self.empirical_variance(norm_latents, sample_size)
            vote = torch.argmin(input, dim=1)

    def empirical_variance(self, latents, sample_size):
        denom = (2*sample_size*(sample_size-1))

        sum_d = torch.zeros(self.z_dim)
        for i in latents:
            d_i = torch.sum((latents - latents[i])**2, dim=1)
            d_i = d_i[:z_dim] + d_i[z_dim:]

            sum_d += d_i

        return sum_d/denom

    def create_subsetloader(self, indices):
        sampler = SubsetRandomSampler(sample_indices)
        dataloader = DataLoader(self.dataset,
                                batch_size=self.batch_size,
                                sampler=sampler,
                                num_workers=self.num_workers,
                                pin_memory=True,
                                drop_last=False)

        return dataloader
