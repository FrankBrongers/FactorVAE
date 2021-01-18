import numpy as np
import random
import torch

from torch.utils.data import SubsetRandomSampler
from torch.utils.data import Dataset, DataLoader


class Disentanglement_score:
    def __init__(self, model, args, dataset, device):
        self.z_dim = args.z_dim
        self.batch_size = args.score_batch_size
        self.num_workers = args.num_workers
        self.device = device
        self.subset_size = args.subset_size
        self.sample_size = args.L
        self.vote_count = args.vote_count

        self.dataset = dataset
        self.classes = dataset.latents_classes
        self.max_classes = dataset.latents_classes[-1]
        self.count_classes = len(self.max_classes)

        self.factor_idxs = [int(i) for i in args.factor_idxs.split(',')]

        l = len(dataset)

        if not self.subset_size:
            self.subset_size = int(np.round(l*args.subset_fraction))

        assert l >= self.subset_size, 'subset cannot be bigger than the dataset'

        subset = random.sample(range(0, l), self.subset_size)

        dataloader = self.create_subsetloader(subset)

        c = 0
        latents = torch.zeros(self.subset_size, self.z_dim*2, requires_grad=False).to(self.device)
        for input, _ in dataloader:
            z = model.encode(input.to(self.device)).detach()
            c2 = c+z.size(0)
            latents[c:c2] = torch.flatten(z, 1)
            c = c2

        self.emp_std = torch.std(latents, dim=0, unbiased=True).cpu()

    def fixed_factor_latents(self, model, fixed_factor_index):
        fixed_factor = random.randrange(self.max_classes[fixed_factor_index]+1)
        sample_indices = np.random.choice(np.where(self.classes[:, fixed_factor_index] == fixed_factor)[0], size=self.sample_size)

        latents = model.encode(self.dataset[sample_indices][0].to(self.device))

        return latents.detach().flatten(start_dim=1)

    def score(self, model):
        tally = torch.zeros(self.z_dim, max(self.factor_idxs)+1, requires_grad=False).to(self.device)

        for _ in range(self.vote_count):
            fixed_factor_index = random.choice(self.factor_idxs)
            latents = self.fixed_factor_latents(model, fixed_factor_index)
            norm_latents = latents/self.emp_std.to(self.device)

            emp_var = self.empirical_variance(norm_latents)

            vote = torch.argmin(emp_var)
            tally[vote, fixed_factor_index] += 1

        return torch.sum(torch.max(tally, dim=1).values).cpu()/self.vote_count

    def empirical_variance(self, latents):
        denom = (2*self.sample_size*(self.sample_size-1))

        sum_d = torch.zeros(self.z_dim).to(self.device)
        for latent in latents:
            d_i = torch.sum((latents - latent)**2, dim=0)
            d_i = d_i[:self.z_dim] + d_i[self.z_dim:]

            sum_d += d_i

        return sum_d/denom

    def create_subsetloader(self, indices):
        sampler = SubsetRandomSampler(indices)
        dataloader = DataLoader(self.dataset,
                                batch_size=self.batch_size,
                                sampler=sampler,
                                num_workers=self.num_workers,
                                pin_memory=True,
                                drop_last=False)

        return dataloader
