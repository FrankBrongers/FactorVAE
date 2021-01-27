import os
import numpy as np
import torch
from torchvision import save_image, make_grid
from model import FactorVAE1
from dataset import return_data


np.random.seed(0)


def load_checkpoint(model, ckptname, verbose=True):
    filepath = os.path.join(self.ckpt_dir, ckptname)
    if os.path.isfile(filepath):
        with open(filepath, 'rb') as f:
            checkpoint = torch.load(f)

        model.load_state_dict(checkpoint['model_states']['VAE'])
        if verbose:
            print("loaded checkpoint '{}'".format(filepath))
    else:
        if verbose:
            print("no checkpoint found at '{}'".format(filepath))


def process_imgs(input, recon, gcam):
    input = input - torch.min(input)
    input = input / torch.max(input)
    recon = recon - torch.min(recon)
    recon = recon / torch.max(recon)
    gcam = gcam - torch.min(gcam)
    gcam = gcam / torch.max(gcam)
    overlay = gcam + recon
    overlay = overlay / torch.max(overlay)

    im_list = torch.cat(input, recon, gcam)
    return im_list


def main(ckptname, z_dim=32, target_layer=0, image_size=64):
    model = FactorVAE1(z_dim)
    load_checkpoint(model, ckptname)

    gcam = GradCAM(model.encode, target_layer, device, image_size)

    _, dataset = return_data(args)

    factors = dataset.latents_classes
    max_factors = dataset.latents_classes[-1]
    n_factors = len(max_factors)

    all_indices = np.array([])
    for k_fixed in range(n_factors):
        fixed_factor = np.random.randint(0, max_factors[k_fixed]+1)
        sample_index = np.random.choice(np.where(factors[:, k_fixed] == fixed_factor)[0], size=1)
        all_indices = np.append(all_indices, sample_index)

    x = dataset[all_indices][0]
    x_recon, mu, logvar, z = model(x)

    cam = gcam.generate(z)
    response = cam.flatten(1).sum(1).flatten()
    picked_cam = cam[torch.max(response)]

    x, x_recon =

    im_list = process_imgs(x, x_recon, picked_cam)
