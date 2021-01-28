import os, argparse
import numpy as np
import torch
import cv2
from torchvision.utils import save_image, make_grid
from model import FactorVAE1

from dataset import return_data
from gradcam import GradCAM


np.random.seed(0)


def load_checkpoint(model, ckpt_dir, ckptname, device, verbose=True):
    filepath = os.path.join(ckpt_dir, ckptname)
    if os.path.isfile(filepath):
        with open(filepath, 'rb') as f:
            checkpoint = torch.load(f, map_location=device)

        model.load_state_dict(checkpoint['model_states']['VAE'])
        if verbose:
            print("loaded checkpoint '{}'".format(filepath))
        return True
    else:
        if verbose:
            print("no checkpoint found at '{}'".format(filepath))
        return False


def process_imgs(input, recon, gcam, second_cam, n_factors):
    input = input - torch.min(input)
    input = input / torch.max(input)
    recon = recon - torch.min(recon)
    recon = recon / torch.max(recon)
    gcam = gcam - torch.min(gcam)
    gcam = gcam / torch.max(gcam)
    second_cam = second_cam - torch.min(second_cam)
    second_cam = second_cam / torch.max(second_cam)

    input = make_grid(input, nrow=n_factors, normalize=False).transpose(0, 2).transpose(0, 1).detach().numpy()
    recon = make_grid(recon, nrow=n_factors, normalize=False).transpose(0, 2).transpose(0, 1).detach().numpy()
    gcam = make_grid(gcam, nrow=n_factors, normalize=False).transpose(0, 2).transpose(0, 1).detach().numpy()
    second_cam = make_grid(second_cam, nrow=n_factors, normalize=False).transpose(0, 2).transpose(0, 1).detach().numpy()

    return input, recon, gcam, second_cam

def add_heatmap(input, gcam):
    gcam = cv2.applyColorMap(np.uint8(255 * gcam), cv2.COLORMAP_JET)
    gcam = np.asarray(gcam, dtype=np.float) + np.asarray(input, dtype=np.float)
    gcam = 255 * gcam / np.max(gcam)

    return np.uint8(gcam)


def main(args):
    use_cuda = args.cuda and torch.cuda.is_available()
    device = 'cuda' if use_cuda else 'cpu'

    model = FactorVAE1(args.z_dim).to(device)
    model_found = load_checkpoint(model, args.dir, args.name, device)

    if not model_found:
        return

    gcam = GradCAM(model.encode, args.target_layer, device, args.image_size)

    _, dataset = return_data(args)

    factors = dataset.latents_classes
    max_factors = dataset.latents_classes[-1]
    n_factors = len(max_factors)

    all_indices = np.array([])
    for k_fixed in range(n_factors):
        fixed_factor = np.random.randint(0, max_factors[k_fixed]+1)
        sample_index = np.random.choice(np.where(factors[:, k_fixed] == fixed_factor)[0], size=1)
        all_indices = np.append(all_indices, sample_index)

    x = dataset[all_indices][0].to(device)
    x_recon, mu, logvar, z = model(x)

    x, x_recon = x.repeat(1, 3, 1, 1), x_recon.repeat(1, 3, 1, 1)

    cam = gcam.generate(z)
    # response = cam.flatten(1).sum(1)
    # picked_cam = cam[torch.argmax(response).item()].unsqueeze(1)
    # cam = torch.cat((cam[:torch.argmax(response).item()], cam[torch.argmax(response).item()+1:]))
    # second_cam = cam[torch.argmax(response).item()].unsqueeze(1)

    cam = cam.transpose(0,1)

    first_cam, second_cam = [], []

    for i in cam:
        response = i.flatten(1).sum(1)
        argmax = torch.argmax(response).item()
        first_cam.append(cam[argmax])
        ncam = torch.cat((cam[:argmax], cam[argmax+1:]))
        second_cam.append(ncam[torch.argmax(response).item()])

    first_cam = (torch.cat(first_cam)).unsqueeze(1)
    second_cam = (torch.cat(second_cam)).unsqueeze(1)

    input, recon, picked_cam, second_cam = process_imgs(x.detach(), x_recon.detach(), picked_cam.detach(), second_cam.detach(), n_factors)

    heatmap = add_heatmap(input, picked_cam)
    heatmap2 = add_heatmap(input, second_cam)

    input = np.uint8(np.asarray(input, dtype=np.float)*255)
    grid = np.concatenate((input, heatmap, heatmap2))

    cv2.imshow('filename', grid)
    cv2.waitKey(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualizer')

    parser.add_argument('--name', default='main', type=str, help='name of the model to be visualized')
    parser.add_argument('--dir', default='checkpoints', type=str, help='name of the directory holding the models weights')
    parser.add_argument('--output_dir', default='visualizations', type=str, help='name of the directory holding the visualizations')
    parser.add_argument('--cuda', type=bool, const=True, default=False, nargs='?', help='add if the gpu should be used')
    parser.add_argument('--z_dim', default=32, type=int, help='dimension of the representation z')
    parser.add_argument('--target_layer', type=str, default='0', help='target layer for the attention maps')

    parser.add_argument('--dset_dir', default='data', type=str, help='dataset directory')
    parser.add_argument('--dataset', default='dsprites', type=str, help='dataset name')
    parser.add_argument('--image_size', default=64, type=int, help='image size. now only (64,64) is supported')
    parser.add_argument('--num_workers', default=1, type=int, help='dataloader num_workers')
    parser.add_argument('--batch_size', default=1, type=int, help='place holder')

    args = parser.parse_args()

    main(args)
