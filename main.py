"""main.py"""

import argparse
import numpy as np
import torch

from solver import Solver
from utils import str2bool

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def main(args):
    print(vars(args))
    return
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    net = Solver(args)
    net.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Factor-VAE')

    parser.add_argument('--seed', default=0, type=int, help='the seed')

    parser.add_argument('--name', default='main', type=str, help='name of the experiment')
    parser.add_argument('--cuda', default=True, type=str2bool, help='enable cuda')
    parser.add_argument('--max_iter', default=1e6, type=float, help='maximum training iteration')
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')

    parser.add_argument('--z_dim', default=10, type=int, help='dimension of the representation z')
    parser.add_argument('--gamma', default=6.4, type=float, help='gamma hyperparameter')
    parser.add_argument('--lr_VAE', default=1e-4, type=float, help='learning rate of the VAE')
    parser.add_argument('--beta1_VAE', default=0.9, type=float, help='beta1 parameter of the Adam optimizer for the VAE')
    parser.add_argument('--beta2_VAE', default=0.999, type=float, help='beta2 parameter of the Adam optimizer for the VAE')
    parser.add_argument('--lr_D', default=1e-4, type=float, help='learning rate of the discriminator')
    parser.add_argument('--beta1_D', default=0.5, type=float, help='beta1 parameter of the Adam optimizer for the discriminator')
    parser.add_argument('--beta2_D', default=0.9, type=float, help='beta2 parameter of the Adam optimizer for the discriminator')

    parser.add_argument('--subset_size', default=0, type=int, help='size of the subset used for empirical std in the disentanglement score, overwrites subset_fraction')
    parser.add_argument('--subset_fraction', default=0.5, type=float, help='fraction of the subset used for empirical std in the disentanglement score')
    parser.add_argument('--L', default=100, type=int, help='L used for creating the votes in the disentanglement score')
    parser.add_argument('--vote_count', default=800, type=int, help='Amount of votes needed for the disentanglement score')
    parser.add_argument('--score_batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--dis_score', default=True, type=str2bool, help='Whether the disentanglement score should be used')

    parser.add_argument('--dset_dir', default='data', type=str, help='dataset directory')
    parser.add_argument('--dataset', default='dsprites', type=str, help='dataset name')
    parser.add_argument('--image_size', default=64, type=int, help='image size. now only (64,64) is supported')
    parser.add_argument('--num_workers', default=2, type=int, help='dataloader num_workers')

    parser.add_argument('--viz_on', default=False, type=str2bool, help='enable visdom visualization')
    parser.add_argument('--viz_port', default=8097, type=int, help='visdom port number')
    parser.add_argument('--viz_ll_iter', default=1000, type=int, help='visdom line data logging iter')
    parser.add_argument('--viz_la_iter', default=5000, type=int, help='visdom line data applying iter')
    parser.add_argument('--viz_ra_iter', default=10000, type=int, help='visdom recon image applying iter')
    parser.add_argument('--viz_ta_iter', default=10000, type=int, help='visdom traverse applying iter')

    parser.add_argument('--print_iter', default=500, type=int, help='print losses iter')

    parser.add_argument('--ckpt_dir', default='checkpoints', type=str, help='checkpoint directory')
    parser.add_argument('--ckpt_load', default=None, type=str, help='checkpoint name to load')
    parser.add_argument('--ckpt_save_iter', default=10000, type=int, help='checkpoint save iter')

    parser.add_argument('--output_dir', default='outputs', type=str, help='output directory')
    parser.add_argument('--output_save', default=True, type=str2bool, help='whether to save traverse results')

    parser.add_argument('--vars_dir', default='vars', type=str, help='vars directory, saves the arguments and outputs')
    parser.add_argument('--vars_save', default=True, type=str2bool, help='whether to save traverse results')

    args = parser.parse_args()

    main(args)
