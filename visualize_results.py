import argparse, os, json
import matplotlib.pyplot as plt
import numpy as np

from collections import defaultdict


def main(args):
    names = args.names.split()

    all_files = os.listdir(args.dir)
    files = [file for file in all_files if file[:-7] in names]

    assert len(files) > 0, 'No files found'

    data = defaultdict(list)
    for file in files:
        data[file[:-7]].append(load_data(os.path.join(args.dir, file)))

    variables = ['vae_recon_loss', 'vae_kld', 'vae_tc_loss', 'D_tc_loss', 'dis_score']
    full_variables = ['Reconstruction Loss', 'KL-divergence', 'VAE TC-Loss', 'Discriminator TC-Loss', 'Disentanglement Score']

    for i, v in enumerate(variables):
        fig = plt.figure()
        for name in names:
            outputs = [file['outputs'][v] for file in data[name]]
            iteration = data[name][0]['outputs']['iteration']
            mean = np.mean(outputs, axis=0)
            std = np.std(outputs, axis=0)

            plt.errorbar(iteration, mean, yerr=std, label=r'$\gamma$'+str(data[name][0]['args']['gamma']))

        plt.xlabel('Iterations')
        plt.ylabel(full_variables[i])
        plt.show()


def load_data(path):
    with open(path) as f:
        data = json.load(f)

    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Factor-VAE')

    parser.add_argument('--names', default='main', type=str, help='names of the experiments to be plotted')
    parser.add_argument('--dir', default='vars', type=str, help='name of the directory holding the results')
    parser.add_argument('--output_dir', default='results', type=str, help='name of the directory holding the results')

    args = parser.parse_args()

    main(args)
