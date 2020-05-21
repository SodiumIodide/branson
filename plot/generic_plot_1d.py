#!usr/bin/env python3

import h5py
import os
import sys
import imageio
import numpy as np
import matplotlib.pyplot as plt

from plot_1d import get_args, sort_nicely

TR_IMAGE_PATHS = []
TE_IMAGE_PATHS = []
TR_IMAGES = []
TE_IMAGES = []

def plot(args):
    directory = args.directory
    h5files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and f.endswith(".h5")]
    # Find maximum (one loop)
    max_t_r = sys.float_info.min
    max_t_e = sys.float_info.min
    min_t_r = sys.float_info.max
    min_t_e = sys.float_info.max
    for h5file in h5files:
        file = h5py.File(h5file, 'r')
        max_t_r = max(file['t_r']) if max(file['t_r']) > max_t_r else max_t_r
        max_t_e = max(file['t_e']) if max(file['t_e']) > max_t_e else max_t_e
        min_t_r = min(file['t_r']) if min(file['t_r']) < min_t_r else min_t_r
        min_t_e = min(file['t_e']) if min(file['t_e']) < min_t_e else min_t_e
    sort_nicely(h5files)
    for imagenumber, h5file in enumerate(h5files):
        file = h5py.File(h5file, 'r')
        x_boundaries = file['x']
        x_data = []
        for i in range(len(x_boundaries) - 1):
            x_data.append((x_boundaries[i] + x_boundaries[i + 1]) / 2.0)
        tr = file['t_r']
        te = file['t_e']
        plt.plot(x_data, tr)
        plt.ylim(ymin=min_t_r, ymax=max_t_r)
        plt.yscale('log')
        plt.grid(which='both', axis='both')
        plt.title(f"T_r Timestep {imagenumber + 1}")
        plt.tight_layout()
        plt.savefig(f"t_r_{imagenumber + 1}.png")
        TR_IMAGE_PATHS.append(f"t_r_{imagenumber + 1}.png")
        plt.cla()
        plt.clf()
        plt.plot(x_data, te)
        plt.ylim(ymin=min_t_e, ymax=max_t_e)
        plt.yscale('log')
        plt.grid(which='both', axis='both')
        plt.title(f"T_e Timestep {imagenumber + 1}")
        plt.tight_layout()
        plt.savefig(f"t_e_{imagenumber + 1}.png")
        TE_IMAGE_PATHS.append(f"t_e_{imagenumber + 1}.png")
        plt.cla()
        plt.clf()
    for timestep in range(len(TR_IMAGE_PATHS)):
        TR_IMAGES.append(imageio.imread(TR_IMAGE_PATHS[timestep]))
    TR_IMAGES.clear()
    imageio.mimsave("t_r.gif", TR_IMAGES, fps=1)
    for timestep in range(len(TE_IMAGE_PATHS)):
        TE_IMAGES.append(imageio.imread(TE_IMAGE_PATHS[timestep]))
    imageio.mimsave("t_e.gif", TE_IMAGES, fps=1)
    TE_IMAGES.clear()

def main():
    """Main function"""
    args = get_args()
    plot(args)

if __name__ == '__main__':
    main()
