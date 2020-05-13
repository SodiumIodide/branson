#!/usr/bin/env python3

import h5py
import os
import sys
import imageio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
        tr = np.array(file['t_r'])
        te = np.array(file['t_e'])
        tr_df = pd.DataFrame(data=tr[:][:][tr.shape[2]//2])
        te_df = pd.DataFrame(data=te[:][:][te.shape[2]//2])
        max_t_r = max([tr_df.max().max(), max_t_r])
        min_t_r = min([tr_df.min().min(), min_t_r])
        max_t_e = max([te_df.max().max(), max_t_e])
        min_t_e = min([te_df.min().min(), min_t_e])
    sort_nicely(h5files)
    for imagenumber, h5file in enumerate(h5files):
        file = h5py.File(h5file, 'r')
        x_boundaries = file['x']
        y_boundaries = file['y']
        z_boundaries = file['z']
        x_data = []
        y_data = []
        z_data = []
        for i in range(len(x_boundaries) - 1):
            x_data.append((x_boundaries[i] + x_boundaries[i + 1]) / 2.0)
        for i in range(len(y_boundaries) - 1):
            y_data.append((y_boundaries[i] + y_boundaries[i + 1]) / 2.0)
        for i in range(len(z_boundaries) - 1):
            z_data.append((z_boundaries[i] + z_boundaries[i + 1]) / 2.0)
        tr = np.array(file['t_r'])
        te = np.array(file['t_e'])
        tr_df = pd.DataFrame(data=tr[:][:][len(z_data)//2], columns=x_data, index=y_data)
        te_df = pd.DataFrame(data=te[:][:][len(z_data)//2], columns=x_data, index=y_data)
        plt.figure()
        sns.heatmap(tr_df, cmap="coolwarm", vmin=min_t_r, vmax=max_t_r)
        plt.title(f"Timestep {imagenumber + 1}")
        plt.tight_layout()
        plt.savefig(f"t_r_{imagenumber + 1}.png")
        TR_IMAGE_PATHS.append(f"t_r_{imagenumber + 1}.png")
        plt.cla()
        plt.clf()
        sns.heatmap(te_df, cmap="coolwarm", vmin=min_t_e, vmax=max_t_e)
        plt.title(f"Timestep {imagenumber + 1}")
        plt.tight_layout()
        plt.savefig(f"t_e_{imagenumber + 1}.png")
        TE_IMAGE_PATHS.append(f"t_e_{imagenumber + 1}.png")
        plt.cla()
        plt.clf()
    for timestep in range(len(TR_IMAGE_PATHS)):
        TR_IMAGES.append(imageio.imread(TR_IMAGE_PATHS[timestep]))
        TE_IMAGES.append(imageio.imread(TE_IMAGE_PATHS[timestep]))
    imageio.mimsave("t_r.gif", TR_IMAGES, fps=1)
    imageio.mimsave("t_e.gif", TE_IMAGES, fps=1)

def main():
    """Main function"""
    args = get_args()
    plot(args)

if __name__ == '__main__':
    main()
