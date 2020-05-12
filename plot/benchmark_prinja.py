#!/usr/bin/env python3

import h5py
import os
import matplotlib.pyplot as plt

from plot_1d import get_args, sort_nicely

TR_IMAGE_PATHS = []
TE_IMAGE_PATHS = []
TR_IMAGES = []
TE_IMAGES = []

TIME_DELTA = 0.03335641

PROB_0 = 0.01
PROB_1 = 0.99

SOL = 299.792458

def plot_benchmark(args):
    directory = args.directory
    h5files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory,f)) and f.endswith(".h5")]
    sort_nicely(h5files)
    time_data = []
    tr_data = []
    te_data = []
    for imagenumber, h5file in enumerate(h5files):
        file = h5py.File(h5file, 'r')
        time_data.append(imagenumber * TIME_DELTA * SOL)
        m0_tr = file['m0_t_r']
        m0_te = file['m0_t_e']
        m1_tr = file['m1_t_r']
        m1_te = file['m1_t_e']
        tr_data.append(m0_tr[-1] * PROB_0 + m1_tr[-1] * PROB_1)
        te_data.append(m0_te[-1] * PROB_0 + m1_te[-1] * PROB_1)
    plt.plot(time_data, tr_data, label="Radiation Temperature")
    plt.plot(time_data, te_data, label="Material Temperature")
    plt.grid(which='both', axis='both')
    plt.legend(loc='best')
    plt.xscale('log')
    plt.tight_layout()
    plt.savefig(f"benchmark_prinja.png")
    plt.cla()
    plt.clf()

def main():
    """Main function"""
    args = get_args()
    plot_benchmark(args)

if __name__ == '__main__':
    main()
