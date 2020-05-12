#!/usr/bin/env python3

import h5py
import argparse
import os
import sys
import re
import imageio
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

TR_IMAGE_PATHS = []
TE_IMAGE_PATHS = []
TR_IMAGES = []
TE_IMAGES = []

class FullPaths(argparse.Action):
    """Expand user- and relative-paths"""
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, os.path.abspath(os.path.expanduser(values)))

def is_dir(dirname):
    """Checks if a path is an actual directory"""
    if not os.path.isdir(dirname):
        msg = "{0} is not a directory".format(dirname)
        raise argparse.ArgumentTypeError(msg)
    else:
        return dirname

def get_args():
    """Get CLI arguments and options"""
    parser = argparse.ArgumentParser(description="""Python HDF5 plotter for Branson""")
    parser.add_argument('directory', help="The folder of files to process", action=FullPaths, type=is_dir)
    return parser.parse_args()

def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)

def plot1d(args):
    directory = args.directory
    h5files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and f.endswith(".h5")]
    # Find maximum (one loop)
    max_t_r = sys.float_info.min
    max_t_e = sys.float_info.min
    min_t_r = sys.float_info.max
    min_t_e = sys.float_info.max
    for h5file in h5files:
        file = h5py.File(h5file, 'r')
        max_t_r = max(file['m0_t_r']) if max(file['m0_t_r']) > max_t_r else max_t_r
        max_t_r = max(file['m1_t_r']) if max(file['m1_t_r']) > max_t_r else max_t_r
        max_t_e = max(file['m0_t_e']) if max(file['m0_t_e']) > max_t_e else max_t_e
        max_t_e = max(file['m1_t_e']) if max(file['m1_t_e']) > max_t_e else max_t_e
        min_t_r = min(file['m0_t_r']) if min(file['m0_t_r']) < min_t_r else min_t_r
        min_t_r = min(file['m1_t_r']) if min(file['m1_t_r']) < min_t_r else min_t_r
        min_t_e = min(file['m0_t_e']) if min(file['m0_t_e']) < min_t_e else min_t_e
        min_t_e = min(file['m1_t_e']) if min(file['m1_t_e']) < min_t_e else min_t_e
    sort_nicely(h5files)
    for imagenumber, h5file in enumerate(h5files):
        file = h5py.File(h5file, 'r')
        x_boundaries = file['x']
        x_data = []
        for i in range(len(x_boundaries) - 1):
            x_data.append((x_boundaries[i] + x_boundaries[i + 1]) / 2.0)
        m0_tr = file['m0_t_r']
        m1_tr = file['m1_t_r']
        m0_te = file['m0_t_e']
        m1_te = file['m1_t_e']
        plt.plot(x_data, m0_tr, color='b', label="Material 0 T_r")
        plt.plot(x_data, m1_tr, color='r', label="Material 1 T_r")
        plt.ylim(ymin=min_t_r, ymax=max_t_r)
        plt.yscale('log')
        plt.grid(which='both', axis='both')
        plt.legend(loc='best')
        plt.title(f"Timestep {imagenumber + 1}")
        plt.tight_layout()
        plt.savefig(f"t_r_{imagenumber + 1}.png")
        TR_IMAGE_PATHS.append(f"t_r_{imagenumber + 1}.png")
        plt.cla()
        plt.clf()
        plt.plot(x_data, m0_te, color='b', label="Material 0 T_e")
        plt.plot(x_data, m1_te, color='r', label="Material 1 T_e")
        plt.ylim(ymin=min_t_e, ymax=max_t_e)
        plt.yscale('log')
        plt.grid(which='both', axis='both')
        plt.legend(loc='best')
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
    '''Main function'''
    args = get_args()
    plot1d(args)

if __name__ == '__main__':
    main()
