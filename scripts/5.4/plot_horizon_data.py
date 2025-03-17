# plot horizon data for the Kerr eccentric equatorial case
# python plot_horizon_data.py -Tobs 2.0 -q 1.0e-5 -spin 0.99 -zaxis e0 -base 27_11_2024

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.lines as mlines
from seaborn import color_palette

from scipy.interpolate import interp1d
import argparse

import h5py

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Computer Modern"]

import pysco 
pysco.plot.default_plotting()


parser = argparse.ArgumentParser(description="Plot horizon data")
parser.add_argument("-Tobs", "--Tobs", type=float, default=1.0, help="Observation time in years")
parser.add_argument("-q", "--q", type=float, default=1.0e-5, help="Mass ratio")
parser.add_argument("-e0", "--e0", type=float, default=5.0e-1, help="initial eccentricity")
parser.add_argument("-spin", "--spin", type=float, default=9.9e-5, help="spin")
parser.add_argument("-zaxis", "--zaxis", type=str, default='e0', help="z-axis")
parser.add_argument("-base", "--base_name", type=str, default='horizon', help="base name of the data file")
parser.add_argument("-cpal", "--cpal", type=str, default='colorblind', help="color palette")
parser.add_argument("-ec", "--every_color", type=int, default=1, help="how many colors to skip")

args = vars(parser.parse_args())

# seaborn colorblind palette
cpal = color_palette(args['cpal'])[::args['every_color']]#, as_cmap=True)
#cpal = color_palette("dark:#5A9_r")[::2]

datadir = './horizon_data/'
plotdir = './figures/'

def add_plot(M_detector, data, ls, colors='k', fill=False, interp=None, interp_kwargs={}, plot_kwargs={}, fig=None, axs=None):
    if fig is None or axs is None:
        fig, axs = plt.subplots(ncols=1, nrows=1, sharex=True)

    if isinstance(colors, str):
        colors = [colors] * len(data.keys())

    Mgrid = np.logspace(5, 8, 1000)
    zmin = 0.0

    for key, color in zip(data.keys(), colors):
        
        z_here = data[key][:]
        M_source = to_M_source(M_detector, z_here)
        if interp is not None:
            x = Mgrid
            interpolant = interp(M_source, z_here, **interp_kwargs) 
            y = interpolant(x)

        else:
            x = M_source
            y = z_here

        if fill:
            #axs.semilogx(x, y, ls=ls, color='k', label=key, **plot_kwargs)
            axs.semilogx(x, y, ls=ls, color=color, label=key, **plot_kwargs)
            axs.fill_between(x, zmin, y, alpha=0.3, zorder=1, hatch='', color=color, rasterized=True)
            zmin = y
        else:
            axs.semilogx(x, y, ls=ls, color=color, label=key, **plot_kwargs)
        
    
    plt.xlabel(r'$M_{\rm source} \, [M_\odot]$')
    plt.ylabel(r'$\bar{z}$')

    return fig, axs


def to_logM_source(logM, z):
    return logM - np.log(1+z) 

def to_M_source(M, z):
    return M / (1+z)

def pastel_map(cmap, c=0.2, n=6):
    """
    Create a lighter version of a colormap
    Arguments:
    cmap : colormap
    c : scale factor for the lighter colors
    n : number of colors to return
    """
    if type(cmap) == str:
        cmap = plt.get_cmap(cmap)
    colors = ((1. - c) * cmap(np.linspace(0., 1., n)) + c * np.ones((n, 4)))
    return colors
    

if __name__ == '__main__':
    Tobs = args['Tobs']
    q = args['q']
    e0 = args['e0']
    spin = args['spin']
    zaxis = args['zaxis']
    base_name = args['base_name']

    if zaxis == 'e0':
        datastring = 'T_{:.1f}_q_{:.1e}_spin_{:.1e}.h5'.format(Tobs, q, spin)

    elif zaxis == 'spin':
        datastring = 'T_{:.1f}_q_{:.1e}_e0_{:.1e}.h5'.format(Tobs, q, e0)


    else:
        raise ValueError("z-axis must be either 'e0' or 'spin'")
    
    savename = plotdir + base_name + '_' + zaxis + '_' + datastring[:-3] + '.pdf'
    
    kerr_kerr_data = base_name + '_traj_kerr_wf_kerr_' + datastring
    kerr_aak_data = base_name + '_traj_kerr_wf_aak_' + datastring
    pn5_aak_data = base_name + '_traj_pn5_wf_aak_' + datastring

    linestyles = ['-', '--', '-.']
    #labels = ['Kerr 0PA', 'Kerr trajectory, AAK amplitudes', 'PN5 trajectory, AAK amplitudes']
    labels = ['Kerr-Kerr', 'Kerr-AAK', 'PN5-AAK']
    handles = []

    interp = interp1d
    interp_kwargs = dict(fill_value='extrapolate')
    zaxis_plot = r'$e_0$' if zaxis == 'e0' else r'$a$'
    fill = False
    zmax = 0.0

    for i, datafile in enumerate([kerr_kerr_data, kerr_aak_data, pn5_aak_data]):
        ls = linestyles[i]
        label = labels[i]
        data = {}
        attr = {}
        try:
            with h5py.File(datadir + datafile, 'r') as f:
                for key in f.attrs.keys():
                    attr[key] = f.attrs[key]
                for key in f.keys():
                    data[key] = f[key][:]

            zmax = max(zmax, max([data[key].max() for key in data.keys()]) + 0.2)
        except:
            print("File not found: ", datafile)
            continue
            
        #sort the data by increasing zaxis keyword
        keys = list(data.keys())
        sorted_keys = np.sort(keys)
        data = {key:data[key] for key in sorted_keys}

        #cpal = pastel_map('Blues_r', c=0.2, n=len(data.keys())+1)[::-1]

        M_detector = attr['M'] # detector-frame mass
        plot_kwargs = dict(zorder=10-i, rasterized=True)
        if i == 0: # create the figure
            fig, ax = add_plot(M_detector, data, ls, colors=cpal, fill=True, interp=interp, interp_kwargs=interp_kwargs, plot_kwargs=plot_kwargs)
            boundaries = list(attr[zaxis])
            nlines = len(boundaries)

        else:
            add_plot(M_detector, data, ls, colors=cpal, fill=False, interp=interp, interp_kwargs=interp_kwargs, plot_kwargs=plot_kwargs, fig=fig, axs=ax)

        handle = mlines.Line2D([], [], color='gray', linestyle=ls, label=label)
        handles += [handle]

    # colorbar
    boundaries = boundaries + [boundaries[-1] + 0.15]
    boundaries_shift = [boundaries[i] - 0.05 for i in range(len(boundaries))]

    norm = mpl.colors.BoundaryNorm(boundaries_shift, nlines)

    cmap = mpl.colors.ListedColormap(cpal[:nlines])
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)

    sm.set_array([])
    cbar_ax = fig.add_axes([1.0, 0.09, 0.03, 0.8])
    cbar = fig.colorbar(sm, ax=ax, pad=0.05, orientation='vertical', cax=cbar_ax, extend='both', drawedges=True)
    cbar.set_ticks(boundaries[:-1])
    cbar.set_label(zaxis_plot, fontsize=18, labelpad=22)
    cbar.ax.yaxis.label.set_rotation(0)
    ax.legend(handles=handles, ncols=3, bbox_to_anchor=(1.038, 1.1))
    fig.tight_layout()
    ax.xaxis.set_tick_params(pad=6)
    ax.set_xlim(5e5, 2e7)
    ax.set_ylim(0., zmax)
    plt.savefig(savename, dpi=300)
    #breakpoint()


    