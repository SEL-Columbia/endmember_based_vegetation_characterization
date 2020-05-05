import numpy as np
import matplotlib.pyplot as plt
from dataloader import load_amhara_polygons, load_fresno_polygons
import os, yaml, argparse
import rasterio
from rasterio.mask import mask
import glob
import seaborn as sns
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               FixedLocator, IndexLocator, LinearLocator)
from scipy.signal import savgol_filter
from chirps_processing import return_nclusters
from spectral_unmixing import interpolate_rainfall
import pandas as pd



colors_xkcd = ['very dark purple', "windows blue", "amber", "greyish",
               "faded green", 'pastel blue']
cmap = sns.color_palette(sns.xkcd_palette(colors_xkcd), desat=1)
sns.set_palette(sns.xkcd_palette(colors_xkcd))
# sns.set_style("whitegrid")
# sns.set_style("white")


def normalize(M):
    """
    Normalizes M to be in range [0, 1].

    Parameters:
      M: `numpy array`
          1D, 2D or 3D data.

    Returns: `numpy array`
          Normalized data.
    """
    minVal = np.min(M)
    maxVal = np.max(M)

    Mn = M - minVal

    if maxVal == minVal:
        return np.zeros(M.shape)
    else:
        return Mn / (maxVal-minVal)

def get_args():
    parser = argparse.ArgumentParser(
        description= 'Predict irrigation presence from abundance maps')

    parser.add_argument('--training_params_filename',
                        type=str,
                        default='params.yaml',
                        help='Filename defining model configuration')

    args = parser.parse_args()
    config = yaml.load(open(args.training_params_filename))
    for k, v in config.items():
        args.__dict__[k] = v


    return args

def plot_endmembers(args, endmember_array):

    n_regional_clusters = return_nclusters(args)


    rainfall_ts_file = os.path.join(args.base_dir, 'saved_rainfall_regions', 'cluster_center_rainfall_ts_csvs',
                                    '{}_rainfall_regions_nclusters_{}_normalized_monthly_ts.csv'.format(
                                        args.unmixing_region, n_regional_clusters))
    monthly_rainfall_ts = np.array(pd.read_csv(rainfall_ts_file, index_col=0))

    rainfall_ts = interpolate_rainfall(args, monthly_rainfall_ts)


    xrange = range(len(rainfall_ts[0]))

    ticknames = ['01/2017', '01/2018', '01/2019']
    minors = np.linspace(0, 69, 37)

    fig, ax = plt.subplots(1, n_regional_clusters)

    for ax_ix in range(n_regional_clusters):
        twin_ax = ax[ax_ix].twinx()

        # ax.plot(xrange[0:23], rainfall_ts[0:23], label = 'Monthly rainfall', linestyle = '-')
        ax[ax_ix].plot(xrange, normalize(endmember_array[:, ax_ix*3]), label = 'In phase', color = 'r')
        ax[ax_ix].plot(xrange, normalize(endmember_array[:, ax_ix*3 + 1]), label = 'Out of phase', color = 'g')
        ax[ax_ix].plot(xrange, endmember_array[:, ax_ix*3 + 2], label = 'Dark', color = 'b')
        ax[ax_ix].set_title(args.unmixing_region + ', region {}'.format(ax_ix))
        ax[ax_ix].set_xlabel('Month')
        ax[ax_ix].grid('on')
        ax[ax_ix].legend(loc='upper left')

        ax[ax_ix].set_xticklabels(ticknames)
        ax[ax_ix].xaxis.set_major_locator(IndexLocator(23, 0))

        ax[ax_ix].xaxis.set_minor_locator(FixedLocator(minors))
        ax[ax_ix].tick_params(axis='x', which='both', length=2)

        twin_ax.plot(xrange, rainfall_ts[ax_ix], linestyle = ':', color = cmap[0])


    # ax.set_ylabel('Average Monthly Rainfall (2009-2019)')
    # ax1.set_ylabel('Normalized EVI')

    # ax1.legend(loc = 'upper right')

    # ticknames = ['09/2016', '09/2017', '09/2018']

    # monthnames = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']


    # ax.xaxis.set_major_locator(IndexLocator(12, 0))


    # ax.set_xticklabels(monthnames)
    # ax.set_xticks(np.linspace(0, 23, 12))



    # print(minors)
    #

    #
    # plt.savefig('/Users/terenceconlon/Documents/Columbia - Spring 2020/presentations/figures_for_presentations'
    #             '/endmembers_only_plotting_unmixing_region_{}.png'.format(unmixing_region))

    plt.show()

def plot_pixel_means(args, region):
    # amhara_irrig_poly_list, amhara_nonirrig_poly_list = load_amhara_polygons(args)
    irrig_poly_list, nonirrig_poly_list = load_fresno_polygons(args)

    map_file = glob.glob(os.path.join(args.base_dir, 'imagery', 'modis',
                            '*{}*.tif'.format(region)))[0]

    nanvalue = 32767

    with rasterio.open(map_file, 'r') as src:

        irrig_pixels, _    = mask(src, irrig_poly_list, nodata=nanvalue)
        nonirrig_pixels, _ = mask(src, nonirrig_poly_list, nodata=nanvalue)

        irrig_pixels       = np.moveaxis(irrig_pixels, 0, -1)
        nonirrig_pixels    = np.moveaxis(nonirrig_pixels, 0, -1)

    irrig_ts_flat = np.reshape(irrig_pixels, (irrig_pixels.shape[0] * irrig_pixels.shape[1], irrig_pixels.shape[2]))
    nonirrig_ts_flat = np.reshape(nonirrig_pixels, (nonirrig_pixels.shape[0] * nonirrig_pixels.shape[1],
                                                    nonirrig_pixels.shape[2]))

    print(irrig_ts_flat.shape)


    irrig_ts_flat    = irrig_ts_flat[~np.any(irrig_ts_flat==nanvalue, axis = 1)] #.any(axis=1)
    nonirrig_ts_flat = nonirrig_ts_flat[~np.any(nonirrig_ts_flat== nanvalue, axis=1)] #.any(axis=1)]

    print(irrig_ts_flat.shape)


    irrig_array_mean    = np.mean(irrig_ts_flat, axis=0)
    nonirrig_array_mean = np.mean(nonirrig_ts_flat, axis=0)

    fig, ax = plt.subplots()

    ax.plot(range(len(irrig_array_mean)), irrig_array_mean, label = 'Irrigated')
    ax.plot(range(len(nonirrig_array_mean)), nonirrig_array_mean, label = 'Non Irrigated')
    ax.grid()
    ax.legend()

    plt.show()

if __name__ == '__main__':

    args =  get_args()
    # plot_pixel_means(args, 'fresno')
    # plot_endmembers()

