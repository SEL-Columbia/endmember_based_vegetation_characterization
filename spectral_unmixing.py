import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.spatial import ConvexHull
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.signal import find_peaks
import glob
from scipy.interpolate import CubicSpline
import datetime
from dateutil.rrule import rrule, MONTHLY
import os
from main import get_args
import rasterio
from pysptools.abundance_maps import UCLS, FCLS, NNLS
from sklearn.metrics import mean_squared_error
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt



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

def pca_transform(args, evi_img):

    n_components = args.pca_components
    evi_img_flattened = np.transpose(np.reshape(evi_img, (evi_img.shape[0], evi_img.shape[1] * evi_img.shape[2])))

    if args.shuffle:
        np.random.seed(args.random_seed)
        np.random.shuffle(evi_img_flattened)

    pca = PCA(n_components= n_components)
    principalComponents = pca.fit_transform(evi_img_flattened)

    max_pixels = np.min((args.num_samples, len(evi_img_flattened)))

    principalComponents = principalComponents[0:max_pixels]
    evi_img_flattened = evi_img_flattened[0:max_pixels]

    return principalComponents, pca, evi_img_flattened

def clustering(args, principalComponents):

    num_samples = len(principalComponents)
    num_clusters = np.min((args.num_clusters, num_samples))

    kmeans_cluster = KMeans(n_clusters=num_clusters, random_state=args.random_seed).fit(principalComponents)
    cluster_predicts = KMeans(n_clusters=num_clusters, random_state=args.random_seed).fit_predict(principalComponents)
    cluster_centers = kmeans_cluster.cluster_centers_

    return cluster_centers, cluster_predicts

def return_convex_hull(cluster_centers):

    hull = ConvexHull(cluster_centers)
    exterior_clusters = cluster_centers[hull.vertices]

    return hull, exterior_clusters

def calculate_cluster_timeseries(cluster_predicts, evi_img_flattened):

    cluster_timeseries = np.zeros(evi_img_flattened.shape)
    unique = np.unique(cluster_predicts)

    for i in range(len(unique)):
        cluster_timeseries[i] = savgol_filter(np.mean(evi_img_flattened[np.where(cluster_predicts == i)], axis = 0),
                                              7,3)

    return cluster_timeseries


def interpolate_rainfall(args): # , img_stack, rainfall_norm):

    if args.unmixing_region == 'fresno':
        domain = 'california'
    elif args.unmixing_region == 'amhara':
        domain = 'ethiopia'
    elif args.unmixing_region == 'uganda':
        domain = 'uganda'


    num_years = (int(args.end_month_year.split('-')[-1])- int(args.start_month_year.split('-')[-1])) + 1
    rainfall_file = glob.glob(os.path.join(args.base_dir, 'chirps/monthly_region_averages',
                                           '*{}*.csv'.format(args.unmixing_region)))[0]

    rainfall_ts = np.tile(np.array(pd.read_csv(rainfall_file, index_col=0))[0], 3)

    strt_dt = datetime.date(int(args.start_month_year.split('-')[-1]), int(args.start_month_year.split('-')[0]), 15)
    end_dt  = datetime.date(int(args.end_month_year.split('-')[-1]), int(args.end_month_year.split('-')[0]), 15)

    start_doy = strt_dt.timetuple().tm_yday
    rainfall_ordinal_dates = [(dt.date() - strt_dt).days + start_doy for dt in rrule(MONTHLY, dtstart=strt_dt,
                                                                                     until=end_dt)]

    modis_imgs_folder = os.path.join(args.base_dir, 'imagery', 'modis', 'individual_evi_images', domain)
    modis_dates = [i.split('_')[-1].replace('.tif', '').replace('doy', '') for i in
                   sorted(glob.glob(modis_imgs_folder + '/*.tif'))] # starts September 14

    modis_dates = [(int(i[0:4]), int(i[4:7])) for i in modis_dates]

    modis_ordinal_dates = [(i[0] - int(args.start_month_year.split('-')[-1]))*365 + i[1] for i in modis_dates]


    rainfall_ordinal_dates = np.insert(rainfall_ordinal_dates, 0, rainfall_ordinal_dates[0]-31)
    rainfall_ts            = np.insert(rainfall_ts, 0, rainfall_ts[-1])

    cs = CubicSpline(rainfall_ordinal_dates, rainfall_ts)
    rainfall_ts = cs(modis_ordinal_dates)

    return rainfall_ts


def determine_rainfall_periodicity(rainfall_ts):
    acf = np.correlate(rainfall_ts, rainfall_ts, 'full')[-len(rainfall_ts):]

    peaks = find_peaks(acf)[0]
    delay = peaks[acf[peaks].argmax()]

    # Divide by 12, as there are 12 samples in a half year
    num_peaks_per_year = 3 - round(delay/12)
    print(num_peaks_per_year)

    assert num_peaks_per_year == 1 or num_peaks_per_year == 2

    return num_peaks_per_year




def find_endmembers(args, cluster_timeseries, rainfall_ts_raw, save_file_endmembers):

    xrange = len(rainfall_ts_raw)
    # 1 year for MODIS
    rainfall_ts = np.zeros(23)


    # Average the rainfall timeseries for endmember extraction
    for i in range(len(rainfall_ts)):
        rainfall_ts[i] = np.mean((rainfall_ts_raw[i], rainfall_ts_raw[23 + i], rainfall_ts_raw[46+i]))

    rainfall_ts = np.tile(rainfall_ts, 3)



    # yhat = savgol_filter(rainfall_ts, 51, 3)  # window size 51, polynomial order 3

    ## Determine rainfall seasonality

    num_peaks_per_year   = determine_rainfall_periodicity(rainfall_ts)
    rainfall_ts_1mo_lag  = np.concatenate((rainfall_ts[-2::], rainfall_ts[0:-2]))
    rainfall_ts_4mo_lag  = np.concatenate((rainfall_ts[-8::], rainfall_ts[0:-8]))
    rainfall_ts_7mo_lag  = np.concatenate((rainfall_ts[-12::], rainfall_ts[0:-12]))
    rainfall_ts_10mo_lag = np.concatenate((rainfall_ts[-20::], rainfall_ts[0:-20]))

    inphase_error        = np.zeros(cluster_timeseries.shape[0])
    outphase_error       = np.zeros((cluster_timeseries.shape[0]))


    for i in range(cluster_timeseries.shape[0]):
        inphase_error[i]        = mean_squared_error(normalize(rainfall_ts_1mo_lag), normalize(cluster_timeseries[i]))
        outphase_error[i]       = mean_squared_error(normalize(rainfall_ts_4mo_lag), normalize(cluster_timeseries[i]))
        # outphase_error[i, 1]    = mean_squared_error(normalize(rainfall_ts_7mo_lag), normalize(cluster_timeseries[i]))
        # outphase_error[i, 2]    = mean_squared_error(normalize(rainfall_ts_10mo_lag), normalize(cluster_timeseries[i]))


    inphase_index    = np.where(inphase_error  == np.min(inphase_error))
    inphase_endmember  = cluster_timeseries[inphase_index][0]

    if num_peaks_per_year == 1:
        outphase_index = np.where(outphase_error == np.min(outphase_error))[0]
        outphase_endmember = cluster_timeseries[outphase_index][0]


    elif num_peaks_per_year == 2:
        outphase_index = np.where(outphase_error == np.min(outphase_error))
        outphase_endmember = cluster_timeseries[outphase_index][0]


    endmember_array = np.zeros((xrange, 3))
    endmember_array[:, 0] = inphase_endmember
    endmember_array[:, 1] = outphase_endmember

    pd.DataFrame(endmember_array, columns= ['inphase', 'outphase', 'dark']).\
           to_csv(save_file_endmembers)



def split_image_for_unmixing(amap, img, endmember_array, squareDim):

    # image is of shape height, width, depth (channel last)

    numberOfCellsHigh = img.shape[0] // squareDim + 1
    numberOfCellsWide = img.shape[1] // squareDim + 1

    print(img.shape)
    print(endmember_array.shape)


    abundance_map_recreated = np.zeros((img.shape[0], img.shape[1], endmember_array.shape[0]+1))

    x, y = 0, 0
    count = 0
    for hc in range(numberOfCellsHigh):

        y0 = hc * squareDim
        y1 = np.min(((hc + 1) * squareDim, img.shape[0]))

        for wc in range(numberOfCellsWide):
            print(count)

            x0 = wc * squareDim
            x1 = np.min(((wc + 1) * squareDim, img.shape[1]))

            img_slice = img[y0:y1, x0:x1, :]

            # Always normalize endmembers as correlation coefficient does not account for scale
            abundance_map_slice = amap.map(img_slice, endmember_array, normalize= True)
            amap.plot('')
            mse_array = calculate_error(img_slice, endmember_array, abundance_map_slice)
            abundance_map_slice = np.append(abundance_map_slice, np.expand_dims(mse_array, -1), axis=-1)

            abundance_map_recreated[y0:y1, x0:x1, :] = abundance_map_slice

            count +=1

    return abundance_map_recreated

def spectral_unmixing_main(args, evi_img, img_meta, endmember_array, unmixing_method):

    # Reorder maps and endmembers
    endmember_array = np.transpose(np.array(endmember_array))
    evi_img = np.moveaxis(evi_img, 0, -1)


    if unmixing_method == 'ucls':
        amap = UCLS()
    elif unmixing_method == 'fcls':
        amap = FCLS()
    elif unmixing_method == 'nnls':
        amap = NNLS()

    # Slice images to prevent memory overflow

    squareDim = 1400
    abundance_map = split_image_for_unmixing(amap, evi_img, endmember_array, squareDim)


    abundance_map = np.moveaxis(abundance_map, -1, 0).astype(np.float32)

    out_file_path = os.path.join(args.base_dir, 'abundance_maps', args.unmixing_region,
                                 '{}_abundancemap_modis_250m_{}_unmixingmethod_automatic_tEMs_'
                                 'outphasetype_{}.tif'.format(args.unmixing_region, args.unmixing_method,
                                                              args.outphase_endmember_type))

    if os.path.exists(out_file_path):
        os.remove(out_file_path)

    # abundance_map_for_saving = np.zeros((4, abundance_map.shape[1], abundance_map.shape[2])).astype(np.float32)
    # abundance_map_for_saving[0]   = abundance_map[0] # inphase
    # abundance_map_for_saving[1]   = np.sum(abundance_map[1:abundance_map.shape[0] - 2], axis=0) # out of phase
    # abundance_map_for_saving[2:4] = abundance_map[abundance_map.shape[0] - 2::] # dark and rmse

    # abundance_map_for_sav


    img_meta['count'] = 4
    img_meta['dtype'] = 'float32'
    img_meta['nodata'] = 'nan'

    with rasterio.open(out_file_path, 'w+', **img_meta) as dest:
        dest.write(abundance_map)



def calculate_error(image_stack, endmember_array, abundance_map):

    # Always normalize endmembers

    image_stack     = normalize(image_stack)
    endmember_array = normalize(endmember_array)

    image_recreated = np.matmul(abundance_map, endmember_array)
    mse_array = ((image_stack - image_recreated)**2).sum(axis = -1)

    return mse_array



def generate_endmembers(args, evi_img, save_file_endmembers):

    print('Find Rainfall Timeseries')
    rainfall_ts = interpolate_rainfall(args)

    print('PCA Transform')
    principalComponents, pca, evi_img_flattened = pca_transform(args, evi_img)

    print('Clustering')
    cluster_centers, cluster_predicts = clustering(args, principalComponents)

    print('Finding Cluster Timeseries')
    cluster_timeseries = calculate_cluster_timeseries(cluster_predicts, evi_img_flattened)

    print('Extract Endmembers')
    find_endmembers(args, cluster_timeseries, rainfall_ts, save_file_endmembers)








