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
import rasterio
from pysptools.abundance_maps.amaps import UCLS, NNLS, FCLS
from chirps_processing import return_nclusters, normalize
from sklearn.metrics import mean_squared_error
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from gurobipy import *
from rasterio.mask import mask
import geopandas as gpd
from rasterio.merge import merge
import time


def pca_transform(args, evi_img):
    # pca_transform takes an array of pixel timeseries and reduces the dimensionality


    n_components = args.pca_components
    evi_img_flattened = np.transpose(np.reshape(evi_img, (evi_img.shape[0], evi_img.shape[1] * evi_img.shape[2])))

    # Only take non-zero  timeseries
    max_pixels = np.min((args.num_samples, len(evi_img_flattened)))
    evi_img_flattened = evi_img_flattened[np.mean(evi_img_flattened, axis = 1)!= 0]

    if args.shuffle:
        np.random.seed(args.random_seed)
        np.random.shuffle(evi_img_flattened)

    # Limit to max_pixel num
    evi_img_flattened = evi_img_flattened[0:max_pixels]


    # Initialize a PCA model and fit to the data
    pca = PCA(n_components= n_components)

    principalComponents = pca.fit_transform(evi_img_flattened)


    return principalComponents, pca, evi_img_flattened

def clustering(args, principalComponents):

    # Cluster the pixels in PC space with a basic k-means clustering method
    num_samples = len(principalComponents)
    num_clusters = np.min((args.num_clusters, num_samples))

    kmeans_cluster = KMeans(n_clusters=num_clusters, random_state=args.random_seed).fit(principalComponents)
    cluster_predicts = KMeans(n_clusters=num_clusters, random_state=args.random_seed).fit_predict(principalComponents)
    cluster_centers = kmeans_cluster.cluster_centers_

    # Return cluster centers and cluster predictions for all the pixels timeseries
    return cluster_centers, cluster_predicts

def return_convex_hull(cluster_centers):

    # Return a convex hull surrounding the clustered pixels in PC space
    hull = ConvexHull(cluster_centers)
    exterior_clusters = cluster_centers[hull.vertices]

    return hull, exterior_clusters

def calculate_cluster_timeseries(cluster_predicts, evi_img_flattened):

    # Find the timeseries for the cluster centers and apply a sav_gol filter to smooth
    cluster_timeseries = np.zeros(evi_img_flattened.shape)
    unique = np.unique(cluster_predicts)

    for i in range(len(unique)):
        cluster_timeseries[i] = savgol_filter(np.mean(evi_img_flattened[np.where(cluster_predicts == i)],
                                                      axis = 0), 7,3)

    return cluster_timeseries


def interpolate_rainfall(args, monthly_rainfall_ts):

    # Create a domain variable for loading in the CHIRPS data which is stored
    # under slightly different directory structure
    if args.unmixing_region == 'fresno':
        domain = 'california'
    elif args.unmixing_region == 'amhara':
        domain = 'ethiopia'
    elif args.unmixing_region == 'uganda':
        domain = 'uganda'
    elif args.unmixing_region == 'catalonia':
        domain = 'catalonia'
    elif args.unmixing_region == 'ethiopia':
        domain = 'ethiopia'

    # Calculate number of years of data for analysis. This is hard coded in some places -- may want to go through and
    # change in the future
    num_years = (int(args.end_month_year.split('-')[-1])- int(args.start_month_year.split('-')[-1])) + 1


    # Need to fix this for Ethiopia
    monthly_rainfall_ts = np.tile(monthly_rainfall_ts, (1, num_years))
    strt_dt = datetime.date(int(args.start_month_year.split('-')[-1]), int(args.start_month_year.split('-')[0]), 15)
    end_dt  = datetime.date(int(args.end_month_year.split('-')[-1]), int(args.end_month_year.split('-')[0]), 15)

    # Find dates for the rainfall predictions
    start_doy = strt_dt.timetuple().tm_yday
    rainfall_ordinal_dates = [(dt.date() - strt_dt).days + start_doy for dt in rrule(MONTHLY, dtstart=strt_dt,
                                                                                     until=end_dt)]
    rainfall_ordinal_dates = np.insert(rainfall_ordinal_dates, 0, rainfall_ordinal_dates[0] - 31)

    # Return the dates for the MODIS images
    modis_imgs_folder = os.path.join(args.base_dir, 'imagery', 'modis', 'individual_evi_images', domain)
    modis_dates = [i.split('_')[-1].replace('.tif', '').replace('doy', '') for i in
                   sorted(glob.glob(modis_imgs_folder + '/*.tif'))] # starts September 14
    modis_dates = [(int(i[0:4]), int(i[4:7])) for i in modis_dates]
    modis_ordinal_dates = [(i[0] - int(args.start_month_year.split('-')[-1]))*365 + i[1] for i in modis_dates]

    # Interpolate rainfall ts to MODIS dates
    rainfall_ts = np.zeros((len(monthly_rainfall_ts), len(modis_ordinal_dates)))
    for i in range(len(monthly_rainfall_ts)):
        monthly_rainfall_ts_padded  = np.insert(monthly_rainfall_ts[i], 0, monthly_rainfall_ts[i][-1])
        cs = CubicSpline(rainfall_ordinal_dates, monthly_rainfall_ts_padded)
        rainfall_ts[i] = cs(modis_ordinal_dates)

    return rainfall_ts


def find_endmembers(args, cluster_timeseries, rainfall_ts, region_index):

    xrange = len(rainfall_ts)

    # Normalize rainfall
    rainfall_ts = normalize(rainfall_ts)

    # 1 year for MODIS
    single_year_rainfall_ts = rainfall_ts[0:23]

    # Determine number of peaks of the rainfall timeseries
    peaks, peaks_dict = find_peaks(single_year_rainfall_ts, prominence=0.4)
    num_peaks_per_year = len(peaks)

    # Only deal with regions that have 1 or 2 peaks for now
    assert num_peaks_per_year == 1 or num_peaks_per_year == 2

    # Set up variables for comparing the rainfall timeseries to the cluster timeseries
    inphase_error = np.zeros(cluster_timeseries.shape[0])
    rainfall_ts_1mo_lag = np.concatenate((rainfall_ts[-2::], rainfall_ts[0:-2]))

    # Calculate error for regions with 1 peak in the dominant rainfall pattern
    # Still TBD the the best rainfall shifting approach for determining the out of phase endmember
    if num_peaks_per_year == 1:
        print('Single rainfall peak per year')

        rainfall_ts_4mo_lag = np.concatenate((rainfall_ts[-8::], rainfall_ts[0:-8]))
        rainfall_ts_7mo_lag = np.concatenate((rainfall_ts[-12::], rainfall_ts[0:-12]))
        rainfall_ts_10mo_lag = np.concatenate((rainfall_ts[-20::], rainfall_ts[0:-20]))


        outphase_error = np.zeros((cluster_timeseries.shape[0], 2))

        for i in range(cluster_timeseries.shape[0]):
            inphase_error[i]        = mean_squared_error(rainfall_ts_1mo_lag,
                                                         normalize(cluster_timeseries[i]))
            outphase_error[i, 0]    = mean_squared_error(rainfall_ts_4mo_lag,
                                                         normalize(cluster_timeseries[i]))
            outphase_error[i, 1]    = mean_squared_error(rainfall_ts_7mo_lag,
                                                         normalize(cluster_timeseries[i]))
            # outphase_error[i, 2]    = mean_squared_error(normalize(rainfall_ts_10mo_lag),
            #                                              normalize(cluster_timeseries[i]))


    # Similar process, except for regions with two peaks in the dominant rainfall pattern
    elif num_peaks_per_year == 2:
        print('Double rainfall peaks per year')


        # We want to shift only a single peak to find the out pf phase endmember, as it is unlikely a region will
        # have 4 discernible vegetation peaks (2 in phase + 2 out of phase). It is more likely that there will be a
        # single out of phase vegetation signal to extract.

        single_rainfall_peak_ts = np.zeros(23)
        peak_base_left, peak_base_right = peaks_dict['left_bases'][0], peaks_dict['left_bases'][1]
        single_rainfall_peak_ts[peak_base_left:peak_base_right] = \
            single_year_rainfall_ts[peak_base_left:peak_base_right]

        # Set up this single peak to be shifted forward and in reverse.
        forward_shift = int((peaks[1] - peaks[0])/2)
        reverse_shift = int(forward_shift - 11.5)


        single_season_rainfall_ts_forward = np.tile(np.concatenate((single_rainfall_peak_ts[-forward_shift::],
                                                                    single_rainfall_peak_ts[0:-forward_shift])),3)

        single_season_rainfall_ts_reverse = np.tile(np.concatenate((single_rainfall_peak_ts[-reverse_shift::],
                                                                    single_rainfall_peak_ts[0:-reverse_shift])), 3)


        single_season_rain_forward_1mo_lag = np.concatenate((single_season_rainfall_ts_forward[-2::],
                                                             single_season_rainfall_ts_forward[0:-2]))

        single_season_rain_reverse_1mo_lag = np.concatenate((single_season_rainfall_ts_reverse[-2::],
                                                             single_season_rainfall_ts_reverse[0:-2]))

        # Calculate error to between shifted rainfall and cluster timeseries to extract endmembers
        outphase_error = np.zeros((cluster_timeseries.shape[0], 2))

        for i in range(cluster_timeseries.shape[0]):
            inphase_error[i]     = mean_squared_error(rainfall_ts_1mo_lag, normalize(cluster_timeseries[i]))
            outphase_error[i, 0] = mean_squared_error(single_season_rain_forward_1mo_lag,
                                                      normalize(cluster_timeseries[i]))
            outphase_error[i, 1] = mean_squared_error(single_season_rain_reverse_1mo_lag,
                                                      normalize(cluster_timeseries[i]))


    # Select endmembers based on minimum error
    inphase_index    = np.where(inphase_error  == np.min(inphase_error))
    inphase_endmember  = cluster_timeseries[inphase_index][0]

    outphase_index = np.where(outphase_error == np.min(outphase_error))[0]
    outphase_endmember = cluster_timeseries[outphase_index][0]

    # Save endmembers to a csv and return
    endmember_array = np.zeros((xrange, 3))
    endmember_array[:, 0] = inphase_endmember
    endmember_array[:, 1] = outphase_endmember

    endmember_df = pd.DataFrame(endmember_array, columns= ['inphase_region{}'.format(region_index),
                                                            'outphase_region{}'.format(region_index),
                                                            'dark_region{}'.format(region_index)] )

    return endmember_df




def flattened_image_unmixing(amap, flattened_img, endmember_array):

    # Calculate abundance map via pysptools spectral unmixing
    max_number_of_rows = 1e5
    number_of_iterations = int(np.ceil(len(flattened_img)/max_number_of_rows))
    abundance_map_recreated = np.zeros((len(flattened_img), endmember_array.shape[0]+1))

    # Calculate abundances in batches to avoid memory overflow
    for iter in range(number_of_iterations):

        print('Calculating abundance for batch {} (of {} total batches)'.format(iter, number_of_iterations))

        top_index    = int(iter * max_number_of_rows)
        bottom_index = int(np.min(((iter+1)*max_number_of_rows, len(flattened_img))))

        # Always normalize imagery
        array_slice = normalize(flattened_img[top_index:bottom_index])
        abundance_map_slice = amap(array_slice, endmember_array)
        mse_array = calculate_error(array_slice, endmember_array, abundance_map_slice)

        # Calculate error and append to abundance map
        abundance_map_slice = np.append(abundance_map_slice, np.expand_dims(mse_array, -1), axis=1)
        abundance_map_recreated[top_index:bottom_index, :] = abundance_map_slice


    return abundance_map_recreated

def calculate_error(image_stack, endmember_array, abundance_map):

    # Mean square error calculation
    image_recreated = np.matmul(abundance_map, endmember_array)
    mse_array = ((image_stack - image_recreated)**2).sum(axis = -1)

    return mse_array


def spectral_unmixing_main(args, img_src, endmember_array, unmixing_method):

    # Find the number of regional clusters for the area of interest
    n_regional_clusters = return_nclusters(args)
    img_meta = img_src.meta

    # Read in polygons file
    polygons_file = os.path.join(args.base_dir, 'saved_rainfall_regions', 'clean_regions',
                                 '{}_rainfall_regions_nclusters_{}_clean.shp'.format(
                                     args.unmixing_region, n_regional_clusters))
    region_polygons = gpd.read_file(polygons_file).to_crs(img_src.meta['crs'])


    # Reorder maps and endmembers
    endmember_array = np.transpose(np.array(endmember_array))




    print('Cropping image and spectral unmixing, starting timer')
    t = time.time()

    for region in range(n_regional_clusters):

        # Crop image to the regional clusters
        cropped_img, cropped_transform = mask(img_src, [region_polygons['geometry'].iloc[region]], crop=True)
        evi_img = np.moveaxis(cropped_img, 0, -1)

        abundance_map = np.zeros((evi_img.shape[0], evi_img.shape[1], 4))*np.nan

        nonzero_indices = np.mean(evi_img, axis = -1) != 0
        evi_img_nonzero = evi_img[nonzero_indices]

        # Set up an unmixing modeling instance
        if unmixing_method == 'ucls':
            amap = UCLS
        elif unmixing_method == 'fcls':
            amap = FCLS
        elif unmixing_method == 'nnls':
            amap = NNLS


        # Select and normalize endmembers
        regional_endmembers = endmember_array[region*3: (region+1)*3]
        for i in range(2):
            regional_endmembers[i] = normalize(regional_endmembers[i])


        print('Unmixing for region {}'.format(region))
        abundance_map[nonzero_indices] = flattened_image_unmixing(amap, evi_img_nonzero, regional_endmembers)


        abundance_map = np.moveaxis(abundance_map, -1, 0).astype(np.float32)

        out_file_path = os.path.join(args.base_dir, 'abundance_maps', args.unmixing_region, 'regional_maps',
                                 '{}_abundancemap_modis_250m_{}_unmixingmethod_automatic_tEMs_'
                                 'outphasetype_{}_region_{}.tif'.format(args.unmixing_region, args.unmixing_method,
                                                              args.outphase_endmember_type, region))

        if os.path.exists(out_file_path):
            os.remove(out_file_path)


        # Update metadata
        img_meta['count'] = 4
        img_meta['dtype'] = 'float32'
        img_meta['nodata'] = 'nan'
        img_meta['transform'] = cropped_transform
        img_meta['height'] = abundance_map.shape[1]
        img_meta['width'] = abundance_map.shape[2]

        # Write out regional abundance map
        with rasterio.open(out_file_path, 'w+', **img_meta) as dest:
            dest.write(abundance_map)

    elapsed = (time.time() - t)

    print('Elapsed time for abundance map creation: {}s'.format(elapsed))
    # Merge the abundance maps into a mosaic
    merge_regional_abundance_maps(args)


def merge_regional_abundance_maps(args):
    regional_map_dir = os.path.join(args.base_dir, 'abundance_maps',
                                                    args.unmixing_region, 'regional_maps')
    out_file_path = os.path.join(args.base_dir, 'abundance_maps', args.unmixing_region,
                                 '{}_abundancemap_modis_250m_{}_unmixingmethod_automatic_tEMs_'
                                 'outphasetype_{}_allregionsmerged.tif'.format(args.unmixing_region, args.unmixing_method,
                                                                        args.outphase_endmember_type))

    # Collect all the regional abundance maps for the area of interest
    regional_tifs = glob.glob(regional_map_dir + '/*{}*.tif'.format(args.unmixing_method))
    img_list = []

    print('Regional tiffs include: {}'.format(regional_tifs))

    for img in regional_tifs:
        src_img = rasterio.open(img, 'r')
        img_list.append(src_img)
        img_meta = src_img.meta.copy()

    # Mosaic the image
    mosaic, out_trans = merge(img_list)

    img_meta['transform'] = out_trans
    img_meta['height'] = mosaic.shape[1]
    img_meta['width'] = mosaic.shape[2]

    with rasterio.open(out_file_path, 'w', **img_meta) as dest:
        dest.write(mosaic)


def return_endmembers(args, src):
    # This is the top-level function for calling all the helper functions to return the endmembers

    save_file_endmembers = os.path.join(args.base_dir, 'saved_endmembers', args.unmixing_region,
                            'extracted_endmembers_{}_outphasetype_{}_nclusters_{}_nsamples_{}.csv'.format(
                             args.unmixing_region, args.outphase_endmember_type,
                             args.num_clusters, args.num_samples))

    if args.calculate_new_endmembers:

        # Calculate new endmembers
        print('Calculating new endmembers')

        n_regional_clusters = return_nclusters(args)
        rainfall_ts_file = os.path.join(args.base_dir, 'saved_rainfall_regions', 'cluster_center_rainfall_ts_csvs',
                                        '{}_rainfall_regions_nclusters_{}_normalized_monthly_ts.csv'.format(
                                            args.unmixing_region, n_regional_clusters))
        monthly_rainfall_ts = np.array(pd.read_csv(rainfall_ts_file, index_col = 0))

        print('Interpolate rainfall timeseries')
        interpolated_rainfall_ts = interpolate_rainfall(args, monthly_rainfall_ts)

        print('Read regional polygons')
        polygons_file = os.path.join(args.base_dir, 'saved_rainfall_regions', 'clean_regions',
                                     '{}_rainfall_regions_nclusters_{}_clean.shp'.format(
                                         args.unmixing_region, n_regional_clusters))
        region_polygons = gpd.read_file(polygons_file).to_crs(src.meta['crs'])

        all_endmembers_df = pd.DataFrame(index= range(len(interpolated_rainfall_ts[0])))

        for region_index in range(n_regional_clusters):
            print('Calculating endmembers for {}, Region {}'.format(args.unmixing_region, region_index))

            masked_evi_img, img_transform = mask(src, [region_polygons['geometry'].iloc[region_index]], nodata=0)

            print('PCA Transform')
            principalComponents, pca, evi_img_flattened = pca_transform(args, masked_evi_img)

            print('Clustering')
            cluster_centers, cluster_predicts = clustering(args, principalComponents)

            print('Finding Cluster Timeseries')
            cluster_timeseries = calculate_cluster_timeseries(cluster_predicts, evi_img_flattened)


            print('Extract Endmembers')
            endmember_df = find_endmembers(args, cluster_timeseries, interpolated_rainfall_ts[region_index],
                                           region_index)
            all_endmembers_df = pd.concat([all_endmembers_df, endmember_df], axis=1)


        # Save extracted endmembers
        all_endmembers_df.to_csv(save_file_endmembers)


    endmember_array = np.array(pd.read_csv(save_file_endmembers, index_col=0, header=0))

    # Return endmember array
    return endmember_array





