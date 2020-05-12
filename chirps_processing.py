import numpy as np
import rasterio
import glob
import os
import geopandas as gpd
from rasterio.mask import mask
import shapely
from shapely.geometry import Polygon, MultiPolygon, mapping, shape
import matplotlib.pyplot as plt
import datetime
from dateutil.rrule import rrule, MONTHLY
import pandas as pd
from copy import copy
from rasterio.features import rasterize
import seaborn as sns
import fiona
from shapely.ops import unary_union
from sklearn.cluster import KMeans
from descartes import PolygonPatch
from rasterio.mask import mask


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


def return_nclusters(args):

    if args.unmixing_region == 'amhara':
        n_clusters = args.amhara_rainfall_nclusters
    elif args.unmixing_region == 'catalonia':
        n_clusters = args.catalonia_rainfall_nclusters
    elif args.unmixing_region == 'ethiopia':
        n_clusters = args.ethiopia_rainfall_nclusters
    elif args.unmixing_region == 'fresno':
        n_clusters = args.fresno_rainfall_nclusters
    elif args.unmixing_region == 'uganda':
        n_clusters = args.uganda_rainfall_nclusters

    return n_clusters

def rainfall_region_plotting(args):

    # Load timeseries data
    n_clusters = return_nclusters(args)
    cluster_ts_file = os.path.join(args.base_dir, 'saved_rainfall_regions', 'cluster_center_rainfall_ts_csvs',
                                    '{}_rainfall_regions_nclusters_{}_normalized_monthly_ts.csv'.format(
                                        args.unmixing_region, n_clusters))
    cluster_centers = np.array(pd.read_csv(cluster_ts_file, index_col=0))

    # Load cluster polygons
    polygon_file = os.path.join(args.base_dir,  'saved_rainfall_regions',  'clean_regions',
                           '{}_rainfall_regions_nclusters_{}_clean.shp'.format(args.unmixing_region,
                                                                         n_clusters))
    cluster_polygons = gpd.read_file(polygon_file)
    polygon_list = [(cluster_polygons['geometry'].iloc[i], cluster_polygons['pixelvalue'].iloc[i])
                     for i in range(len(cluster_polygons))]

    # Set plotting parameters
    colors_xkcd = ['very dark purple', "amber", "windows blue",
                   "faded green", "pumpkin orange", "dusty purple", "greyish", 'pastel blue', "darkish red"]

    cmap = sns.color_palette(sns.xkcd_palette(colors_xkcd), desat=1)

    # Plot time series and polygons next to each other
    fig, ax = plt.subplots(1,2)
    for i in range(len(cluster_centers)):
        ax[0].plot(range(12), cluster_centers[i], color=cmap[i+1], linewidth = 2, label = 'Region {}'.format(i+1))

    xticks = range(12)
    monthnames = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']
    ax[0].set_xticks(xticks)
    ax[0].set_xticklabels(monthnames)
    ax[0].set_xlabel('Month')
    ax[0].set_ylabel('Normalized Rainfall')
    ax[0].legend()

    for ix, (poly, val) in enumerate(polygon_list):
        patch = PolygonPatch(poly, fc=cmap[val], ec='k')
        ax[1].add_patch(patch)

    ax[1].axis('scaled')
    ax[1].grid()
    ax[1].set_xlabel('Longitude')
    ax[1].set_ylabel('Latitude')

    plt.show()

def cluster_rainfall(args):

    # Return the number of regional clusters for this area of interest
    n_clusters = return_nclusters(args)

    in_file = os.path.join(args.base_dir, 'chirps', 'clipped_regions_monthly',
              'chirps_{}_monthly_rainfall_averages_2009_2020_mm.tif'.format(args.unmixing_region))

    with rasterio.open(in_file, 'r') as src:
        img = src.read()

    # Reshape and normalize rainfall timeseries
    rainfall_ts = np.transpose(np.reshape(img, (img.shape[0], img.shape[1]*img.shape[2])))
    for i in range(len(rainfall_ts)):
        rainfall_ts[i] = normalize(rainfall_ts[i])

    # Only take non-zero valued timeseries
    zero_indices = np.mean(rainfall_ts, axis=1) != 0
    rainfall_ts = rainfall_ts[zero_indices]

    # Set up a clustering model and cluster the rainfall timeseries
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(rainfall_ts)
    cluster_centers = kmeans.cluster_centers_

    out_csv_filename = os.path.join(args.base_dir, 'saved_rainfall_regions', 'cluster_center_rainfall_ts_csvs',
                                    '{}_rainfall_regions_nclusters_{}_normalized_monthly_ts.csv'.format(
                                        args.unmixing_region, n_clusters))

    monthnames = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']
    pd.DataFrame(cluster_centers, columns= monthnames).to_csv(out_csv_filename)


    img_rainfall_preds = np.zeros((img.shape[1], img.shape[2])).astype(np.int16)

    # Assign each rainfall timeseries to a cluster
    for i in range(img.shape[1]):
        for j in range(img.shape[2]):
            if np.mean(img[:, i, j]) != 0:
                img_rainfall_preds[i, j] = kmeans.predict([img[:, i, j]]) + 1

    # Save the outputted polygons
    create_polygons(args, src, img_rainfall_preds, n_clusters)

def create_polygons(args, src, img_rainfall_preds, n_clusters):

    # Get the index for the clustered polygons
    unique_values = np.unique(img_rainfall_preds)
    nonzero_unique_values = unique_values[unique_values != 0]

    shapes = list(rasterio.features.shapes(img_rainfall_preds, transform=src.transform))

    shp_schema = {
        'geometry': 'MultiPolygon',
        'properties': {'pixelvalue': 'int'}
    }

    outfile = os.path.join(args.base_dir,  'saved_rainfall_regions',  'raw_regions',
                           '{}_rainfall_regions_nclusters_{}_raw.shp'.format(args.unmixing_region,
                                                                         n_clusters))

    # Save the area covered by the clusters as shapefiles
    with fiona.open(outfile, 'w', 'ESRI Shapefile', shp_schema, src.crs) as shp:
        for pixel_value in nonzero_unique_values:
            polygons = [shape(geom) for geom, value in shapes
                        if value == pixel_value]
            multipolygon = MultiPolygon(polygons)
            shp.write({
                'geometry': mapping(multipolygon),
                'properties': {'pixelvalue': int(pixel_value)}
            })

    # Clean the polygons list (remove polygons less than 10000 km2')
    cleaned_polygon_list = polygon_cleaning(args, outfile, n_clusters)

    return cleaned_polygon_list

def polygon_cleaning(args, infile, n_clusters):

    poly_file = gpd.read_file(infile)

    # Select valid polygons
    multipolys = poly_file[poly_file['pixelvalue'] != 0]

    small_poly_list = []
    large_poly_list = []

    for i in range(len(multipolys)):
        value = multipolys['pixelvalue'].iloc[i]
        polygons = multipolys['geometry'].iloc[i]

        for j in polygons:
            # Small polygons have an area less than 10000 km2
            if j.area <1:
                small_poly_list.append((j, value))
            else:
                large_poly_list.append((j, value))

    new_poly_list = []

    # Assign values in small polygons to the value of the nearest large polygon
    for geom, value in small_poly_list:
        distances_to_large_polys = [geom.centroid.distance(large_geom) for (large_geom, large_value)
                                    in large_poly_list]
        new_value = large_poly_list[int(np.argmin(distances_to_large_polys))][1]
        new_poly_list.append((geom, new_value))

    # Add the large polygons to the new polygon list
    new_poly_list.extend([poly for poly in large_poly_list])

    shp_schema = {
        'geometry': 'Polygon',
        'properties': {'pixelvalue': 'int'}
    }

    # Collect cluster polygons
    polygon_list = []
    for pixel_value in range(1, n_clusters+1):
        polygons = [poly[0] for poly in new_poly_list if poly[1] == pixel_value]
        union = (unary_union(polygons), pixel_value)
        polygon_list.append(union)

    outfile = os.path.join(args.base_dir,  'saved_rainfall_regions',  'clean_regions',
                           '{}_rainfall_regions_nclusters_{}_clean.shp'.format(args.unmixing_region,
                                                                         n_clusters))
    # Save cleaned polygons to out file
    with fiona.open(outfile, 'w', 'ESRI Shapefile', shp_schema, poly_file.crs) as shp:
        for union in polygon_list:
            shp.write({
                'geometry': mapping(union[0]),
                'properties': {'pixelvalue': int(union[1])}
            })


def load_clip_average_and_save_data(args):
    strt_dt = datetime.date(2009, 1, 15)
    end_dt = datetime.date(2019, 12, 15)

    # Load all saved CHIRPS files
    chirps_dir = os.path.join(args.base_dir, '/chirps/global_monthly/individual_images')
    all_files = glob.glob(chirps_dir + '/*.tif')

    # Select relevant shapefile
    chirps_region = 'catalonia'
    shapefile = glob.glob(os.path.join(args.base_dir, 'shapefiles_and_templates', chirps_region,
                                       '{}.shp'.format(chirps_region)))[0]
    shapefile_mask = gpd.read_file(shapefile)
    shp_list = [shapefile_mask['geometry'].iloc[i] for i in range(len(shapefile_mask))]

    date_tuple_list = [str(dt.year) + '.' + str(dt.month).zfill(2) for dt in
                       rrule(MONTHLY, dtstart=strt_dt, until=end_dt)]

    month_tuple_list = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']

    all_files = sorted([i for i in all_files if i.split('v2.0.')[-1].replace('.tif', '') in date_tuple_list])
    rainfall_ts = np.zeros((int(len(all_files) / 12), 12))

    print('Crop and save rainfall averages csv')
    for ix, file in enumerate(all_files):
        print(ix)
        year_ix = int(np.floor_divide(ix, 12))
        month_ix = int(np.remainder(ix, 12))

        # Mask the CHIRPS files to the shapefiles
        src = rasterio.open(file)
        img_clipped, img_meta = mask(src, shp_list, crop=True)

        rainfall_ts[year_ix, month_ix] = np.mean(img_clipped)

    # Save the average rainfall timeseries to an .csv
    out_file = os.path.join(args.base_dir, '/chirps/monthly_region_averages',
                            'monthly_average_chirps_{}_{}_{}.csv'.format(
                                chirps_region, date_tuple_list[0], date_tuple_list[-1]))

    rainfall_ts = np.expand_dims(np.mean(rainfall_ts, axis=0), 0)
    out_df = pd.DataFrame(rainfall_ts, columns=month_tuple_list)
    out_df.to_csv(out_file)

    print('Crop and save average monthly rainfall tif')
    # Will need to change the rows and cols on a case-by-case basis
    rows = 48
    cols = 64
    rainfall_averages = np.zeros((12, rows, cols)).astype(np.float32)
    for ix, month in enumerate(month_tuple_list):
        print(ix)

        month_files = [i for i in all_files if i.split('.')[-2] == month]
        rainfall_by_month = np.zeros((rows, cols, len(month_files)))

        for iy, file in enumerate(month_files):
            with rasterio.open(file) as src:
                rainfall_by_month[:, :, iy], trans = mask(src, shp_list, crop=True, nodata=0)
                chirps_meta = copy(src.meta)

        rainfall_averages[ix] = np.mean(rainfall_by_month, axis=-1)

    # Update meta
    chirps_meta.update({'count': '12', 'height': rows, 'width': cols, 'nodata': '0', 'transform': trans})
    outfile = os.path.join(chirps_dir, 'chirps_{}_monthly_rainfall_averages_2009_2020_mm.tif').format(chirps_region)

    # Write out the average monthly rainfall tif
    with rasterio.open(outfile, 'w', **chirps_meta) as dest:
        dest.write(rainfall_averages)





