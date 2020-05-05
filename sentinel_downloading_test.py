from collections import OrderedDict
from sentinelsat import SentinelAPI

import numpy as np
import pandas as pd
import os
import copy
import fiona
import geopandas as gpd
import glob
import zipfile
# from osgeo import gdal
from datetime import date
from multiprocessing.dummy import Pool as ThreadPool
import subprocess
from subprocess import PIPE
import pickle
import rasterio
from rasterio.mask import mask, raster_geometry_mask
from rasterio.windows import Window
from rasterio import features
# import geojson
from shapely.geometry import shape,mapping, Point, Polygon, MultiPolygon, box
# from utils import splitImageIntoCells, imageStacking, writeImageOut
from pyproj import Proj, transform
from rasterio import Affine
from rasterio.warp import reproject, Resampling
from collections import OrderedDict
# from retrying import retry
# from utils import create_tif_template

import datetime
from dateutil.rrule import rrule, MONTHLY

fiona.drvsupport.supported_drivers['gml'] = 'rw'
fiona.drvsupport.supported_drivers['GML'] = 'rw'


api = SentinelAPI('tconlon', 'Hx0sC26qZGgY')

def find_all_images(tile):

    query_kwargs = {
            'platformname': 'Sentinel-2',
            'producttype': 'S2MSI1C',
            'date': ('20190101', '20200101')}

    products = OrderedDict()

    kw = query_kwargs.copy()
    kw['filename'] = '*_T{}_*'.format(tile)  # products after 2016-12-01
    pp = api.query(**kw)
    products.update(pp)

    products_df = api.to_dataframe(products)

    products_df['size']  = products_df['size'].apply(lambda x: float(x.split(' ')[0]))
    products_df['year']  = products_df['title'].apply(lambda x: int(x.split('_')[2][0:4]))
    products_df['month'] = products_df['title'].apply(lambda x: int(x.split('_')[2][4:6]))
    products_df['day']   = products_df['title'].apply(lambda x: int(x.split('_')[2][6:8]))

    products_df = products_df[products_df['size'] > 750]

    print(len(products_df))

    products_df.to_csv('test_copernicus.csv')

    return products_df

def return_single_monthly_image(products_df, year, month):

    imgs_df = products_df[(products_df['year'] == year) & (products_df['month'] == month)]
    imgs_df = imgs_df.sort_values(by = ['cloudcoverpercentage', 'size'])

    if len(imgs_df) > 0:
        return imgs_df, True

    else:
        return '', False




def create_dirs(save_dir_base, tile_name, year, month, valid_bands):

    folder_level_1 = os.path.join(save_dir_base, tile_name)
    folder_level_2 = os.path.join(folder_level_1, year)
    folder_level_3 = os.path.join(folder_level_2, month)
    folder_level_cloud = os.path.join(folder_level_3, 'cloud_cover')

    for band in valid_bands:
        folder_level_band = os.path.join(folder_level_3, band)

    if not os.path.isdir(folder_level_1):
        os.mkdir(folder_level_1)
    if not os.path.isdir(folder_level_2):
        os.mkdir(folder_level_2)
    if not os.path.isdir(folder_level_3):
        os.mkdir(folder_level_3)
    if not os.path.exists(folder_level_cloud):
        os.mkdir(folder_level_cloud)
    for band in valid_bands:
        folder_level_band = os.path.join(folder_level_3, band)
        if not os.path.isdir(folder_level_band):
            os.mkdir(folder_level_band)



def download_and_create_evi_image(id):

    product_dict = api.download(id, directory_path='/Volumes/sel_external/sentinel_copernicus')



def reproject_and_save_tile(gs_tuple, pixel_size = 10):

    tile_folder_base = '/Volumes/sel_external/sentinel_imagery/reprojected_tiles'
    tile_folder = os.path.join(tile_folder_base, tile_name)

    if not os.path.exists(tile_folder):
        os.mkdir(tile_folder)

    print('Downloading Tile: {}'.format(gs_tuple[-1]))

    valid_bands = ['B02', 'B03', 'B04', 'B08']

    cmd_string = '/Users/terenceconlon/google-cloud-sdk/bin/gsutil ls -r ' + gs_tuple[-1] +'/GRANULE'
    image_list_output = subprocess.Popen(cmd_string, shell= True, stdout=subprocess.PIPE)
    image_list_clean = [j.decode('utf-8') for j in image_list_output.stdout.readlines()]

    image_list_output.kill()

    jp2_list      = sorted([j.replace('\n', '') for j in image_list_clean if '.jp2' in j])
    jp2_band_list = sorted([i for i in jp2_list if i.split('_')[-1][0:3] in valid_bands])
    jp2_single_image = [j.replace('\n', '') for j in image_list_clean if valid_bands[0]+'.jp2' in j][0]
    cloud_cover_gml = [j.replace('\n', '') for j in image_list_clean if 'CLOUDS' in j][0]

    year  = jp2_single_image.split('_')[-2][0:4]
    month = jp2_single_image.split('_')[-2][4:6]
    day   = jp2_single_image.split('_')[-2][6:8]

    dir_folder, cloud_folder = create_dirs(tile_folder_base, tile_name, year, month, valid_bands)


    cloud_path = os.path.join(cloud_folder,
                              'cloud_cover_polygons_{}_{}_{}.shp'.format(year, month, day))

    tile_template = '/Volumes/sel_external/sentinel_imagery/tile_template_tifs/' \
                    'pixel_{}m/template_{}.tif'.format(pixel_size, tile_name)

    with rasterio.open(tile_template, 'r') as band_dest:
        metadata = band_dest.meta.copy()


        for file in jp2_band_list:
            band = file.split('_')[-1][0:3]
            band_folder = os.path.join(dir_folder, band)
            previously_saved_images = glob.glob(band_folder + '/*.tif')

            save_file_str = '{}_{}_{}_{}.tif'.format(year, month, day, band)
            save_file = os.path.join(band_folder, save_file_str)
            save_new_file = True

            print(save_file)
            if os.path.exists(save_file):
                # with rasterio.open(save_file, 'r', driver='GTiff') as saved_img:
                #     print(np.max(saved_img.read()))
                #     if np.max(saved_img.read()) > 0:
                save_new_file = False

            print(save_new_file)
            if save_new_file:

                with rasterio.open(file, 'r', driver='JP2OpenJPEG',) as band_src:
                    print(band_src.meta)
                    print(np.mean(band_src.read()))

                    print('before reproject: {}'.format(file))

                    with rasterio.open(save_file, 'w', **metadata) as dest_jp2:
                        reproject(source=rasterio.band(band_src,1),
                                    destination=rasterio.band(dest_jp2,1),

                                  resampling=Resampling.nearest)

                    print('after reproject')

    cloud_file = os.path.join(tile_folder, cloud_path)

    if not os.path.exists(cloud_file):
        with fiona.open(cloud_cover_gml, 'r') as src_cc:
            with fiona.open(cloud_file, 'w', crs=src_cc.crs, driver='ESRI Shapefile',
                                schema=src_cc.schema) as output:
                for f in src_cc:
                    output.write(f)

if __name__ == '__main__':
    tile = '37PCM'

    months = range(1,2)
    years  = [2019]

    products_df = find_all_images(tile)
    selected_images_df = pd.DataFrame()

    for year in years:
        for month in months:
            img_df, valid_img = return_single_monthly_image(products_df, year, month)

            if valid_img:
                product_id =  img_df.index[0]
                print(product_id)
                download_and_create_evi_image(product_id)


                selected_images_df = selected_images_df.append(img_df)

            else:
                print('No image available for year: {}, month: {}'.format(year, month))









# api.download_all(products)