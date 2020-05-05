import numpy as np
import rasterio
from rasterio.mask import mask
import os
import geopandas as gpd
from sklearn.model_selection import train_test_split
import glob
import pandas as pd
from scipy.signal import find_peaks
import random
import matplotlib.pyplot as plt


def load_amhara_polygons(args):
    irrig_polys_path = os.path.join(args.base_dir, 'shapefiles_and_templates', 'irrigation_presence_polygons',
                                    'ethiopia_irrigated_polygons.shp')
    nonirrig_polys_path = os.path.join(args.base_dir, 'shapefiles_and_templates', 'irrigation_presence_polygons',
                                    'ethiopia_nonirrigated_polygons.shp')

    irrig_polys = gpd.read_file(irrig_polys_path).to_crs('EPSG:{}'.format(args.amhara_epsg))
    nonirrig_polys = gpd.read_file(nonirrig_polys_path).to_crs('EPSG:{}'.format(args.amhara_epsg))

    irrig_poly_list = [irrig_polys['geometry'].iloc[i] for i in range(len(irrig_polys))]
    nonirrig_poly_list = [nonirrig_polys['geometry'].iloc[i] for i in range(len(nonirrig_polys))]

    return irrig_poly_list, nonirrig_poly_list

def load_catalonia_polygons(args):
    full_polys_path = os.path.join(args.base_dir, 'shapefiles_and_templates', 'catalonia',
                                   'sigpac_catalonia_3_ha_min.shp')

    all_polys = gpd.read_file(full_polys_path)

    irrig_shps = all_polys[(all_polys['sr'] == 100) & (all_polys['us'] == 'TA') & ((all_polys['id_com'] == '9') |
                                                                                   (all_polys['id_com'] == '22') |
                                                                                   (all_polys['id_com'] == '33') |
                                                                                   (all_polys['id_com'] == '23') ) ]
    nonirrig_shps = all_polys[(all_polys['sr'] == 0) & (all_polys['us'] == 'TA')]

    irrig_poly_list = [irrig_shps['geometry'].iloc[i] for i in range(len(irrig_shps))]
    nonirrig_poly_list = [nonirrig_shps['geometry'].iloc[i] for i in range(len(nonirrig_shps))]

    return irrig_poly_list, nonirrig_poly_list

def load_fresno_polygons(args):
    crop_labels = ['G', 'R', 'F', 'P', 'T', 'D', 'C', 'V', 'I', 'NC', 'NV', 'NR', 'NB']

    irrig_polys_path = os.path.join(args.base_dir, 'shapefiles_and_templates', 'california_cropland_map',
                                    'T11SKA_cropland_intersection.shp')

    nonirrig_polys_path = os.path.join(args.base_dir, 'shapefiles_and_templates', 'california_cropland_map',
                                       'fresno_cropland_intersection.shp')

    irrig_crop_shp = gpd.read_file(irrig_polys_path).to_crs('EPSG:{}'.format(args.fresno_epsg))
    nonirrig_crop_shp = gpd.read_file(nonirrig_polys_path).to_crs('EPSG:{}'.format(args.fresno_epsg))

    single_crop_shp_file_list = [irrig_crop_shp['geometry'].iloc[i] for i in range(len(irrig_crop_shp)) if
                                 (irrig_crop_shp['MULTIUSE'].iloc[i] == 'S' and
                                  nonirrig_crop_shp['IRR_TYP2PA'].iloc[i] != 'n' and
                                  irrig_crop_shp['CLASS2'].iloc[i] in crop_labels)]  # irrig

    double_crop_shp_file_list = [irrig_crop_shp['geometry'].iloc[i] for i in range(len(irrig_crop_shp)) if
                                 irrig_crop_shp['MULTIUSE'].iloc[i] == 'D']  # irrig

    nonirrig_poly_list = [nonirrig_crop_shp['geometry'].iloc[i] for i in range(len(nonirrig_crop_shp)) if
                          (nonirrig_crop_shp['IRR_TYP2PA'].iloc[i] == 'n' and
                           nonirrig_crop_shp['CLASS2'].iloc[i] in crop_labels)]

    irrig_poly_list = single_crop_shp_file_list + double_crop_shp_file_list

    return irrig_poly_list, nonirrig_poly_list


def format_data_for_training(args, irrig_pixels, nonirrig_pixels):


    irrig_ts_flat    = np.reshape(irrig_pixels, (irrig_pixels.shape[0] * irrig_pixels.shape[1], irrig_pixels.shape[2]))
    nonirrig_ts_flat = np.reshape(nonirrig_pixels, (nonirrig_pixels.shape[0] * nonirrig_pixels.shape[1],
                                                    nonirrig_pixels.shape[2]))

    irrig_ts_flat    = irrig_ts_flat[~np.isnan(irrig_ts_flat).any(axis = 1)]
    nonirrig_ts_flat = nonirrig_ts_flat[~np.isnan(nonirrig_ts_flat).any(axis = 1)]

    # Shuffle and clip
    np.random.seed(args.random_seed)
    np.random.shuffle(irrig_ts_flat)
    np.random.shuffle(nonirrig_ts_flat)

    min_num_pixels = np.min((len(irrig_ts_flat), len(nonirrig_ts_flat)))

    irrig_ts_flat    = irrig_ts_flat[0:min_num_pixels]
    nonirrig_ts_flat = nonirrig_ts_flat[0:min_num_pixels]


    if args.prediction_method == 'baseline' and args.baseline_prediction_shift and training_bool == False:

        train_rainfall_file = glob.glob(os.path.join(args.base_dir, 'chirps', 'monthly_region_averages',
                                               '*{}*.csv'.format(args.train_region)))[0]
        train_rainfall_ts = np.array(pd.read_csv(train_rainfall_file, index_col=0))[0]

        test_rainfall_file = glob.glob(os.path.join(args.base_dir, 'chirps' , 'monthly_region_averages',
                                                     '*{}*.csv'.format(args.test_region)))[0]
        test_rainfall_ts = np.array(pd.read_csv(test_rainfall_file, index_col=0))[0]

        train_peak_indices = find_peaks(train_rainfall_ts, height=0.5 * np.max(train_rainfall_ts))[0]
        test_peak_indices  = find_peaks(test_rainfall_ts, height=0.5 * np.max(test_rainfall_ts))[0]

        shift = int(np.mean(test_peak_indices - train_peak_indices))

        print('Shifting full EVI stack by {} months'.format(shift))
        irrig_ts_flat = np.concatenate((irrig_ts_flat[2*shift::], irrig_ts_flat[0:2*shift]), axis=1)
        nonirrig_ts_flat = np.concatenate((nonirrig_ts_flat[2*shift::], nonirrig_ts_flat[0:2*shift]), axis=1)



    irrig_labels = np.ones(len(irrig_ts_flat))
    nonirrig_labels = np.zeros(len(nonirrig_ts_flat))

    print('Irrig Samples: {}'.format(len(irrig_ts_flat)))
    print('Non-Irrig Samples: {}'.format(len(nonirrig_ts_flat)))

    X_train, X_val, y_train, y_val = train_test_split(np.concatenate((irrig_ts_flat, nonirrig_ts_flat)),
                                                        np.concatenate((irrig_labels, nonirrig_labels)),
                                                        train_size = args.train_size, random_state = args.random_seed)

    return X_train, X_val, y_train, y_val


def return_polygon_pixels(args, region, irrig_poly_list, nonirrig_poly_list):



    if args.prediction_method == 'endmember':
        map_file = os.path.join(args.base_dir, 'abundance_maps', region,
                                 '{}_abundancemap_modis_250m_{}_unmixingmethod_automatic_tEMs_'
                                 'outphasetype_{}.tif'.format(region, args.unmixing_method,
                                                              args.outphase_endmember_type))


    elif args.prediction_method == 'baseline':
        map_file = os.path.join(args.base_dir, 'imagery', 'modis',
                                'evi_{}_16day_2016257_1019241_250m.tif'.format(region))


    with rasterio.open(map_file, 'r') as src:

        irrig_pixels, _    = mask(src, irrig_poly_list, nodata=np.nan)
        nonirrig_pixels, _ = mask(src, nonirrig_poly_list, nodata=np.nan)

        irrig_pixels       = np.moveaxis(irrig_pixels, 0, -1)
        nonirrig_pixels    = np.moveaxis(nonirrig_pixels, 0, -1)



    X_train, X_val, y_train, y_val = format_data_for_training(args, irrig_pixels, nonirrig_pixels)

    return X_train, X_val, y_train, y_val


class DataGenerator():
    'This selects and prepares test, training and validation data'
    def __init__(self,args):
        self.args = args


    def return_data(self):

        amhara_irrig_poly_list, amhara_nonirrig_poly_list = load_amhara_polygons(self.args)
        catalonia_irrig_poly_list, catalonia_nonirrig_poly_list = load_catalonia_polygons(self.args)
        fresno_irrig_poly_list, fresno_nonirrig_poly_list = load_fresno_polygons(self.args)


        X_train_fresno, X_val_fresno, y_train_fresno, y_val_fresno = return_polygon_pixels(
            self.args, 'fresno', fresno_irrig_poly_list, fresno_nonirrig_poly_list)

        X_train_catalonia, X_val_catalonia, y_train_catalonia, y_val_catalonia = return_polygon_pixels(
            self.args, 'catalonia', catalonia_irrig_poly_list, catalonia_nonirrig_poly_list)


        X_train_amhara, X_val_amhara, y_train_amhara, y_val_amhara = return_polygon_pixels(
            self.args, 'amhara', amhara_irrig_poly_list, amhara_nonirrig_poly_list)


        if self.args.train_region == 'fresno':
            X_train = X_train_fresno
            y_train = y_train_fresno

        elif self.args.train_region == 'amhara':
            X_train = X_train_amhara
            y_train = y_train_amhara

        elif self.args.train_region == 'catalonia':
            X_train = X_train_catalonia
            y_train = y_train_catalonia



        elif self.args.train_region == 'both':
            if self.args.equal_training_fracs:
                min_training_num = np.min((len(X_train_fresno), len(X_train_amhara)))

                X_train_amhara = X_train_amhara[0:min_training_num]
                y_train_amhara = y_train_amhara[0:min_training_num]

                X_train_fresno = X_train_fresno[0:min_training_num]
                y_train_fresno = y_train_fresno[0:min_training_num]

            X_train = np.concatenate((X_train_amhara, X_train_fresno))
            y_train = np.concatenate((y_train_amhara, y_train_fresno))
            # np.random.shuffle(X_train)
            # np.random.shuffle(y_train)


        if self.args.test_region == 'fresno':
            X_val = X_val_fresno
            y_val = y_val_fresno

        elif self.args.test_region == 'amhara':
            X_val = X_val_amhara
            y_val = y_val_amhara

        elif self.args.test_region == 'catalonia':
            X_val = X_val_catalonia
            y_val = y_val_catalonia




        return X_train, X_val, y_train, y_val,



