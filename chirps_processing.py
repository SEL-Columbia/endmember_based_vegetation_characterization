import numpy as np
import rasterio
import glob
import os
import geopandas as gpd
from rasterio.mask import mask
import shapely
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import datetime
from dateutil.rrule import rrule, MONTHLY
import pandas as pd




def return_shp_file(tile_name):

    if tile_name == 'T10_centralvalley':

        all_tiles_shp = gpd.read_file('/Users/terenceconlon/Documents/Columbia - Spring 2020/satellite_imagery/'
                                      'california_data/shape_files/CA_climate_zones_epsg_4326.shp')
        tile_shp = all_tiles_shp[all_tiles_shp['BZone'] == '13']
        tile_shp_list = [tile_shp['geometry'].iloc[i] for i in range(len(tile_shp))]


    elif tile_name == 'T37_amhara':
        epsg_num = 4326
        all_tiles_shp = gpd.read_file('/Volumes/sel_external/ethiopia_shapefiles/Ethiopia Admin Shape Files v2/'
                                      'eth_admbnda_adm1_csa_20160121.shp')
        tile_shp = all_tiles_shp[all_tiles_shp['admin1Name'] == 'Amhara'].to_crs(({'init': 'epsg:{}'.format(epsg_num)}))
        tile_shp_list = [tile_shp['geometry'].iloc[i] for i in range(len(tile_shp))]


    elif tile_name == 'uganda':
        tile_shape = gpd.read_file('/Volumes/sel_external/uganda_shapefiles/UGA_outline_SHP/UGA_outline_epsg4326.shp')
        tile_shp_list = [tile_shape['geometry'].iloc[i] for i in range(len(tile_shape))]


    return tile_shp_list

def load_clip_and_save_data(tile_name, tile_shp_list):

    strt_dt = datetime.date(2009, 1, 15)
    end_dt = datetime.date(2019, 12, 15)

    chirps_dir = '/Volumes/sel_external/irrigation_detection/chirps/africa_monthly'
    all_files = glob.glob(chirps_dir + '/*.tif')


    date_tuple_list = [str(dt.year) +'.' +  str(dt.month).zfill(2) for dt in
                       rrule(MONTHLY, dtstart=strt_dt, until=end_dt)]

    month_tuple_list = ['01', '02', '03', '04', '05', '06', '07', '08', '09','10', '11', '12']

    all_files = sorted([i for i in all_files if i.split('v2.0.')[-1].replace('.tif', '') in date_tuple_list])
    rainfall_ts = np.zeros((int(len(all_files)/12), 12))

    for ix,file in enumerate(all_files):
        print(ix)
        year_ix = int(np.floor_divide(ix, 12))
        month_ix = int(np.remainder(ix, 12))

        src = rasterio.open(file)
        img_clipped, img_meta = mask(src, tile_shp_list, crop=True)

        rainfall_ts[year_ix, month_ix] = np.mean(img_clipped)

    out_file = os.path.join('/Volumes/sel_external/irrigation_detection/chirps/monthly_region_averages',
                            'monthly_average_chirps_{}_{}_{}.csv'.format(
        tile_name, date_tuple_list[0], date_tuple_list[-1]))

    rainfall_ts = np.expand_dims(np.mean(rainfall_ts, axis= 0),0)
    print(rainfall_ts)

    out_df = pd.DataFrame(rainfall_ts, columns= month_tuple_list)
    out_df.to_csv(out_file)

    fig,ax = plt.subplots()
    ax.plot(range(12), rainfall_ts[0])

    plt.show()

    return rainfall_ts





if __name__ == '__main__':
    tile_name = 'uganda'

    tile_shp_list = return_shp_file(tile_name)

    load_clip_and_save_data(tile_name, tile_shp_list)




