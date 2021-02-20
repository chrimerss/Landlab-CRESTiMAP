'''
An interface to control CREST-iMAP
'''

# Define some variables
# ============basics=============#
BASIN_SHP= '/Users/allen/OneDrive - University of Oklahoma/CRESTHH/subbasins/08076700.shp'
DEM= '/Users/allen/OneDrive - University of Oklahoma/CRESTHH/reinfiltration/DSM.tif'
FROM_CRS= 'EPSG:4326'
TO_CRS = 'EPSG:32215'

# ============OBS================#
# include any observation points here
GAUGE1_LOC= (266648.7618,3316464.1819)
GAUGE1_NAME= '08075900'
GAUGE2_LOC= (277287.3178,3311964.4192)
GAUGE2_NAME= '08076000'
GAUGE3_LOC= (284362.476,3313541.542)
GAUGE3_NAME= '08076180'
GAUGE4_LOC= (274446.6892,3305767.1611)
GAUGE4_NAME= '08076500'
OUTLET_LOC= (284129.440,3302957.433)
OUTLET_NAME= 'outlet'
GAUGES= {GAUGE1_NAME:GAUGE1_LOC,
        GAUGE2_NAME: GAUGE2_LOC,
        GAUGE3_NAME: GAUGE3_LOC,
        GAUGE4_NAME: GAUGE4_LOC,
        OUTLET_NAME: OUTLET_LOC}
GAUGE_CRS= 'EPSG:32215'

# ============Parameters=========#
B='/Users/allen/OneDrive - University of Oklahoma/CRESTHH/data/CREST_para/b_10m.tif'
WM='/Users/allen/OneDrive - University of Oklahoma/CRESTHH/data/CREST_para/wm_10m.tif'
KSAT='/Users/allen/OneDrive - University of Oklahoma/CRESTHH/data/CREST_para/ksat.tif'
IM='/Users/allen/OneDrive - University of Oklahoma/CRESTHH/data/CREST_para/im.tif'
SM0='/Users/allen/OneDrive - University of Oklahoma/CRESTHH/data/sm.20170826.120000.tif'
FRICTION='/Users/allen/OneDrive - University of Oklahoma/CRESTHH/data/friction/manningn.tif'
KE= 1

# ============Forcing=============#
RAIN_PATH = '/Users/allen/OneDrive - University of Oklahoma/CRESTHH/data/mrms201708/'
# recognizable freq. D for day, H for hour, T for minute, S for second
RAIN_FREQ= '10T'
RAIN_PATTERN= 'PrecipRate_00.00_%Y%m%d-%H%M%S.grib2-var0-z0.tif'
EVAP_PATH = '/Users/allen/OneDrive - University of Oklahoma/CRESTHH/data/evap/'
EVAP_FREQ = '1D'
EVAP_PATTERN= 'cov_et%y%m%d.asc.tif'

# ============System==============#
# date to start e.g., 20170826110000
START= '20170826110000'
END= '20170830000000'
FREQ= '1H'
OUTPUT_TS=True
REINFILTRATION=True
EXCESS_RAIN=True
FILL_SINK= False
#repo to output asc file, set to None if no output
OUTPUT_DIR='results'
# vars to output, only support 'surface_water__depth'|'surface_water__discharge'|'SM'
OUTPUT_VARS= ['surface_water__depth', 'surface_water__discharge']
#choose to parallelize, 0: disable, >0: number of cores used to parallelize
PARALLEL=0
VERBOSE=True




'''
==================================Program========================================
'''
# Import dependencies
import xarray as xr
import rioxarray
import geopandas as gpd
from landlab.grid import RasterModelGrid
from landlab.components import OverlandFlow, SinkFiller
from landlab.components import CRESTHH, map_gauge_loc_to_node
import matplotlib.pyplot as plt
from landlab.io import read_geotif
from landlab.grid.mappers import map_mean_of_link_nodes_to_link
import numpy as np

def load_param(path, lons, lats):
    param= rioxarray.open_rasterio(path)
    param= param.rio.write_crs('EPSG:4326').rio.reproject('EPSG:32215')
    param= param.sel(x= lons, y=lats, method='nearest')
    return param.values

landing="""
======================================================================================
   _____   _____    ______    _____   _______            _   __  __              _____
  / ____| |  __ \\  |  ____|  / ____| |__   __|          (_) |  \\/  |     /\\     |  __ \\
 | |      | |__) | | |__    | (___      | |     ______   _  | \\  / |    /  \\    | |__) |
 | |      |  _  /  |  __|    \\___ \\     | |    |______| | | | |\\/| |   / /\\ \\   |  ___/
 | |____  | | \\ \\  | |____   ____) |    | |             | | | |  | |  / ____ \\  | |
  \\_____| |_|  \\_\\ |______| |_____/     |_|             |_| |_|  |_| /_/    \\_\\ |_|

=======================================================================================

Version: 1.1
        """

if __name__ == '__main__':
    print(landing)
    basin_shp= gpd.read_file(BASIN_SHP).set_crs(FROM_CRS).to_crs(TO_CRS)
    dem_region= rioxarray.open_rasterio(DEM).rio.write_crs(FROM_CRS).rio.reproject(TO_CRS)
    basin_dem= dem_region.rio.clip(basin_shp.geometry, basin_shp.crs).squeeze().sortby('y', ascending=True)
    del dem_region
    basin= RasterModelGrid(shape= basin_dem.squeeze().shape,
                       xy_spacing=np.diff(basin_dem.x.values)[0],
                       xy_of_lower_left= [basin_dem.x.values[0], basin_dem.y.values[0]]
                      )
    basin.axis_units = ('m', 'm')
    z= basin.add_field('topographic__elevation', basin_dem.values.astype(float),
                   at="node", clobber=True, dtype=float)
    if FILL_SINK:
        sf= SinkFiller(basin, routing='D4')
        sf.fill_pits()
    basin.status_at_node[basin.nodes_at_bottom_edge] = basin.BC_NODE_IS_FIXED_VALUE
    basin.status_at_node[np.isclose(z, basin_dem._FillValue)] = basin.BC_NODE_IS_CLOSED
    gauged_id = [map_gauge_loc_to_node(basin, GAUGES[name][0], GAUGES[name][1], GAUGE_CRS, TO_CRS) for name in GAUGES.keys()]
    gauges= {name:gauged_id[i] for i,name in enumerate(GAUGES.keys())}
    basin.set_watershed_boundary_condition_outlet_id(gauges['outlet'], 'topographic__elevation')


    print('-------------------------')
    print('Loading parameters ...')
    param= load_param(B,basin_dem.x.values,basin_dem.y.values)
    _= basin.add_field('B', param, at="node", dtype=np.float32,clobber=True)
    param= load_param(WM,basin_dem.x.values,basin_dem.y.values)
    _= basin.add_field('WM', param, at="node", units='mm',clobber=True, dtype=np.float32)
    param= load_param(KSAT,basin_dem.x.values,basin_dem.y.values)
    _= basin.add_field('Ksat', param, at="node", units='mm/hr',clobber=True, dtype=np.float32)
    param= load_param(IM,basin_dem.x.values,basin_dem.y.values)
    _= basin.add_field('IM', param, at="node", units='%',clobber=True, dtype=np.float32)
    param= load_param(SM0,basin_dem.x.values,basin_dem.y.values)
    _= basin.add_field('SM0', param, at="node", units='%',clobber=True, dtype=np.float32)
    param= load_param(FRICTION,basin_dem.x.values,basin_dem.y.values)
    _= basin.add_field('friction_node', param, at="node",clobber=True, dtype=np.float32)
    friction_link= map_mean_of_link_nodes_to_link(basin, 'friction_node')
    _= basin.add_field('friction', friction_link.astype(np.float32), at="link",clobber=True, units='-')
    _= basin.add_ones('KE',at='node', dtype= np.float32,clobber=True) * KE
    print('-------------------------')
    print('set up domain ...')
    cresthh= CRESTHH(basin,
                     proj=TO_CRS,
                     start= START,
                     end= END,
                     freq=FREQ,
                     precip_path=RAIN_PATH,
                     precip_freq=RAIN_FREQ,
                     precip_pattern= RAIN_PATTERN,
                     evap_path= EVAP_PATH,
                     evap_freq= EVAP_FREQ,
                     evap_pattern= EVAP_PATTERN,
                     outlet=gauges['outlet'],
                     gauges= gauges,
                     outlet_ts= OUTPUT_TS,
                     reinfiltration= REINFILTRATION,
                     excess_rain=EXCESS_RAIN,
                     output_dir= OUTPUT_DIR,
                     output_vars= OUTPUT_VARS,
                     parallel=PARALLEL,
                     verbose=VERBOSE
                    )
    print(cresthh)
    print('-------------------------')
    print('Start runing ...')
    print('-------------------------')
    cresthh.run()

