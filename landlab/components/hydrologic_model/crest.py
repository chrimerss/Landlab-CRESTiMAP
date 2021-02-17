from .crest_simp import CREST
from landlab.components import OverlandFlow
from landlab import Component, FieldError
from landlab.io import write_esri_ascii
import xarray as xr
import rioxarray
import pandas as pd
from datetime import datetime
import numpy as np
from netCDF4 import Dataset
from affine import Affine
from pyproj import Proj, transform
import os
import time

__author__ = 'Allen (Zhi) Li'
__date__ = '2021/02/16'

#######################################
# Pickle class in case for parallel computing
# import sys
# import types
# #Difference between Python3 and 2
# if sys.version_info[0] < 3:
#     import copy_reg as copyreg
# else:
#     import copyreg

# def _pickle_method(m):
#     class_self = m.im_class if m.im_self is None else m.im_self
#     return getattr, (class_self, m.im_func.func_name)

# copyreg.pickle(types.MethodType, _pickle_method)

def map_gauge_loc_to_node(grid, lon, lat, from_proj, to_proj):
    '''
    Locate gauge to grid node ids.

    Args:
    ----------------------------
    grid: RasterModelGrid
    lon: float, longitude in EPSG:4326
    lat: float

    Output:
    ----------------------------
    node_id
    '''
    from_proj= Proj(from_proj)
    to_proj= Proj(to_proj)
    x,y= transform(from_proj, to_proj, lon, lat)
    x_node= grid.x_of_node
    y_node= grid.y_of_node
    node_id= np.argmin((x-x_node)**2+ (y-y_node)**2)

    return node_id


class CRESTHH(Component):
    '''
    This is the implementation of coupled CREST (hydrologic model) and 2D simplified SWE (routing)
    '''
    _name = "CRESTHH"
    _unit_agnostic = False
    _info = {
    # =============Input Parameters==================
        "SM0":{
            "dtype": np.float32,
            "intent": "in",
            "optional": False,
            "units": "%",
            "mapping": "node",
            "doc": "Initial Soil Moisture"
        },
        "WM":{
            "dtype": np.float32,
            "intent": "in",
            "optional": False,
            "units": "mm",
            "mapping": "node",
            "doc": "Mean Max Soil Capacity"
        },
        "friction":{
            "dtype": np.float32,
            "intent": "in",
            "optional": False,
            "units": "-",
            "mapping": "link",
            "doc": "Initial Soil Moisture"
        },
        "B":{
            "dtype": np.float32,
            "intent": "in",
            "optional": False,
            "units": "-",
            "mapping": "node",
            "doc": "Exponent of VIC model"
        },
        "IM":{
            "dtype": np.float32,
            "intent": "in",
            "optional": False,
            "units": "%",
            "mapping": "node",
            "doc": "Impervious area ratio"
        },
        "KE":{
            "dtype": np.float32,
            "intent": "in",
            "optional": False,
            "units": "-",
            "mapping": "node",
            "doc": "Evaporation factor -> from PET to AET"
        },
    # ==============States===================
        "SM":{
            "dtype": np.float32,
            "intent": "out",
            "optional": False,
            "units": "%",
            "mapping": "node",
            "doc": "Soil Moisture"
        }}
    #     "surface_water__depth": {
    #         "dtype": float,
    #         "intent": "inout",
    #         "optional": False,
    #         "units": "m",
    #         "mapping": "node",
    #         "doc": "Depth of water on the surface",
    #     },
    #     "surface_water__discharge": {
    #         "dtype": float,
    #         "intent": "out",
    #         "optional": False,
    #         "units": "m3/s",
    #         "mapping": "link",
    #         "doc": "Volumetric discharge of surface water",
    #     },
    #     "topographic__elevation": {
    #         "dtype": float,
    #         "intent": "in",
    #         "optional": False,
    #         "units": "m",
    #         "mapping": "node",
    #         "doc": "Land surface topographic elevation",
    #     },
    #     "water_surface__gradient": {
    #         "dtype": float,
    #         "intent": "out",
    #         "optional": False,
    #         "units": "-",
    #         "mapping": "link",
    #         "doc": "Downstream gradient of the water surface.",
    #     },
    # }

    def __init__(self, grid,
                    proj,
                    start,
                    end,
                    freq,
                    precip_path,
                    precip_freq,
                    precip_pattern,
                    evap_path,
                    evap_pattern,
                    evap_freq,
                    outlet,
                    outlet_ts=True,
                    gauges= None,
                    reinfiltration=True,
                    output_dir= None,
                    output_vars=[],
                    parallel=0):
        '''
        Args:
        ------------------------------------------------------------
        grid: RasterGrid
        proj: str, e.g., 'EPSG:32215'
        start: str, time to start %Y%m%d%H%M%S
        end: str, time to end simulation %Y%m%d%H%M%S
        freq: str, frequency to store output
        precip_path: str, precipitation path
        precip_freq: str, precipitation frequency e.g., 2T
        precip_pattern: str, e.g., precip.%Y%m%d.%H%M%S.tif
        evap_path: str, Potential evaporation path
        evap_freq: str, evaporation frequency e.g., 2T
        evap_pattern: str, e.g., evap.%Y%m%d.%H%M%S.tif
        outlet: int, node number for basin outlet
        gauges: dict of basin id's that output time series, e.g., {'08076700': 150}
        reinfiltration: bool, whether to activate reinfiltration scheme
        output_dir: str; where to store variables
        output_vars: list, list of variables to store, default is empty list
        parallel: int, number of cores to parallelize, if 0, then single thread
        '''
        super().__init__(grid)
        self.time_start= pd.to_datetime(start, format='%Y%m%d%H%M%S')
        self.time_end= pd.to_datetime(end, format='%Y%m%d%H%M%S')
        self.freq= pd.Timedelta(freq)
        self.precip_path= precip_path
        self.evap_path= evap_path
        self.precip_freq= precip_freq
        self.evap_freq= evap_freq
        self.precip_pattern= precip_pattern
        self.evap_pattern= evap_pattern
        self.parallel= parallel
        self.outlet_id= outlet
        self.reinfiltration= True
        self.output_dir= output_dir
        self.output_vars= output_vars

        # configuration
        self.precip_time_stamp= pd.date_range(self.time_start,
                                            self.time_end,
                                             freq=precip_freq)
        self.evap_time_stamp= pd.date_range(self.time_start,
                                            self.time_end,
                                             freq=evap_freq)
        self.grid.add_field('SM', self.grid.at_node['SM0'],clobber=True)
        #  initialize surface water depth
        self.grid.add_zeros('surface_water__depth', at='node', clobber=True, dtype=float)
        self.grid.at_node['surface_water__depth'].fill(1e-10)

        xllcorner, yllcorner= grid._xy_of_lower_left

        self.lons= grid.x_of_node.reshape(grid.shape)[0,:]; self.lats= grid.y_of_node.reshape(grid.shape)[:,0]
        self.proj= proj
        self.output= {}
        self.m, self.n= self.grid.shape

        if parallel>0:
            self.pool= Pool(nodes=self.parallel)

        self.outlet_ts= outlet_ts
        if self.outlet_ts:
            self.outlet= {}
            self.monitored_id= []
            if gauges is None: gauges= {'outlet': self.outlet_id}
            for stn, node_id in gauges.items():
                self.outlet[stn]= {'time':[], 'Q':[], 'H':[], 'P':[], 'SM':[]}
                self.monitored_id.append(node_id)


    def run(self, verbose=True):

        end_seconds= (self.time_end- self.time_start).total_seconds()
        time_now= 0
        time_UTC= self.time_start
        precip_count= []
        evap_count= []
        freq_count= []
        start_timer= time.time()
        while time_now<end_seconds:
            # update evaporation
            if time_now//pd.Timedelta(self.evap_freq).total_seconds() not in evap_count:
                try:
                    evap_stamp= self.evap_time_stamp[int(time_now//pd.Timedelta(self.evap_freq).total_seconds())]
                    fname_evap= self._get_fname(self.evap_path, self.evap_pattern, time_UTC)
                    ET= self.map_to_grid(fname_evap)
                except OSError:
                    msg= '%s not found in evaporation, assume 0 everywhere'%fname_evap
                    print(msg)
                    ET= np.zeros((self.m, self.n))
                finally:
                    evap_count.append(time_now//pd.Timedelta(self.evap_freq).total_seconds())
            # update rainfall field
            if time_now//pd.Timedelta(self.precip_freq).total_seconds() not in precip_count:
                try:
                    precip_stamp= self.precip_time_stamp[int(time_now//pd.Timedelta(self.precip_freq).total_seconds())]
                    fname_precip= self._get_fname(self.precip_path, self.precip_pattern, precip_stamp)
                    P= self.map_to_grid(fname_precip)
                except OSError:
                    msg= '%s not found in precipitation, assume 0 everywhere'%fname_precip
                    print(msg)
                    P= np.zeros((self.m, self.n))
                finally:
                    precip_count.append(time_now//pd.Timedelta(self.precip_freq).total_seconds())


                self.grid.add_field('P', P, units='mm/h', clobber=True, dtype=float)
                self.grid.add_field('ET', ET, units='mm/d', clobber=True, dtype=np.float32)
                self.single(pd.Timedelta(self.precip_freq).total_seconds())
                self.flow= OverlandFlow(self.grid, steep_slopes=True, mannings_n='friction')

            dt= self.flow.calc_time_step()

            self.flow.run_one_step(dt=dt)
            self.grid['node']['surface_water__discharge']= self.flow.discharge_mapper(self.flow._q, convert_to_volume=True)
            # self.grid['node']['water_surface__gradient']= (self.flow._water_surface_slope[self.grid.links_at_node]*self.grid.active_link_dirs_at_node).max(axis=1)
            time_now+= dt
            time_UTC+= pd.Timedelta(seconds=dt)
            if time_now//self.freq.total_seconds() not in freq_count:

                time_freq= self.time_start+ self.freq*(time_now//self.freq.total_seconds())
                #save indicated variables
                if self.output_dir is not None:
                    if not os.path.exists(self.output_dir):
                        os.system('mkdir %s'%self.output_dir)
                    self.export_to_asc(time_freq, self.output_dir, self.output_vars)
                end_timer= time.time()
                if verbose:
                    print('time: %s\n----------------------------\nelapsed time: %.2f hrs\ntime step: %.2f sec\noutlet depth: %.2f m\nSM: %.1f%%\nwater surface gradient: %.2f\ndischarge: %.2f m^3/s\n-----------------------'%(
                             time_UTC,
                             (end_timer- start_timer)/3600.,
                             dt,
                             self.grid.at_node['surface_water__depth'][self.outlet_id],
                             self.grid.at_node['SM'].mean(), self.flow._water_surface_slope.max(),
                             self.grid['node']['surface_water__discharge'][self.outlet_id]))
                if self.outlet_ts:
                    for i, _id in enumerate(self.monitored_id):
                        stn= list(self.outlet.keys())[i]
                        self.outlet[stn]['time'].append(time_freq.strftime('%Y-%m-%d %H:%M:%S'))
                        self.outlet[stn]['H'].append(self.grid['node']['surface_water__depth'][_id])
                        self.outlet[stn]['Q'].append(self.grid['node']['surface_water__discharge'][_id])
                        self.outlet[stn]['P'].append(np.nanmean(P))
                        self.outlet[stn]['SM'].append(np.nanmean(self.grid['node']['SM']))
                freq_count.append(time_now//self.freq.total_seconds())

        for stn in self.outlet.keys():
            df= pd.DataFrame(index=self.outlet[stn]['time'])
            df['discharge (m^3/s)']= self.outlet[stn]['Q']
            df['water_depth (m)']= self.outlet[stn]['H']
            df['soil_moisture (%)']= self.outlet[stn]['SM']
            df['precipitation (mm/hr)']= self.outlet[stn]['P']
            df.to_csv(os.path.join(self.output_dir, 'ts.%s.csv'%stn))

    def single_thread(self):
        #reserve for future parallel implementation
        P= self.grid.at_node['P'][node]/1000./3600. #input unit in mm/hr
        ET= self.grid.at_node['ET'][node]/1000./3600./24. # input unit in mm/day
        B= self.grid.at_node['B'][node]
        IM= self.grid.at_node['IM'][node]/100.
        Ksat= self.grid.at_node['Ksat'][node]
        WM= self.grid.at_node['WM'][node]
        SM= (self.grid.at_node['SM'][node]*WM)/1000.
        depth= self.grid.at_node['surface_water__depth'][node]
        KE= self.grid.at_node['KE'][node]
        SM, overland, interflow,actET= CREST(P, depth, ET, SM, Ksat, WM, B, IM, KE, dt)
        self.grid.at_node['surface_water__depth'][node]= overland
        SM*=(1000/WM)
        self.grid.at_node['SM'][node]= SM

    def single(self, dt):
        '''Node-based vertical transform is called'''
        #only calculate active nodes
        _active_node= (self.grid.status_at_node!=self.grid.BC_NODE_IS_CLOSED)
        P= self.grid.at_node['P'][_active_node]/1000./3600. #input unit in mm/hr
        ET= self.grid.at_node['ET'][_active_node]/1000./3600./24. # input unit in mm/day
        B= self.grid.at_node['B'][_active_node]
        IM= self.grid.at_node['IM'][_active_node]/100.
        Ksat= self.grid.at_node['Ksat'][_active_node]
        WM= self.grid.at_node['WM'][_active_node]
        SM= (self.grid.at_node['SM'][_active_node]*WM)/1000./100.
        depth= self.grid.at_node['surface_water__depth'][_active_node]
        KE= self.grid.at_node['KE'][_active_node]
        DT= np.ones(len(KE) ).astype(np.float32)* dt
        if self.reinfiltration:
            array_in= np.stack([P, depth, ET, SM, Ksat, WM, B, IM, KE, DT])
        else:
            array_in= np.stack([P, np.zeros(len(P)), ET, SM, Ksat, WM, B, IM, KE, DT])
        array_out= np.apply_along_axis(self.single_numpy, 0, array_in)
        # SM, overland, interflow,actET=
        SM= array_out[0]
        # print('Maximum water depth by CREST: ', array_out[1].max())
        self.grid.at_node['surface_water__depth'][_active_node]= array_out[1]
        # self.flow._grid["link"]["surface_water__depth"]= self.flow._grid.map_max_of_link_nodes_to_link(array_out[1])
        SM*=(1000/WM)
        self.grid.at_node['SM'][_active_node]= SM*100

    def single_numpy(self, args):
        # precipIn, double overland, double petIn, double SM, double Ksat,
        # double WM, double B, double IM, double KE,
        # double timestep
        return CREST(args[0], args[1],args[2], args[3], args[4],
                    args[5],args[6],args[7],args[8],args[9])


    def map_to_grid(self, fname):
        field= rioxarray.open_rasterio(fname)
        field= field.rio.write_crs('EPSG:4326').rio.reproject(self.proj)
        field= field.sel(x=self.lons, y= self.lats, method='nearest')

        return field.values

    def export_to_asc(self, timestamp, dst_dir, fields, verbose=False):
        '''
        Export field to geotiff and reproject to EPSG:4326

        Args:
        -------------------
        timestamp: str, %Y%m%d.%H%M%S
        dst: str; output dir
        field: list or any iterable str, e.g., ['SM', 'surface_water__depth']

        Output:
        --------------------
        file in the name: %Y%m%d.%H%M%S_field,asc, e.g., 20170825.060000_surface_water__depth.asc
        '''
        for field in fields:
            if field not in self.grid['node'].keys():
                msg= 'invalid field name, only %s are suportted'%self.grid['node'].keys()
                raise KeyError(msg)

        dst= os.path.join(dst_dir, timestamp.strftime('%Y%m%d.%H%M%S.asc'))
        write_esri_ascii(dst, self.grid, names=fields)
        if verbose:
            print(dst, 'saved!')


    #TODO
    def export_netcdf(self,field):
        pass

    def _get_fname(self, path, pattern, time):
        fname= os.path.join(path, pattern.replace('%Y', '%04d'%time.year).replace('%m', '%02d'%time.month).\
            replace('%d', '%02d'%time.day).replace('%H', '%02d'%time.hour).replace('%M', '%02d'%time.minute).\
            replace('%S', '%02d'%time.second).replace('%y', '%02d'%(time.year-2000)))

        return fname

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['pool']
        return self_dict

    def ___call__(self,node):
        return self.single(node)

    def __str__(self):
        # print some basics, e.g., number of cells, dx,
        pass
