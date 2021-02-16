from .crest_simp import CREST
from landlab.components import OverlandFlow
from landlab import Component, FieldError
import xarray as xr
import rioxarray
import pandas as pd
from datetime import datetime
import numpy as np
import multiprocess
from pathos.multiprocessing import ProcessingPool as Pool
from pathos.helpers import freeze_support
freeze_support()
import dill
import os


#######################################
import sys
import types
#Difference between Python3 and 2
if sys.version_info[0] < 3:
    import copy_reg as copyreg
else:
    import copyreg

def _pickle_method(m):
    class_self = m.im_class if m.im_self is None else m.im_self
    return getattr, (class_self, m.im_func.func_name)

copyreg.pickle(types.MethodType, _pickle_method)

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
                    parallel=0):
        '''
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
        parallel: int, number of cores to parallelize, if 0, then single thread
        '''
        super().__init__(grid)
        self.time_start= pd.to_datetime(start, format='%Y%m%d%H%M%S')
        self.time_end= pd.to_datetime(end, format='%Y%m%d%H%M%S')
        self.precip_path= precip_path
        self.evap_path= evap_path
        self.precip_freq= precip_freq
        self.evap_freq= evap_freq
        self.precip_pattern= precip_pattern
        self.evap_pattern= evap_pattern
        self.parallel= parallel

        # configuration
        self.precip_time_stamp= pd.date_range(self.time_start,
                                            self.time_end,
                                             freq=precip_freq)
        self.evap_time_stamp= pd.date_range(self.time_start,
                                            self.time_end,
                                             freq=evap_freq)
        self.grid.add_field('SM', self.grid.at_node['SM0']/100.* self.grid.at_node['WM'],
                         clobber=True)
        #  initialize surface water depth
        grid.add_zeros('surface_water__depth', at='node', clobber=True, dtype=np.float32)
        grid.at_node['surface_water__depth'].fill(1e12)
        self.flow= OverlandFlow(grid, steep_slopes=True)
        xllcorner, yllcorner= grid._xy_of_lower_left

        self.lons= grid.x_of_node.reshape(grid.shape)[0,:]; self.lats= grid.y_of_node.reshape(grid.shape)[:,0]
        self.proj= proj
        self.output= {}
        self.m, self.n= self.grid.shape

        if parallel>0:
            self.pool= Pool(nodes=self.parallel)



    def run(self, verbose=True):

        end_seconds= (self.time_end- self.time_start).total_seconds()
        time_now= 0
        time_UTC= self.time_start
        precip_count= []
        evap_count= []
        while time_now<end_seconds:
            # update rainfall field
            if time_now//pd.Timedelta(self.precip_freq).total_seconds() not in precip_count:
                try:
                    fname_precip= self._get_fname(self.precip_path, self.precip_pattern, time_UTC)
                    P= self.map_to_grid(fname_precip)
                except OSError:
                    msg= '%s not found in precipitation, assume 0 everywhere'%fname_precip
                    print(msg)
                finally:
                    precip_count.append(time_now//pd.Timedelta(self.precip_freq).total_seconds())

            if time_now//pd.Timedelta(self.evap_freq).total_seconds() not in evap_count:
                try:
                    fname_evap= self._get_fname(self.evap_path, self.evap_pattern, time_UTC)
                    ET= self.map_to_grid(fname_evap)
                except OSError:
                    msg= '%s not found in evaporation, assume 0 everywhere'%fname_evap
                    print(msg)
                finally:
                    evap_count.append(time_now//pd.Timedelta(self.evap_freq).total_seconds())

            self.grid.add_field('P', P, units='mm/h', clobber=True, dtype=np.float32)
            self.grid.add_field('ET', ET, units='mm/d', clobber=True, dtype=np.float32)
            dt= self.flow.calc_time_step()
            # if self.parallel>0:
            #     self.pool.map(self.single, [i for i in range(self.m*self.n)])
            # else:
            #     [self.single(i) for i in range(self.m*self.n)]
            self.single(dt)

            self.flow.run_one_step(dt=dt)

            time_now+= dt
            time_UTC+= pd.Timedelta(seconds=dt)
            if verbose:
                print('time: %s, outlet discharge: %.2f'%time_UTC, self.flow._grid.at_node['surface_water__depth'].max())

    def single_thread(self):
        P= self.flow._grid.at_node['P'][node]/1000./3600. #input unit in mm/hr
        ET= self.flow._grid.at_node['ET'][node]/1000./3600./24. # input unit in mm/day
        B= self.flow._grid.at_node['B'][node]
        IM= self.flow._grid.at_node['IM'][node]/100.
        Ksat= self.flow._grid.at_node['Ksat'][node]
        WM= self.flow._grid.at_node['WM'][node]
        SM= (self.flow._grid.at_node['SM'][node]*WM)/1000.
        depth= self.flow._grid.at_node['surface_water__depth'][node]
        KE= self.flow._grid.at_node['KE'][node]
        SM, overland, interflow,actET= CREST(P, depth, ET, SM, Ksat, WM, B, IM, KE, dt)
        self.flow._grid.at_node['surface_water__depth'][node]= overland
        SM*=(1000/WM)
        self.flow._grid.at_node['SM'][node]= SM

    def single(self, dt):
        P= self.flow._grid.at_node['P']/1000./3600. #input unit in mm/hr
        ET= self.flow._grid.at_node['ET']/1000./3600./24. # input unit in mm/day
        B= self.flow._grid.at_node['B']
        IM= self.flow._grid.at_node['IM']/100.
        Ksat= self.flow._grid.at_node['Ksat']
        WM= self.flow._grid.at_node['WM']
        SM= (self.flow._grid.at_node['SM']*WM)/1000.
        depth= self.flow._grid.at_node['surface_water__depth']
        KE= self.flow._grid.at_node['KE']
        DT= np.ones(len(KE) ).astype(np.float32)* dt
        array_in= np.stack([P, depth, ET, SM, Ksat, WM, B, IM, KE, DT])
        array_out= np.apply_along_axis(self.single_numpy, 0, array_in)
        # SM, overland, interflow,actET=
        SM= array_out[0]
        self.flow._grid.at_node['surface_water__depth']= array_out[1]
        SM*=(1000/WM)
        self.flow._grid.at_node['SM']= SM

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
