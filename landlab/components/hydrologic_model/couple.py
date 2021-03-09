'''
This module is a full coupling of physcially based hydrologic process

It solves differential equation with implicit time stepping to satisfy stability

----------------------------------------------
Processes overview (top-down):
----------------------------------------------

1. Canopy interception

2. Overland flow and infiltration

3. Recharge within unsaturated soil

4. Groundwater routing

5. Channel routing

----------------------------------------------
Inputs
----------------------------------------------

1. spatial distributed rainfall field (raster)

2. spatial distributed PET (raster)

3. Canopy (raster)

4. Model parameters (raster)

5. River segments (vector)

----------------------------------------------
Parameters:
----------------------------------------------

1. Surface parameters: manning's n (raster)

1. Soil parameters: WM – mean max soil capacity; B – exponent of VIC equation; SM0 – initial soil moisture; Ksat –       hydraulic conductivity in unsaturated soil layer; theta – soil porosity

2. Groundwater parameters: hydraulic conductivity

'''
__author__ = 'Allen (Zhi) Li'
__date__ = '2021/02/16'


from warnings import warn

import numpy as np

from landlab import Component

class CoupledHydrologicProcess(Component):
'''
    This is the implementation of coupled CREST (hydrologic model) and 2D simplified SWE (routing)
    '''
    _name = "Physical Hydrologic Model"
    _unit_agnostic = False
    _info = {
    # =============Input Parameters==================
        "SM0__param":{
            "dtype": np.float32,
            "intent": "in",
            "optional": False,
            "units": "%",
            "mapping": "node",
            "doc": "Initial Soil Moisture"
        },
        "WM__param":{
            "dtype": np.float32,
            "intent": "in",
            "optional": False,
            "units": "mm",
            "mapping": "node",
            "doc": "Mean Max Soil Capacity"
        },
        "manning_n__param":{
            "dtype": np.float32,
            "intent": "in",
            "optional": False,
            "units": "-",
            "mapping": "link",
            "doc": "Initial Soil Moisture"
        },
        "B__param":{
            "dtype": np.float32,
            "intent": "in",
            "optional": False,
            "units": "-",
            "mapping": "node",
            "doc": "Exponent of VIC model"
        },
        "KE__param":{
            "dtype": np.float32,
            "intent": "in",
            "optional": False,
            "units": "-",
            "mapping": "node",
            "doc": "Evaporation factor -> from PET to AET"
        },
    # ==============States===================

        "topographic__elevation":{
            "dtype": np.float32,
            "intent": "in",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "Surface elevation"
        },
        "aquifer_base__elevation":{
            "dtype": np.float32,
            "intent": "in",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "Base elevation of aquifer"
        },
        "surface_water__discharge":{
            "dtype": np.float32,
            "intent": "out",
            "optional": False,
            "units": "m^3/s",
            "mapping": "node",
            "doc": "Surface discharge"
        },
        "soil_moisture__content":{
            "dtype": np.float32,
            "intent": "out",
            "optional": False,
            "units": "%",
            "mapping": "node",
            "doc": "Soil Moisture Content"
        },
        "surface_water__elevation":{
            "dtype": np.float32,
            "intent": "out",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "Surface water elevation"
        },
        "ground_water__elevation":{
            "dtype": np.float32,
            "intent": "out",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "Ground water table"
        },
        "river__stage":{
            "dtype": np.float32,
            "intent": "out",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "River stage"
        },
        }

        def __init__(self,
                    grid,
                    time_start,
                    time_end,
                    dt,
                    precip_path,
                    evap_path,
                    precip_pattern,
                    evap_patter,
                    ):
            super().__init__(grid)
            # initialize inputs
            self.time_start= pd.to_datetime(start, format='%Y%m%d%H%M%S')
            self.time_end= pd.to_datetime(end, format='%Y%m%d%H%M%S')
            self.dt= pd.Timedelta(dt)
            self.precip_path= precip_path
            self.evap_path= evap_path
            self.precip_pattern= precip_pattern
            self.evap_pattern= evap_pattern

            # initialize states
            self.grid.add_field('soil_moisture__content', self.grid.at_node['SM0__param']*self.grid.at_node['WM__param'],clobber=True, dtype=np.float32)
            self.grid.add_field('ground_water__elevation', self.grid.at_node['aquifer_base__elevation'],clobber=True, dtype=np.float32)
            self.grid.add_field('surface_water__elevation', self.grid.at_node['topographic__elevation'],clobber=True, dtype=np.float32)
            self.grid.add_zeros('surface_water__discharge', clobber=True, dtype=np.float32)


        def _surface_flux(self):
            pass

        def _unsaturated_flux(self):
            pass

        def _ground_water_flux(self):
            pass

        def _river_channel_flux(self):
            pass

        def solver(self):
            pass

        def _map_field_to_nodes(self,fname):
            # map tif file to node locations
            pass

        @property
        def precip(self):
            return self._precip

        @precip.setter
        def precip(self, fname):
            self._precip= _map_field_to_nodes(fname)

        @property
        def evap(self):
            return self.evap

        @evap.setter
        def evap(self, fname):
            self._precip= _map_field_to_nodes(fname)


        def _transform_proj(self, from_CRS, to_CRS):
            pass
