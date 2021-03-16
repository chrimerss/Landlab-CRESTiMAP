'''
This module is a fully coupled and physcially based hydrologic process

It solves differential equation with implicit time stepping to satisfy stability

----------------------------------------------
Processes overview (top-down):
----------------------------------------------

1. Canopy interception

2. Overland flow and infiltration

3. Recharge within unsaturated soil

4. Groundwater routing

5. Channel routing (Lake routing)

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
__date__ = '2021/03/09'


from warnings import warn

import numpy as np

from landlab import Component, LinkStatus
from landlab.components import FlowDirectorSteepest
from landlab.grid import RasterModelGrid
from .utils import _regularize_G, _regularize_R
import fiona
import rioxarray
from scipy.integrate import odeint
# import numpy asn

# define some constants
GRAVITY=0.98


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
        "units": "m",
        "mapping": "node",
        "doc": "Mean Max Soil Capacity"
    },
    "manning_n__param":{
        "dtype": np.float32,
        "intent": "in",
        "optional": False,
        "units": "-",
        "mapping": "node",
        "doc": "manning roughness"
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
    "Ksat_groundwater__param":{
        "dtype": np.float32,
        "intent": "in",
        "optional": False,
        "units": "m/s",
        "mapping": "link",
        "doc": "horizontal hydraulic conductivity in groundwater"
    },
    "Ksat_unsaturated__param":{
        "dtype": np.float32,
        "intent": "in",
        "optional": False,
        "units": "m/s",
        "mapping": "node",
        "doc": "Soil hydraulic conductivity in unsaturated zone"
    },
    "riv_width__param":{
        "dtype": np.float32,
        "intent": "in",
        "optional": False,
        "units": "m",
        "mapping": "node",
        "doc": "River width"
    },
    "riv_topo__param":{
        "dtype": np.float32,
        "intent": "in",
        "optional": False,
        "units": "m",
        "mapping": "node",
        "doc": "River bottom elevation, possibly estimated from high-res DEM"
    },
    "riv_manning__param":{
        "dtype": np.float32,
        "intent": "in",
        "optional": True,
        "units": "m",
        "mapping": "node",
        "doc": "River roughness values, default 0.03"
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
    "aquifer__thickness":{
        "dtype": np.float32,
        "intent": "in",
        "optional": False,
        "units": "m",
        "mapping": "node",
        "doc": "Thickness of confined aquifer"
    },
    "surface_water__discharge":{
        "dtype": np.float32,
        "intent": "out",
        "optional": True,
        "units": "m^3/s",
        "mapping": "link",
        "doc": "Surface discharge"
    },
    "ground_water__discharge":{
        "dtype": np.float32,
        "intent": "out",
        "optional": True,
        "units": "m^3/s",
        "mapping": "link",
        "doc": "Groundwater discharge"
    },
    "soil_moisture__content":{
        "dtype": np.float32,
        "intent": "out",
        "optional": False,
        "units": "m",
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
        "intent": "inout",
        "optional": True,
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
                cwr= 0.7,
                regularization_factor=0.2,
                porosity=1,
                proj='None'
                ):
        '''
        CONSTANT PARAMETERS
        ----------------------------------------------
        :cwr – Weir discharge coefficient
        :regularization_factor – smooth factor in ground water seepage
        :porosity – soil porosity - TODO: how to treat water balance when factoring porosity
        '''
        super().__init__(grid)

        # carve river to grid object
        # self._carve_river(river)

        # initialize states
        self.grid.add_field('soil_moisture__content', self.grid.at_node['SM0__param']*self.grid.at_node['WM__param'],
                        clobber=True, dtype=np.float32, at='node')
        self.grid.add_field('ground_water__elevation', self.grid.at_node['aquifer_base__elevation'],
                        clobber=True, dtype=np.float32, at='node')
        self.grid.add_field('surface_water__elevation', self.grid.at_node['topographic__elevation'],
                        clobber=True, dtype=np.float32, at='node')
        self.grid.add_field('river__stage', self.grid.at_node['riv_topo__param'], dtype=np.float32,
         at='node', clobber=True)
        self.grid.add_zeros('surface_water__discharge', clobber=True, dtype=np.float32, at='node')
        self.grid.add_zeros('surface_water__discharge', clobber=True, dtype=np.float32, at='link')
        self.grid.add_zeros('ground_water__discharge', clobber=True, dtype=np.float32, at='node')
        # self.grid.add_

        self.zsf= self.grid['node']['surface_water__elevation']
        self.zsf_base= self.grid['node']['topographic__elevation']
        if 'ground_water__elevation' not in self.grid['node'].keys():
            self.grid.add_zeros('ground_water__elevation', clobber=True, dtype=np.float32, at='node')
        self.zgw= self.grid['node']['ground_water__elevation'] + self.grid['node']['aquifer_base__elevation']
        self.zgw_base= self.grid['node']['aquifer_base__elevation']
        self._river_cores= self.grid.nodes.reshape(-1)[self.grid['node']['riv_width__param']>0]
        self.zrv= self.grid['node']['river__stage'][self._river_cores]
        self.zrv_btm= self.grid['node']['riv_topo__param']
        self.wrv= self.grid['node']['riv_width__param'][self._river_cores]
        
        self.zrv_bank= np.array([self.zsf[self.grid.active_adjacent_nodes_at_node[_node]].max() for _node in self._river_cores]) #only at river nodes
        self.hus= self.zsf_base - self.zgw # height of unsaturated zone
        self.hsf= self.zsf - self.zsf_base
        self.hgw= self.zgw - self.zgw_base # height of ground water table
        self.haq= self.grid['node']['aquifer__thickness']
        self.zaq= self.zgw_base + self.haq
        if (self.zaq>self.zsf_base).any():
            raise ValueError('confined layer exceeds surface... please check aquifer thickness')
        self.hsf_link= self.grid.map_max_of_link_nodes_to_link(self.hsf)
        self.qsf = self.grid['node']['surface_water__discharge']
        self.qsf_link= self.grid['link']['surface_water__discharge']
        self.qgw= self.grid['node']['ground_water__discharge']
        self.friction= self.grid['node']['manning_n__param']
        self.B= self.grid['node']['B__param']
        self.WM= self.grid['node']['WM__param']
        self.SM= self.grid['node']['soil_moisture__content']
        self.Kgw= self.grid['link']['Ksat_groundwater__param']
        self.Kus= self.grid['node']['Ksat_unsaturated__param']
        self.friction= self.grid['node']['manning_n__param']
        self.friction_link= self.grid.map_max_of_link_nodes_to_link(self.friction)
        self.rv_roughness= self.grid['node']['riv_manning__param']
        self._infiltration= np.zeros_like(self.zsf)
        
        #initialize flow field
        zriv_grid= self.zsf.copy().astype(float)
        zriv_grid[self._river_cores]= self.zrv_btm[self._river_cores]
        new_grid= RasterModelGrid(self.grid.shape, xy_spacing= self.grid.dx)
        new_grid.add_field('topographic__elevation', zriv_grid)
        self._flow_dir= FlowDirectorSteepest(new_grid)
        self._flow_dir.run_one_step()

        self.base_grad= self.grid.calc_grad_at_link(self.zgw_base)

        self._cores= self.grid.core_nodes
        # Model parameters
        self._r= regularization_factor
        self._porosity= porosity
        self._cwr= cwr

        # get geo information from grid
        if proj is None:
            raise ValueError('Please specify model projection by set grid.proj=')
        else:
            self._proj= proj
        self._x= self.grid.x_of_node
        self._y= self.grid.y_of_node
        self._forc_proj= None


    def run_one_step(self,dt):
        '''
        advance in one time step, but inside it may involve multiple time step to obtain a converged solution
        '''
            
        #prepare for the ODE system
        _input= np.concatenate([self.zsf[self._cores], self.SM[self._cores],
                                self.zgw[self._cores], self.zrv])
        
        
    def implicit_update(self, ):
        pass
    
    #TODO
    def _canopy_intercept(self):
        pass

    def _surface_flux(self,zsf, SM, zgw, dt):
        zsf[zsf<self.zsf_base]= self.zsf_base[zsf<self.zsf_base]
        zsf[zsf<zgw]= zgw[zsf<zgw]
        hsf= zsf - self.zsf_base
        print(hsf)
        hsf_link= self.grid.map_max_of_link_nodes_to_link(hsf)
        hsf_link[self.grid.status_at_link==LinkStatus.INACTIVE]= 0.0
        # Here we first do infiltration and recharge, then lateral flow
        #infiltration
        # cond 1: if groundwater table > surface water stage, where exfiltration occurs
        # here we use negative infiltration value
        cond1= np.where(zgw>zsf)
        _infiltration= np.zeros_like(self.zsf_base)
        _infiltration[cond1]= 0
        SM[cond1]= self.WM[cond1]
        zsf[cond1]= zgw[cond1]
        
        # normal condition: groundwater table < surface water stage, where infiltration occurs
        cond2= np.where(zgw<=zsf)
        precipSoil= hsf[cond2]
        Wmaxm= self.WM[cond2] * (self.B[cond2]+1)
        SM[SM<0]= 0
        SM[SM>self.WM]= self.WM[SM>self.WM] #cannot exceed soil capacity
        A = Wmaxm * (1-(1.0-SM[cond2]/Wmaxm)**(1.0/(1.0+self.B[cond2])))
        _infiltration[cond2]=self.WM[cond2]*((1-A/Wmaxm)**(1+self.B[cond2])-(1-(A+precipSoil)/Wmaxm)**(1+self.B[cond2]))
        _infiltration[cond2][_infiltration[cond2]>precipSoil]= precipSoil[_infiltration[cond2]>precipSoil]
#         print(self.WM, A, Wmaxm, self.B, precipSoil)
        #horizontal flow
        # Here we use Bates et al. (2010)
        s0= self.grid.calc_grad_at_link(zsf)
        qsf_link= self.grid.dx/self.friction_link * hsf_link**(5/3)*abs(s0)**0.5*np.sign(s0) #link value
#         qsf_link= (qsf_link-GRAVITY*hsf_link*dt*abs(s0))/(1.0+GRAVITY+\
#                     hsf_link*dt*self.friction_link**2*abs(qsf_link)/hsf_link**(10.0/3.0))
        qsf_link[self.grid.status_at_link==LinkStatus.INACTIVE]= 0.0
        # sum all outflows
        qsf= abs(self.discharge_out_mapper(qsf_link))

        return self._precip - self._evap - qsf/self.grid.dx/self.grid.dy - self._infiltration/dt, _infiltration, SM, qsf, qsf_link
    
    #TODO add lateral flow in the unsaturated zone
    def _unsaturated_flux(self, zgw, SM, _infiltration, dt):
        hus= self.zsf_base - zgw
        hus[self.hus<0]=0
        _recharge= (_infiltration+SM)/self.WM/2*self.Kus
        _recharge[_recharge>_infiltration]= _infiltration[_recharge>_infiltration]
        
        return _infiltration/dt-_recharge/dt, _recharge

    def _ground_water_flux(self,zsf, zgw, zrv, _recharge, dt):
        '''
        Implementation of Dupuit Percolator
        Solves Boussinesq equation with unconfined aquifer
        '''
        zgw[zgw>self.zsf_base]= self.zsf_base[zgw>self.zsf_base]
        zgw[zgw<self.zgw_base]= self.zgw_base[zgw<self.zgw_base]
        hgw= zgw - self.zgw_base

        cosa= np.cos(np.arctan(self.base_grad))
        #calculate hydraulic gradient
        _zgw_grad= self.grid.calc_grad_at_link(zgw) * cosa

        #calculate groundwater velocity
        vel = -self.Kgw * _zgw_grad
        vel[self.grid.status_at_link==LinkStatus.INACTIVE]= 0.0

        # aquifer thickness at links
        hgw_link= self.grid.map_value_at_max_node_to_link(zgw,hgw) * cosa

        #calculate specific discharge
        _q= hgw_link * vel

        #calculate flux divergence
        dqdx= self.grid.calc_flux_div_at_node(_q)

        #determine relative thickness
        soil_present= (zsf - self.zgw_base)>0.0
        rel_thickness = np.ones_like(zsf)
        rel_thickness[soil_present]= np.minimum(1, hgw/(self.zsf_base[soil_present]-self.zgw_base[soil_present]))

        #calculate seepage to surface, only when groundwater table>surface elevation
        cond= np.where(zgw>self.zsf_base)
        _qs= np.zeros_like(self.qsf)
        _qs[cond]= _regularize_G(rel_thickness[cond], self._r) * _regularize_R(_recharge[cond]/dt - dqdx[cond])
        qgw_to_sf= _qs * self.grid.dx*self.grid.dy
        #calculate seepage to river channel
        cond= np.where(zgw[self._river_cores]>zrv)
        qgw_to_riv= _regularize_G(rel_thickness[self._river_cores][cond], self._r) * _regularize_R(_recharge[self._river_cores][cond]/dt - dqdx[self._river_cores][cond])
        qgw= dqdx * self.grid.dx*self.grid.dy

        # mass balance
        _dhdt= (1/self._porosity) * (_recharge/dt - abs(_qs) - abs(dqdx))

        return _dhdt, qgw_to_sf, qgw_to_riv, qgw


    def _river_channel_flux(self,zrv,zsf,zgw,qsf,qsf_link,
                            qgw_to_sf,qgw_to_riv, dt):

        '''
        Channel receives water from overland, baseflow and upstream cell

        Caveats: 1. if river overbank, flow in opposite direction
                 2. if ground water table < river bottom, base flow to river channel is not possible
        '''
        zsf[zsf<self.zsf_base]= self.zsf_base[zsf<self.zsf_base]
        hsf= zsf- self.zsf_base
        qsf+= abs(qgw_to_sf)
        zrv[zrv<self.zrv_btm[self._river_cores]]= self.zrv_btm[self._river_cores][zrv<self.zrv_btm[self._river_cores]]
        # 1) surface
        #   [1] If river stage>bank: overbank flow (Weir flow)
        Qsurf_to_riv= np.zeros_like(qsf[self._river_cores])
        cond1= np.where(zrv>self.zrv_bank)[0]
        if len(cond1)>0:
            Qweir= - self._cwr * (2*GRAVITY*(zrv[cond1] - self.zrv_bank[cond1])**0.5*self.grid.dy*\
                (zrv[cond1] - self.zrv_bank[cond1]))
            Qsurf_to_riv[cond1]= Qweir
        #   [2] If river stage<bank: inflow
        cond2= np.where(zrv<=self.zrv_bank)
        Qinflow=  self._cwr * (2*GRAVITY*(hsf[self._river_cores][cond2]))**0.5*\
                    self.grid.dy*(hsf[self._river_cores][cond2])
        
        Qsurf_to_riv[cond2]= Qinflow
        cond= np.where(Qsurf_to_riv+qsf[self._river_cores]<0)
        Qsurf_to_riv[cond]= qsf[self._river_cores][cond] # In case surface flow < Qinflow, maximum inflow would be qsf to balance water
        # 2) subsurface
        #   [1] If groundwater table>river bed (inflow)
        Qsub_to_riv = np.zeros_like(qsf[self._river_cores])
        cond1= np.where(zgw[self._river_cores]>self.zrv_btm[self._river_cores])[0]

        if len(cond1)>0:
            Qsub_to_riv[cond1]= qgw_to_riv[cond1] # positive
        #   [2] If ground water table < river bed (recharge) Darcy's law q=dh/dx * Ksat
        cond2= np.where(zgw[self._river_cores]<self.zrv_btm[self._river_cores])
        Qsub_to_riv[cond2]= - (zrv[cond2]-self.zrv_btm[self._river_cores][cond2])*self.Kus[self._river_cores][cond2]*\
                                self.grid.dx # negative sign to represent direction
        _recharge_next=-Qsub_to_riv.copy()
        _recharge_next[cond1]=0
        zgw[self._river_cores]+= _recharge_next/self.grid.dx/self.grid.dx * dt
        # manning's equation for downstream flow
        
        # 3) Downward flow: Manning equation

        Qdown= self._apply_manning_eq(zrv)      
        self.map_node_to_downstream_link(Qdown, self._river_cores)

        # 4) receiption of upstream flow
        Qup= abs(self.discharge_in_mapper(qsf_link)[self._river_cores])

#         print(Qsurf_to_riv, Qsub_to_riv, Qup, Qdown)
        
        qsf[self._river_cores]= (Qup+Qsub_to_riv+Qsurf_to_riv-Qdown)

        return (Qsurf_to_riv + Qsub_to_riv + Qup + Qdown)/self.grid.dx/self.grid.dy, qsf

    def map_node_to_downstream_link(self, values_at_node, node_ids):
        '''
        Inputs:
        ----------------------
        values_at_node - node values
        node_ids - node id, the same dimension and order as values_at_node
        
        Output:
        ----------------------
        values_at_link
        '''
        links= self._flow_dir.links_to_receiver[node_ids]
        self.qsf_link[links]= values_at_node
        
    def discharge_in_mapper(self, input_discharge):
        '''
        From Adams et al., 2017
        
        This method takes the discharge values on links and determines the
        links that are flowing INTO a given node. The fluxes moving INTO a
        given node are summed.

        This method ignores all flow moving OUT of a given node.

        This takes values from the OverlandFlow component (by default) in
        units of [L^2/T]. If the convert_to_cms flag is raised as True, this
        method converts discharge to units [L^3/T] - as of Aug 2016, only
        operates for square RasterModelGrid instances.

        The output array is of length grid.number_of_nodes and can be used
        with the Landlab imshow_grid plotter.

        Returns a numpy array (discharge_vals)        
        '''
        discharge_vals = np.zeros(self._grid.number_of_links)
        discharge_vals[:] = input_discharge[:]

        discharge_vals = (
            discharge_vals[self._grid.links_at_node] * self._grid.link_dirs_at_node
        )

        discharge_vals = discharge_vals.flatten()

        discharge_vals[np.where(discharge_vals > 0)] = 0.0

        discharge_vals = discharge_vals.reshape(self._grid.number_of_nodes, 4)

        discharge_vals = np.nansum(discharge_vals,axis=1)

        return discharge_vals     
    
    def discharge_out_mapper(self, input_discharge):
        '''
        From Adams et al., 2017
        
        This method takes the discharge values on links and determines the
        links that are flowing INTO a given node. The fluxes moving INTO a
        given node are summed.

        This method ignores all flow moving OUT of a given node.

        This takes values from the OverlandFlow component (by default) in
        units of [L^2/T]. If the convert_to_cms flag is raised as True, this
        method converts discharge to units [L^3/T] - as of Aug 2016, only
        operates for square RasterModelGrid instances.

        The output array is of length grid.number_of_nodes and can be used
        with the Landlab imshow_grid plotter.

        Returns a numpy array (discharge_vals)        
        '''
        discharge_vals = np.zeros(self._grid.number_of_links)
        discharge_vals[:] = input_discharge[:]

        discharge_vals = (
            discharge_vals[self._grid.links_at_node] * self._grid.link_dirs_at_node
        )

        discharge_vals = discharge_vals.flatten()

        discharge_vals[np.where(discharge_vals < 0)] = 0.0

        discharge_vals = discharge_vals.reshape(self._grid.number_of_nodes, 4)

        discharge_vals = np.nansum(discharge_vals,axis=1)

        return discharge_vals      
        

    def _carve_river(self, vector):
        '''
        Carve river node into grid object for channel routing, this is implementaed in Cython file
        '''

        pass

    def _apply_manning_eq(self, zrv,):
        downQ= np.zeros_like(zrv)
        cond= zrv>self.zrv_btm[self._river_cores]
        cross_section_area= (zrv[cond]-self.zrv_btm[self._river_cores][cond])*self.wrv[cond]
        wet_perimeter= (zrv[cond]-self.zrv_btm[self._river_cores][cond])*2+self.wrv[cond]
        slope= self.grid.calc_grad_at_link(self.zrv_btm)[self._river_cores][cond]
#         print(slope)
#         roughness= self.grid.map_max_of_link_nodes_to_link(self.riv_roughness)
        downQ[cond]= cross_section_area/self.rv_roughness[self._river_cores][cond]*(cross_section_area/wet_perimeter)**2/3*abs(slope)**0.5*np.sign(slope)
        
        return downQ
    
    def free_flux(self):
        self.qsf=0
        self.qsf_link=0
        self.qgw= 0
        
        
    def solver(self):
        pass

    def _map_field_to_nodes(self,fname):
        # map tif file to node locations
        if fname.split('.')[-1] in ['asc', 'tif']:
            field= rioxarray.open_rasterio(fname)
            field= field.rio.write_crs(self._forc_proj).rio.reproject(self._proj)
            field= field.sel(x=self.lons, y= self.lats, method='nearest')
            return field.values
        else:
            raise ValueError('only support .tif or .asc file')

    @property
    def precip(self):
        return self._precip

    @precip.setter
    def precip(self, _input):
        '''
        set precipitation field
        Input a file name or a numpy array
        
        If it is a file name:
            self._forc_proj cannot be None because we need to transform forcing projection to grid projection
        If it is a numpy array:
            you need to make sure it has the same dimension as grid
        '''
        if type(_input) is str:
            if self._forc_proj is None:
                raise ValueError('Please set forcing projection first by grid.forc_proj=')
            ifExists= self._check_fn_exists(_input, raiseError=False)
            if ifExists:
                self._precip= _map_field_to_nodes(_input)
            else:
                self._precip= np.zeros(self.grid.shape)
        elif type(_input) is np.ndarray:
            status= self._check_same_dimension(_input)
            if status==0:
                msg= 'precipitation array does not have the same dimension as grid'
                raise ValueError(msg)
            self._precip= _input.reshape(-1)
        else:
            raise ValueError('Unsupported input type %s'%(type(_input)))

    @property
    def evap(self):
        return self.evap

    @evap.setter
    def evap(self, _input):
        '''
        set evaporation field
        Input a file name and we load it and transform to model domain
        '''
        if type(_input) is str:
            if self._forc_proj is None:
                raise ValueError('Please set forcing projection first by grid.forc_proj=')
            ifExists= self._check_fn_exists(_input, raiseError=False)
            if ifExists:
                self._evap= _map_field_to_nodes(_input)
            else:
                self._evap= np.zeros(self.grid.shape)
        elif type(_input) is np.ndarray:
            status= self._check_same_dimension(_input)
            if status==0:
                msg= 'precipitation array does not have the same dimension as grid'
                raise ValueError(msg)
            self._evap= _input.reshape(-1)
        else:
            raise ValueError('Unsupported input type %s'%(type(_input)))            

    @property
    def proj(self):
        return self._proj

    @proj.setter
    def proj(self, proj):
        self._proj= proj
        
    @property
    def forc_proj(self):
        return self._forc_proj

    @forc_proj.setter
    def forc_proj(self, proj):
        self._forc_proj= proj      

    def _transform_proj(self, from_CRS, to_CRS):
        pass

    def _check_fn_exists(self,fname,raiseError=False):
        if not os.path.exists(fname):
            if raiseError:
                raise FileNotFoundError('%s file does not exist, please check carefully'%fname)
            else:
                warn('%s file does not exist, please check carefully'%fname, RuntimeWarning)
                return 0
        else:
            return 1
        
    def _check_same_dimension(self, arr):
        if arr.shape != self.grid.shape:
            return 0
        else: return 1


    def _rasterize_polygons(self, polygon_shp, template_raster,
                             fields):
        """ generate a categorical raster based on polygons

        :rtype: None
        :param polygon_shp: input polygon shapefile
        :param template_raster: raster template for cellsize and extent
        :param out_raster: output raster file
        """
        out_raster= 'temp.tif'

        gdal.UseExceptions()
        # Open the data source and read in the extent
        source_ds = ogr.Open(polygon_shp)
        source_layer = source_ds.GetLayer()

        target_ds = gdal.GetDriverByName('GTiff').Create(out_raster, len(self.grid._x),
                                         len(self.grid._y), 1, gdal.GDT_Float32)
        target_ds.SetGeoTransform([self.grid._x[0], self.grid.dx,0, self.grid._y[0], 0, self.grid.dy])
        if isinstance(self._proj, str):
            target_ds.SetProjection(self._proj)
        else:
            raise ValueError('Initialize grid projection first')

        band = target_ds.GetRasterBand([1,2,3])
        band.SetNoDataValue(-9999.)
        # Rasterize
        gdal.RasterizeLayer(target_ds, [1,2,3], source_layer, options=["ATTRIBUTE={}".format(field)])
