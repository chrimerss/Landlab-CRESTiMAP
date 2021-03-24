from .groundwater import Groundwater
from .overland import OverlandFlow
import numpy as np
from landlab import Component

class CRESTPHYS(Component):

    _name= "CREST-Physical"
    _cite_as= '''
                add later
              '''
    _info = {
        "WM__param":{
            "dtype": np.float32,
            "intent": "in",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "Mean Max Soil Capacity"
        },
        "IM__param":{
            "dtype": np.float32,
            "intent": "in",
            "optional": False,
            "units":"%",
            "mapping": "node",
            "doc":"impervious area ratio for generating fast runoff"
        },
        "manning_n__param":{
            "dtype": np.float32,
            "intent": "in",
            "optional": True,
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
        "Ksat_soil__param":{
            "dtype": np.float32,
            "intent": "in",
            "optional": False,
            "units": "m/s",
            "mapping": "node",
            "doc": "vertical Soil saturated hydraulic conductivity in vadose zone"
        },
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
        "ground_water__discharge":{
            "dtype": np.float32,
            "intent": "out",
            "optional": False,
            "units": "m^3/s",
            "mapping": "node",
            "doc": "Groundwater discharge"
        },
        "soil_moisture__content":{
            "dtype": np.float32,
            "intent": "out",
            "optional": False,
            "units": "mm",
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
        }

    def __init__(self,
                 grid,
                 porosity=1,
                 proj=None):

        super().__init__(grid)

        self.initialize_output_fields()


        #===store some parameters for use=========#
        self._im= self._grid['node']['IM__param']/100. # convert to unitness value
        self._im[self._im>1]=1
        self._im[self._im<0]=0
        self._ksat_soil= self._grid['node']['Ksat_soil__param']
        self._ksat_gw= self._grid['link']['Ksat_groundwater__param']
        self._ke= self._grid['node']['KE__param']
        self._b= self._grid['node']['B__param']
        self._b[self._b<0]=1
        self._wm= self._grid['node']['WM__param']
        self._wm[self._wm<0]= 100
        self._ksat_soil[self._ksat_soil<0]=1
        
        self._manning_n= self._grid['node']['manning_n__param']
        self._manning_n_link= self._grid.map_mean_of_link_nodes_to_link(self._manning_n)

        self._zsf= self._grid['node']['surface_water__elevation']
        self._elev= self._grid['node']['topographic__elevation']
        self._zgw= self._grid['node']['ground_water__elevation']
        self._z_base= self._grid['node']['aquifer_base__elevation']

        self._qsf = self._grid['node']['surface_water__discharge']
        self._qgw= self._grid['node']['ground_water__discharge']
        self._sm= self._grid['node']['soil_moisture__content']
        self._sm[self._sm<0]=0
        
        self._zsf[self._grid.status_at_node==self._grid.BC_NODE_IS_CLOSED]= 0
        self._zgw[self._grid.status_at_node==self._grid.BC_NODE_IS_CLOSED]= 0
        self._qsf[self._grid.status_at_node==self._grid.BC_NODE_IS_CLOSED]= 0
        self._qgw[self._grid.status_at_node==self._grid.BC_NODE_IS_CLOSED]= 0

        #===instantiate groundwater and flow accumulator component===#

        self._router= OverlandFlow(
                                    self._grid,
                                    rainfall_intensity=0,
                                    mannings_n= self._manning_n_link,
                                    steep_slopes=True
                                    )
        
        self._gw= Groundwater(
                            self._grid,
                            hydraulic_conductivity=self._ksat_gw,
                            porosity=porosity,
                            recharge_rate=0,
                            regularization_f=0.01,
                            courant_coefficient=1)

        self.lons= self._grid.x_of_node.reshape(self._grid.shape)[0,:]
        self.lats= self._grid.y_of_node.reshape(self._grid.shape)[:,0]


    def run_one_step(self, dt):
        '''
        control function to link overland flow and ground water
        '''
        if not hasattr(self, '_precip') or not hasattr(self, '_evap'):
            msg= 'Missing precipitation or evaporation information, please check...'
            raise ValueError(msg)
        zsf= self._grid['node']['surface_water__elevation']
        zgw= self._grid['node']['ground_water__elevation']
        # here we combine surface water and precipitation to turn on reinfiltration
        precip= self._precip * dt   #convert to mm for convenience
        evap= self._evap * dt
        adjPET= evap * self._ke
        # condition 1: precipitation > PET
        cond= (precip>\
               adjPET)
#         precip[precip<adjPET]= adjPET[precip<adjPET]

        # First generate fast runoff
        precipSoil= np.zeros_like(precip)
        precipImperv= np.zeros_like(precip)
        precipSoil[cond]= (precip[cond] - adjPET[cond]) * (1-self._im[cond])
        precipImperv[cond]= precip[cond] - adjPET[cond] - precipSoil[cond]

        # infiltration
        interflowExcess= np.zeros_like(precip)
        interflowExcess[cond]= self._sm[cond] - self._wm[cond]
        interflowExcess[interflowExcess<0]= 0
        self._sm= np.where(self._sm>self._wm, self._wm, self._sm) # turns out to be faster for large array than simply reassigning
#         self._sm[self._sm>self._wm]= self._wm[self._sm>self._wm]
        cond1= (self._sm<self._wm) & (cond) #if soil content less than maximum capacity
        Wmaxm= np.zeros_like(precip)
        A= np.zeros_like(precip)
        Wmaxm[cond1]= self._wm[cond1] * (1+self._b[cond1])

        A[cond1]= Wmaxm[cond1] * (1.0-(1.0-self._sm[cond1]/self._wm[cond1])**(1.0/(1.0+self._b[cond1]))) #length of True cond

        R= np.zeros_like(self._sm)
        Wo= np.zeros_like(self._wm)
        infiltration= np.zeros_like(Wo, dtype=float)
        mask1= (precipSoil+A>=Wmaxm) & (cond1) #length of True cond
        R[mask1]= precipSoil[mask1] - (self._wm[mask1]- self._sm[mask1])
        R[R<0]=0.0
        Wo[mask1]= self._wm[mask1]

        mask2= (precipSoil+A<Wmaxm) & (cond1) #length of False cond
        infiltration[mask2]= self._wm[mask2] *((1-A[precipSoil+A<Wmaxm]/Wmaxm[precipSoil+A<Wmaxm])**(1+self._b[mask2])-(1-(A[precipSoil+A<Wmaxm]+precipSoil[mask2])/Wmaxm[precipSoil+A<Wmaxm])**(1+self._b[mask2]))
        infiltration[mask2]= np.where(infiltration[mask2]>precipSoil[mask2], precipSoil[mask2], infiltration[mask2])
        R[mask2]= precipSoil[mask2]- infiltration[mask2]
        R[R<0]=0
        Wo[mask2]= self._sm[mask2] + infiltration[mask2]
        
        cond2= (self._sm>=self._wm) & (cond)
        R[cond2]= precipSoil[cond2]
        Wo[cond2]= self._wm[cond2]
        
        temX= (self._sm+Wo)/self._wm/2*self._ksat_soil*dt
        recharge= np.zeros_like(temX)
        overland= np.zeros_like(temX)
        recharge[R<=temX]= R[R<=temX]
        recharge[R>temX]= temX[R>temX]
        
        actET= np.zeros_like(adjPET)
        actET[cond]= adjPET[cond]
        recharge[cond]+= interflowExcess[cond]
        overland[cond]= R[cond] - recharge[cond] + precipImperv[cond]

        #condition 2: precip<PET
        cond= (precip<adjPET)
        overland[cond]= 0.0
        interflowExcess[cond]= self._sm[cond] - self._wm[cond]
        interflowExcess[interflowExcess<0]=0
        recharge[cond]= interflowExcess[cond]
        self._sm[self._sm>self._wm]=self._wm[self._sm>self._wm]
        excessET= (adjPET - precip).copy()
        excessET[excessET<0]=0
        
        excessET_sf = (zsf - self._elev) - excessET #potential evaporation from surface water
        excessET_sf[excessET_sf<0]= 0
        excessET_sf[excessET_sf>excessET]= excessET[excessET_sf>excessET]
        self._grid['node']['surface_water__elevation']-= (excessET_sf/1000.) 
        
        excessET_sm= (excessET - excessET_sf) * (self._sm/self._wm) # potential evaporation from soil
        excessET_sm[excessET_sm>(excessET-excessET_sf)]= (excessET-excessET_sf)[excessET_sm>(excessET-excessET_sf)]
        Wo[(excessET_sm<self._sm) & cond]= self._sm[(excessET_sm<self._sm) & cond] - excessET_sm[(excessET_sm<self._sm) & cond] #evaporation from soil
        Wo[(excessET_sm>self._sm) & cond]=0.0
        excessET_gw= excessET- excessET_sf - excessET_sm #remaining evaporation from ground water
        excessET_gw[excessET_gw<0]=0
        excessET_gw[excessET_gw>(zgw-self._z_base)] = (zgw-self._z_base)[excessET_gw>(zgw-self._z_base)]
        self._grid['node']['ground_water__elevation']-= (excessET_gw/1000.)
        actET[precip<adjPET]= excessET_sf[precip<adjPET] +excessET_sm[precip<adjPET]+excessET_gw[precip<adjPET]+\
                              precip[precip<adjPET]
        
        #comment this to save time if you are confident with inner model balance
        self._check_water_balance(precip[self._grid.core_nodes],
                                  actET[self._grid.core_nodes],
                                  Wo[self._grid.core_nodes]-\
                                  self._sm[self._grid.core_nodes],
                                  overland[self._grid.core_nodes],
                                  recharge[self._grid.core_nodes],
                                  excessET_gw[self._grid.core_nodes]+\
                                  excessET_sf[self._grid.core_nodes])
        self._sm[:]= Wo
#         print(infiltration[self._grid.core_nodes], R[self._grid.core_nodes], Wo[self._grid.core_nodes])
        self._grid['node']['soil_moisture__content'][:]= self._sm

        self._router.rainfall_intensity= (overland/1000.)/dt
        self._gw.recharge= (recharge/1000.)/dt
#         print('zgw before:', self._grid['node']['ground_water__elevation'][self._grid.core_nodes])
        self._router.run_one_step(dt)
#         print('zsf after:', self._grid['node']['surface_water__elevation'][self._grid.core_nodes])        
        self._gw.run_with_adaptive_time_step_solver(dt)
#         print('zgw after:', self._grid['node']['ground_water__elevation'][self._grid.core_nodes])


    @property
    def sm(self):
        return self._sm

    @sm.setter
    def sm(self, factor):
        '''
        You can set initial soil moisture here
        '''
        self._sm= factor * self._wm

    @property
    def precip(self):
        return self._precip

    @precip.setter
    def precip(self, intensity):
        '''
        set precipitation rate as m/s
        '''
        if not isinstance(intensity,np.ndarray):
            msg= 'Expected numpy array, but got %s'%type(intensity)
            raise ValueError(msg)
        self._precip= intensity.reshape(self._grid.number_of_nodes).astype(float)
        self._precip[self._precip<0]=0

    @property
    def evap(self):
        return self._evap

    @precip.setter
    def evap(self, intensity):
        '''
        set precipitation rate as m/s
        '''
        if not isinstance(intensity,np.ndarray):
            msg= 'Expected numpy array, but got %s'%type(intensity)
            raise ValueError(msg)
        self._evap= intensity.reshape(self._grid.number_of_nodes).astype(float)
        self._evap[self._evap<0]=0
        
    @property
    def actET(self):
        '''
        return actual evaporation
        '''
        return self._actET
    
    @property
    def discharge(self):
        '''
        return outlet discharge
        '''
        return self._router.discharge_mapper(self._grid['node']['surface_water__discharge'], convert_to_volume=True)
        

    def _coerce_nan(self, field, to_value):
        pass
    
    #TODO
    def write_netcdf(self):
        pass
    
    def _check_water_balance(self, precip, actET, sm, overland, recharge, change):
        '''
        Attest water balance conservation for the overland and groundwater interface
        
        change = in - out
        
        here change means water level change, especially groundwater level
        '''
        balanced= precip - actET -sm - recharge-overland+change
        _isBalanced= np.isclose(balanced, 0, atol=1e-04)
        if not _isBalanced.all():
            raise WaterNotBalancedError('Water not balanced for cell deficit %s with precip %s, ET %s, sm %s, overland %s, recharge %s, and change %s'%(balanced[~_isBalanced], precip[~_isBalanced],actET[~_isBalanced],sm[~_isBalanced],overland[~_isBalanced],recharge[~_isBalanced],change[~_isBalanced]))

            
class WaterNotBalancedError(Exception):
    """Raise water balance error"""
    pass