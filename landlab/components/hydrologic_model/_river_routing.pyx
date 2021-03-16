'''
This module implements vector river routing at reach level

Input:
-----------------------
1D arrays of:
-connectivity: flow from upstream to downstream order
-strahler order: river order | researve for future parallelization
-reach slope
-reach manning coeficient
-reach width

Output:
-----------------------
discharge at node grid

__author__: Allen (Zhi) Li
__date__: 03/10/2021
'''
import numpy as np

cimport numpy as np

DTYPE=np.float32

def _river_routing( np.ndarray zrv_bottom,
                    np.ndarray zrv,
                    np.ndarray Qsurf_to_riv,
                    np.ndarray Qsub_to_riv,
                    np.ndarray stage,
                    np.ndarray width,
                    np.ndarray slope,
                    np.ndarray manning_n
                    ):
    cross_section_area= (stage-bottom_elevation)*width
    wet_perimeter= (stage-bottom_elevation)*2+width
    downQ= cross_section_area/manning_n*(cross_section_area/wet_perimeter)**2/3*slope**0.5
    pass

