"""
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
"""
import numpy as np
import cython

cimport numpy as np

from scipy.optimize.cython_optimize cimport brentq

DTYPE=np.float32

ctypedef struct params:
    double a
    double b
    double c
    double d
    double e

cdef double update_f(double x, void *args):
    """Evaluates the solution to the water-depth equation.

    Called by scipy.newton() to find solution for :math:`x`
    using Newton's method.

    Parameters
    ----------
    x : float
        Water depth at new time step.
    a : float
        "alpha" parameter (see below)
    b : float
        Weighting factor on new versus old time step. :math:`b=1` means purely
        implicit solution with all weight on :math:`H` at new time
        step. :math:`b=0` (not recommended) would mean purely explicit.
    c : float
        Water depth at old time step (time step :math:`t` instead
        of :math:`t+1`)
    d : float
        Depth-discharge exponent; normally either 5/3 (Manning) or 3/2 (Chezy)
    e : float
        Water inflow volume per unit cell area in one time step.

    This equation represents the implicit solution for water depth
    :math:`H` at the next time step. In the code below, it is
    formulated in a generic way.  Written using more familiar
    terminology, the equation is:

    math::

        H - H_0 + \alpha ( w H + (w-1) H_0)^d - \Delta t (R + Q_{in} / A)

    math::

        \alpha = \frac{\Delta t \sum S^{1/2}}{C_f A}

    where :math:`H` is water depth at the given node at the new
    time step, :math:`H_0` is water depth at the prior time step,
    :math:`w` is a weighting factor, :math:`d` is the depth-discharge
    exponent (2/3 or 1/2), :math:`\Delta t` is time-step duration,
    :math:`R` is local runoff rate, :math:`Q_{in}` is inflow
    discharge, :math:`A` is cell area, :math:`C_f` is a
    dimensional roughness coefficient, and :math:`\sum S^{1/2}`
    represents the sum of square-root-of-downhill-gradient over
    all outgoing (downhill) links.
    """
    cdef params *myargs= <params *> args

    return x - myargs.c + myargs.a * (myargs.b+x+(myargs.b-1.0)*myargs.c)**myargs.d-myargs.e

def routing(np.ndarray nodes_ordered,
            np.ndarray status_at_node,
            np.ndarray runoff_rate,
            np.ndarray disch_in,
            np.ndarray grad_width_sum,
            np.ndarray area_of_node,
            np.ndarray adjacent_nodes_at_node,
            np.ndarray proportions,
            np.ndarray depth,
            np.ndarray alpha,
            float dt,
            float weight,
            float depth_exp,
            float vel_coef):
    """
    Description goes here
    """

    cdef int n, i
    cdef float aa,cc,ee, Heff
    cdef np.ndarray outflow=np.zeros_like(depth, dtype=np.float32)
    cdef params myargs

    for i in range(len(nodes_ordered)-1,-1,-1):
        n= nodes_ordered[i]
        if status_at_node[i]==0:
            aa= alpha[n]
            cc= depth[n]
            ee= dt*runoff_rate[n] + (dt*disch_in[n]/area_of_node[n])
            myargs= {'a':aa,'b':weight,'c':cc,'d':depth_exp,'e':ee}
            depth[n]= brentq(update_f, max(0,depth[n]-2), depth[n]+2, <params *> &myargs, 1e-3, 1e-3, 100, NULL)
            Heff= weight * depth[n] + (1-weight) *cc
            outflow[n]= (vel_coef*Heff**depth_exp)*grad_width_sum[n]
            disch_in[adjacent_nodes_at_node[n]] += outflow[n] * proportions[n]

    return depth, disch_in, outflow
