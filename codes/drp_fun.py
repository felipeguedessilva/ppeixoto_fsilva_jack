#===========================================================================================
# Python Packges
#===========================================================================================
import numpy as np
import scipy as sp
import sympy as sym
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from functools import partial
from matplotlib import patheffects
from numpy import linalg as la
import segyio
import sys
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import RectBivariateSpline
#===========================================================================================

#===========================================================================================
# Devito Import
#===========================================================================================
from   devito                  import *
from   examples.seismic        import TimeAxis
from   examples.seismic        import RickerSource
from   examples.seismic        import Receiver
#===========================================================================================

#===========================================================================================
def generate_forn(M):
    
    x = [(1-(-1)**n*(2*n+1))//4 for n in range(2*M + 1)]
    N = 2
    weights  = sym.finite_diff_weights(N, x, 0)
    fornberg = np.array(weights[-1][-1][::2], dtype=np.float64)

    return fornberg
#===========================================================================================

#===========================================================================================
def critical_dt(weights,h,vmax):
    
    a = h*np.sqrt(2/np.sum([np.abs(a) for a in weights]))/vmax
    
    return float(a)
#===========================================================================================

#===========================================================================================
def critical_h(weights,dt,vmax):

    a = float(np.sqrt(2/np.sum([np.abs(a) for a in weights]))/(vmax*dt))
    b = 1/a
    
    return b
#===========================================================================================

#===========================================================================================
def acoustic(weightsx,weightsy,hx,hy,dt,vmodel,f,extentx,extenty,t1,factor=1,vtype=1):

    origin      = (0.,0.)
    grid_extent = (extentx, extenty)
    snx         = int(extentx/hx) #+ 1
    sny         = int(extenty/hy) #+ 1
    shape       = (snx, sny)
    grid         = Grid(shape=shape, extent=grid_extent)
    x, y         = grid.dimensions

    t0         = 0.0
    ntmax      = int((t1-t0)/dt)
    dt0        = (t1-t0)/(ntmax)
    time_range = TimeAxis(start=t0,stop=t1,num=ntmax+1)
    nt         = time_range.num - 1

    vmax = np.amax(vmodel)

    rvalue = dt0*vmax/min(hx,hy)

    weightsx    = np.concatenate([weightsx[::-1], weightsx[1:]])
    weightsy    = np.concatenate([weightsy[::-1], weightsy[1:]])
    space_order = len(weightsx) - 1

    velocity = Function(name="velocity", grid=grid, space_order=space_order)
    
    if(factor!=1):

        xoriginal        = np.linspace(0, extentx, vmodel.shape[0])
        yoriginal        = np.linspace(0, extenty, vmodel.shape[1])
        interp_spline    = RectBivariateSpline(xoriginal, yoriginal, vmodel)
        xx               = np.linspace(0, extentx, snx)
        yy               = np.linspace(0, extenty, sny)        
        vvelinterp       = interp_spline(xx, yy)
        velocity.data[:] = vvelinterp[:]

    else:

        velocity.data[:] = vmodel[:]
  
    sx = extentx/2

    if(vtype==1) : sy = extenty/2
    if(vtype==2) : sy = 2*hy
  
    src = RickerSource(name='src',grid=grid,f0=f,npoint=1,time_range=time_range,staggered=NODE,dtype=np.float64)
    src.coordinates.data[:, 0] = sx
    src.coordinates.data[:, 1] = sy

    nrecv = vmodel.shape[0]
    rx    = np.linspace(origin[0], grid_extent[0], nrecv)

    if(vtype==1): ry = (extentx/2)*np.ones(nrecv)
    if(vtype==2): ry = 2*hy*np.ones(nrecv)
    
    rec = Receiver(name='rec',grid=grid,npoint=nrecv,time_range=time_range,staggered=NODE,dtype=np.float64)
    rec.coordinates.data[:, 0] = rx
    rec.coordinates.data[:, 1] = ry

    u         = TimeFunction(name="u", grid=grid, time_order=2, space_order=space_order)
    pde       = (1/velocity**2)*u.dt2 - u.dx2(weights=weightsx) - u.dy2(weights=weightsy)
    stencil   = Eq(u.forward, solve(pde, u.forward))
    src_term  = src.inject(field=u.forward, expr=src*factor*factor*velocity*velocity*dt*dt)
    rec_term  = rec.interpolate(expr=u.forward)
    op        = Operator([stencil] + src_term + rec_term, subs=grid.spacing_map)
    
    op(time=nt,dt=dt0)    
    return u.data[-1], rec.data, rvalue
#===========================================================================================

#===========================================================================================
def critical_cfl(weights,h,dt,vmax):

    limit  = np.sqrt(2/np.sum([np.abs(a) for a in weights]))
    rvalue = (dt*vmax)/h

    print('Limit Value: %f'%limit)
    print('RVALUE: %f'%rvalue)
    
    if(limit>rvalue): 
        
        print('Stable Choice of weights,h,dt and vel!')
        print('')

    else:

        print('WARNING!"')
        print('Unstable Choice of weights, h,dt and vel!')
        print('')
        
    return
#===========================================================================================

#===========================================================================================
def objective(a):
    x = np.linspace(0, np.pi/2, 201)
    m = np.arange(1, len(a) + 1)
    y = x**2 + a[0] + 2*np.sum([a_ * np.cos(m_*x) for a_, m_ in zip(a[1:], m)], axis=0)
    return sp.integrate.trapezoid(y**2, x=x)
#===========================================================================================

#===========================================================================================
def dispersion_difference(weights,h,dt,v,k,alpha):
    if k == 0:
        diff = 0
    else:
        m = len(weights)
        cosines = np.array([
            np.cos(m*k*h*np.cos(alpha)) + np.cos(m*k*h*np.sin(alpha)) - 2
            for m in range(1, m)
        ])
        total = np.sum(np.array(weights)[1:]*cosines)
        theta = 1 + (v**2)*(dt**2)*total/(h**2)        
        diff = np.abs(np.acos(theta)/(k*dt) - v)
    return diff
#===========================================================================================

#===========================================================================================
def objective2(a, h, dt, vmin, vmax, fmax=100, alphamin=0, alphamax=np.pi/4, res=31):
    diff_wrapper = partial(dispersion_difference, weights=a, h=h, dt=dt)

    k_integral = np.zeros(res)
    v_space = np.linspace(vmin, vmax, res)
    alpha_space = np.linspace(alphamin, alphamax, res)
    for ii, v in enumerate(v_space):
        alpha_integral = np.zeros(res)
        k_space = np.linspace(0, 2*np.pi*fmax/v, res)
        for jj, k in enumerate(k_space):
            alpha_data = np.array([
                diff_wrapper(alpha=alpha, k=k, v=v) for alpha in alpha_space
            ])
            alpha_integral[jj] = np.trapezoid(alpha_data, alpha_space)
        k_integral[ii] = np.trapezoid(alpha_integral, k_space)
    v_integral = np.trapezoid(k_integral, v_space)

    return v_integral
#===========================================================================================

#===========================================================================================
def generate_drp1(M):

    fornberg      = generate_forn(M)
    initial_guess = fornberg
    
    constraints = [{
        'type': 'eq',
        'fun': lambda x: x[0] + 2*np.sum(x[1:])
    }]
    constraints += [{
        'type': 'eq',
        'fun': lambda x: np.sum([xi*m**2 for m, xi in enumerate(x)]) - 1
    }]
    constraints += [{
        'type': 'eq',
        'fun': lambda x: np.sum([xi*m**(2*jj) for m, xi in enumerate(x)])
    } for jj in range(2, (len(initial_guess) + 1)//2)]


    opt1 = sp.optimize.minimize(objective, initial_guess, method='SLSQP', constraints=constraints, options=dict(ftol=1e-15, maxiter=500))
    drp_stencil1 = opt1.x

    return drp_stencil1
#===========================================================================================

#===========================================================================================
def generate_drp2(M,h,dt,vmin,vmax):

    fornberg           = generate_forn(M)    
    initial_guess      = fornberg

    constraints = [{
        'type': 'eq',
        'fun': lambda x: x[0] + 2*np.sum(x[1:])
    }]
    constraints += [{
        'type': 'eq',
        'fun': lambda x: np.sum([xi*m**2 for m, xi in enumerate(x)]) - 1
    }]
    constraints += [{
        'type': 'eq',
        'fun': lambda x: np.sum([xi*m**(2*jj) for m, xi in enumerate(x)])
    } for jj in range(2, (len(initial_guess) + 1)//2)]


    objective2_wrapper = partial(objective2,h=h,dt=dt,vmin=vmin,vmax=vmax)
    opt2               = sp.optimize.minimize(objective2_wrapper, initial_guess, constraints=constraints, method='SLSQP')
    drp_stencil2       = opt2.x

    return drp_stencil2
#===========================================================================================