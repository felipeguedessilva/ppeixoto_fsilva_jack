#===========================================================================================
# Python Packges
#===========================================================================================
import numpy             as np
import scipy             as sp
import sympy             as sym
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import segyio
import sys
from functools import partial
from matplotlib import patheffects
from numpy import linalg as la
from scipy.interpolate import RectBivariateSpline
#===========================================================================================

#===========================================================================================
# Local Imports
#===========================================================================================
import drp_fun as drpfun
#===========================================================================================

#===========================================================================================
# Devito Imports
#===========================================================================================
from devito import Grid, Function, TimeFunction, SparseTimeFunction, Eq, Operator, solve
#===========================================================================================

#===========================================================================================
# Global Configs
#===========================================================================================
normtype  = np.inf
vmodel    = 1

if(vmodel==1):

    f0        = 0.02
    npoints   = 101
    extent    = 1000
    h         = extent/(npoints-1)
    t1        = 300
    vvel      = 1.5*np.ones((npoints,npoints))


if(vmodel==2):
    vvel      = np.load("../velmodels/salt.npy")
    hx        = 10
    hy        = 15
    f0        = 0.02
    npointsx  = vvel.shape[0]
    npointsy  = vvel.shape[1]
    extentx   = hx*vvel.shape[0]
    extenty   = hy*vvel.shape[1]
    t1        = 1200

vmax      = np.amax(vvel)
vmin      = np.amin(vvel)
#===========================================================================================

#===========================================================================================
# Simple Test
#===========================================================================================
if(vmodel==1):

    M                       = 20
    fornbergteste           = drpfun.generate_forn(M)
    dtinitial               = drpfun.critical_dt(fornbergteste,h,vmax)
    dtnew                   = 0.5*dtinitial
    drpfun.critical_cfl(fornbergteste,1*h,dtnew,vmax)
    factor                  = 1
    hteste                  = factor*h
    uteste,datateste,rteste = drpfun.acoustic(fornbergteste,fornbergteste,hteste,hteste,dtnew,vvel,f0,extent,extent,t1,factor=factor)

if(vmodel==2):

    M                       = 20
    fornbergteste           = drpfun.generate_forn(M)
    h                       = min(hx,hy)
    dtinitial               = drpfun.critical_dt(fornbergteste,h,vmax)
    dtnew                   = 0.5*dtinitial
    drpfun.critical_cfl(fornbergteste,h,dtnew,vmax)
    factor                  = 1
    uteste,datateste,rteste = drpfun.acoustic(fornbergteste,fornbergteste,factor*hx,factor*hy,dtnew,vvel,f0,extentx,extenty,t1,factor=factor,vtype=2)
#===========================================================================================

#===========================================================================================
# Run Setup
#===========================================================================================
if(vmodel==1):

    vh       = np.linspace(0.1*h,h,10)
    nvh      = vh.shape[0]

if(vmodel==2):

    vhx      = np.linspace(0.1*hx,hx,10)
    vhy      = np.linspace(0.1*hy,hy,10)
    nvh      = vhx.shape[0]

morder   = 8
compfact = 5
#===========================================================================================

#===========================================================================================
# Fornberg
#===========================================================================================

#===========================================================================================
lnormuf   = []

for k1 in range(0,nvh):
    
    try:
        
        M           = morder
        fornberg    = drpfun.generate_forn(M)
        
        if(vmodel==1): 
            
            hloc        = vh[k1] 
            factor      = h/hloc
            uf,dataf,rf = drpfun.acoustic(fornberg,fornberg,hloc,hloc,dtnew,vvel,f0,extent,extent,t1,factor=factor)
        
        if(vmodel==2): 
            
            hlocx       = vhx[k1] 
            hlocy       = vhy[k1] 
            factor      = h/hlocx
            uf,dataf,rf = drpfun.acoustic(fornberg,fornberg,hlocx,hlocy,dtnew,vvel,f0,extentx,extenty,t1,factor=factor,vtype=2)

        lnormu = la.norm(uf,normtype)
        lnormuf.append(lnormu)

    except:
        
            lnormuf.append(np.nan)    
#===========================================================================================

#===========================================================================================
lnormuf1 = lnormuf
for k1 in range(0,nvh-1):

    if(lnormuf1[k1]>compfact*lnormuf1[-1]): lnormuf1[k1] = np.nan
#===========================================================================================

#===========================================================================================
plt.figure(figsize = (5,4))
if(vmodel==1): plt.plot(vh,lnormuf,label='Fornberg')
if(vmodel==2): plt.plot(vhx,lnormuf,label='Fornberg')
plt.grid()
plt.legend()
plt.title('Error for h - Fornberg - Displacement')
plt.xlabel('[h]')
plt.ylabel('[Error]')
plt.savefig('figures/fornberg_results.jpeg',dpi=200,bbox_inches='tight')
plt.close()
#===========================================================================================

#===========================================================================================
#DRPS 1
#===========================================================================================

#===========================================================================================
lnormudrp1   = []

for k1 in range(0,nvh):

    try:
        
        M            = morder
        drp_stencil1 = drpfun.generate_drp1(M)
        
        if(vmodel==1): 

            hloc         = vh[k1] 
            factor       = h/hloc
            uf,dataf,rf  = drpfun.acoustic(drp_stencil1,drp_stencil1,hloc,hloc,dtnew,vvel,f0,extent,extent,t1,factor=factor)
        
        if(vmodel==2): 

            hlocx       = vhx[k1] 
            hlocy       = vhy[k1] 
            factor      = h/hlocx
            uf,dataf,rf = drpfun.acoustic(drp_stencil1,drp_stencil1,hlocx,hlocy,dtnew,vvel,f0,extentx,extenty,t1,factor=factor,vtype=2)

        lnormu = la.norm(uf,normtype)
        lnormudrp1.append(lnormu)
    
    except:
    
        lnormudrp1.append(np.nan)
#===========================================================================================

#===========================================================================================
lnormudrp11 = lnormudrp1
for k1 in range(0,nvh-1):

    if(lnormudrp11[k1]>compfact*lnormudrp11[-1]): lnormudrp11[k1] = np.nan
#===========================================================================================

#===========================================================================================
plt.figure(figsize = (5,4))
if(vmodel==1): plt.plot(vh,lnormudrp11,label='DRP1')
if(vmodel==2): plt.plot(vhx,lnormudrp11,label='DRP1')
plt.grid()
plt.legend()
plt.title('Error for h - DRP1 - Displacement')
plt.xlabel('[h]')
plt.ylabel('[Error]')
plt.savefig('figures/drps1_results.jpeg',dpi=200,bbox_inches='tight')
plt.close()
#===========================================================================================

#===========================================================================================
# DRPS2
#===========================================================================================
lnormudrp2   = []

for k1 in range(0,nvh):

    try:
        
        M = morder

        if(vmodel==1): 

            hloc         = vh[k1] 
            drp_stencil2 = drpfun.generate_drp2(M,hloc,dtnew,vmin,vmax)
            factor       = h/hloc
            uf,dataf,rf  = drpfun.acoustic(drp_stencil2,drp_stencil2,hloc,hloc,dtnew,vvel,f0,extent,extent,t1,factor=factor)
        
        if(vmodel==2): 

            hlocx        = vhx[k1] 
            hlocy        = vhy[k1]
            hlocmin      = min(vhx[k1],vhy[k1]) 
            drp_stencil2 = drpfun.generate_drp2(M,hlocmin,dtnew,vmin,vmax)
            factor       = h/hlocx
            uf,dataf,rf  = drpfun.acoustic(drp_stencil2,drp_stencil2,hlocx,hlocy,dtnew,vvel,f0,extentx,extenty,t1,factor=factor,vtype=2)

        lnormu = la.norm(uf,normtype)   
        lnormudrp2.append(lnormu)
    
    except:
    
        lnormudrp2.append(np.nan)
#===========================================================================================

#===========================================================================================
lnormudrp21 = lnormudrp2
for k1 in range(0,nvh-1):

    if(lnormudrp21[k1]>compfact*lnormudrp21[-1]): lnormudrp21[k1] = np.nan
#===========================================================================================

#===========================================================================================
plt.figure(figsize = (5,4))
if(vmodel==1): plt.plot(vh,lnormudrp21,label='DRP2')
if(vmodel==2): plt.plot(vhx,lnormudrp21,label='DRP2')
plt.grid()
plt.legend()
plt.title('Error for h - DRP2 - Displacement')
plt.xlabel('[h]')
plt.ylabel('[Error]')
plt.savefig('figures/drps2_results.jpeg',dpi=200,bbox_inches='tight')
plt.close()
#===========================================================================================

#===========================================================================================
# Comparison Results
#===========================================================================================

#===========================================================================================
plt.figure(figsize = (5,4))
plt.plot(vhx,lnormuf,label='Fornberg')
plt.plot(vhx,lnormudrp1,label='DRP1')
plt.plot(vhx,lnormudrp2,label='DRP2')
plt.grid()
plt.legend()
plt.title('Error for h - Displacement')
plt.xlabel('[h]')
plt.ylabel('[Error]')
plt.yscale('log')
plt.savefig('figures/comp_full.jpeg',dpi=200,bbox_inches='tight')
plt.close()
#===========================================================================================

#===========================================================================================
plt.figure(figsize = (5,4))
plt.plot(vhx,lnormuf,label='Fornberg')
plt.plot(vhx,lnormudrp2,label='DRP2')
plt.grid()
plt.legend()
plt.title('Erro for h - Displacement')
plt.xlabel('[h]')
plt.ylabel('[Error]')
plt.yscale('log')
plt.savefig('figures/comp_forn_drp2.jpeg',dpi=200,bbox_inches='tight')
plt.close()
#===========================================================================================

#===========================================================================================
plt.figure(figsize = (5,4))
plt.plot(vhx,lnormuf,label='Fornberg')
plt.plot(vhx,lnormudrp1,label='DRP1')
plt.grid()
plt.legend()
plt.title('Erro for h - Displacement')
plt.xlabel('[h]')
plt.ylabel('[Error]')
plt.yscale('log')
plt.savefig('figures/comp_forn_drp1.jpeg',dpi=200,bbox_inches='tight')
plt.close()
#===========================================================================================

#===========================================================================================
plt.figure(figsize = (5,4))
plt.plot(vhx,lnormudrp1,label='DRP1')
plt.plot(vhx,lnormudrp2,label='DRP2')
plt.grid()
plt.legend()
plt.title('Erro for h - Displacement')
plt.xlabel('[h]')
plt.ylabel('[Error]')
plt.yscale('log')
plt.savefig('figures/comp_drp1_drp2.jpeg',dpi=200,bbox_inches='tight')
plt.close()
#===========================================================================================