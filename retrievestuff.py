

import numpy as np
import copy
import time
import imagestuff as ims
from scipy.interpolate import griddata
#import importlib; importlib.reload(ims)


def get_heights(nsegments,nx1list,nx2list,ny1list,ny2list,dx,dy,solution,isegment):

        # Extract this segment
        nx1=nx1list[isegment]; nx2=nx2list[isegment]; nxsegment = nx2-nx1+1
        ny1=ny1list[isegment]; ny2=ny2list[isegment]; nysegment = ny2-ny1+1
        surf_xseg = np.linspace(0,(nxsegment-1)*dx,nxsegment); 
        surf_yseg = np.linspace(0,(nysegment-1)*dy,nysegment); 
        surf_xseggrid, surf_yseggrid = np.meshgrid(surf_xseg,surf_yseg) # 1st index is y, 2nd is x
        surf_zseggrid = copy.copy(np.flipud(solution[ny1:ny2+1,nx1:nx2+1])) # This flips the y-coordinate

        # Fit a plane to the data and adjust data to start at the origin
        m = ims.polyfit2d(surf_xseggrid.reshape(nysegment*nxsegment), \
                          surf_yseggrid.reshape(nysegment*nxsegment), \
                          surf_zseggrid.reshape(nysegment*nxsegment), \
                          linear=True,order=1)

        # Get the angles of the plane
        dzdy = m[1]; thetay = np.arctan(dzdy)*180/np.pi; #print 'y:', thetay

        # Get rotation matrix & flatten in one direction
        Roty = ims.myrotation_matrix([1,0,0], -thetay)
        surf_xseggridp, surf_yseggridp, surf_zseggridp = \
            ims.flatten(surf_xseggrid, surf_yseggrid, surf_zseggrid, Roty)

        # Fit a plane to the data and adjust data to start at the origin
        mp = ims.polyfit2d(surf_xseggridp.reshape(nysegment*nxsegment), \
                           surf_yseggridp.reshape(nysegment*nxsegment), \
                           surf_zseggridp.reshape(nysegment*nxsegment), \
                           linear=True,order=1)

        # Get the angle of the plane in another direction
        dzdx = mp[2]; thetaxp = np.arctan(dzdx)*180/np.pi; #print 'x:', thetaxp

        # Get rotation matrix & flatten in another direction
        Rotxp = ims.myrotation_matrix([0,1,0], thetaxp)
        surf_xseggridpp, surf_yseggridpp, surf_zseggridpp = \
            ims.flatten(surf_xseggridp, surf_yseggridp, surf_zseggridp, Rotxp)


        # Trying out the polyval2d
        surf_zseggrid_theory_long = ims.polyval2d( \
                                        surf_xseggrid.reshape(nysegment*nxsegment), \
                                        surf_yseggrid.reshape(nysegment*nxsegment), \
                                        m)
        surf_zseggrid_theory = surf_zseggrid_theory_long.reshape(nysegment,nxsegment)
        #surf_zseggrid_theory -= z0
        surf_xseggridp_theory, surf_yseggridp_theory, surf_zseggridp_theory = \
            ims.flatten(surf_xseggrid, surf_yseggrid, surf_zseggrid_theory, Roty)
        surf_xseggridpp_theory, surf_yseggridpp_theory, surf_zseggridpp_theory = \
            ims.flatten(surf_xseggridp_theory, surf_yseggridp_theory, surf_zseggridp_theory, Rotxp)

        # Now rotate
        deltay = surf_yseggridpp_theory[0,-1]-surf_yseggridpp_theory[0,0]
        deltax = surf_xseggridpp_theory[0,-1]-surf_xseggridpp_theory[0,0]
        thetazpp = -np.arctan(deltay/deltax)*180/np.pi;
        Rotzpp = ims.myrotation_matrix([0,0,1], thetazpp)
        surf_xseggridppp, surf_yseggridppp, surf_zseggridppp = \
            ims.flatten(surf_xseggridpp, surf_yseggridpp, surf_zseggridpp, Rotzpp)
        surf_xseggridppp_theory, surf_yseggridppp_theory, surf_zseggridppp_theory = \
            ims.flatten(surf_xseggridpp_theory, surf_yseggridpp_theory, surf_zseggridpp_theory, Rotzpp)

        # Now we have to extract an orthogonal subset
        dxsub = dysub = dx
        xsubstart = np.max(surf_xseggridppp_theory[[0,-1],0])+dxsub*2
        xsubstop = np.min(surf_xseggridppp_theory[[0,-1],-1])-dxsub*2
        ysubstart = np.max(surf_yseggridppp_theory[0,[0,-1]])+dysub*2
        ysubstop = np.min(surf_yseggridppp_theory[-1,[0,-1]])-dysub*2
        xsub = np.arange(xsubstart,xsubstop,dxsub)
        ysub = np.arange(ysubstart,ysubstop,dysub)
        sub_xseggrid, sub_yseggrid = np.meshgrid(xsub,ysub) # 1st index is y, 2nd is x
        nsuby, nsubx = np.shape(sub_xseggrid)
        surf_xseggridppp_theory_long = np.reshape(surf_xseggridppp_theory,nysegment*nxsegment)
        surf_yseggridppp_theory_long = np.reshape(surf_yseggridppp_theory,nysegment*nxsegment)
        points = np.vstack((surf_xseggridppp_theory_long,surf_yseggridppp_theory_long)).T # rows are x,y pairs
        values = np.reshape(surf_zseggridppp,nysegment*nxsegment)
        sub_zseggrid_long = griddata(points, values, (sub_xseggrid, sub_yseggrid), method='cubic')
        sub_zseggrid = np.reshape(sub_zseggrid_long,(nsuby, nsubx))

        # Now we get the heights relative to a reference
        zreference = np.median(sub_zseggrid)
        
#         # Accumulate the binned data
#         if isegment in accumlist:
            

        # Get out
        return sub_zseggrid

    
def getrhoofz2(sollast_in,dx,dy,nbins=10,transposeflag=False):
    
    # Transpose, if flagged
    if transposeflag:
        sollast = sollast_in.T
    else:
        sollast = sollast_in
    
    # Dimensions 
    Nx, Ny = np.shape(sollast)
    
    # Calculate the gradient squared (Z2)
    dzdx = np.diff(sollast, axis=0)/dx
    dzdy = np.diff(sollast, axis = 1)/dy #we are not sure which axis is which
    Z2 = dzdx[:, 1:]**2+dzdy[1:, :]**2
    
    # Get the probability distribution
    Z2flat = np.reshape(Z2, (Nx-1)*(Ny-1))
    counts, bins = np.histogram(Z2flat,bins=nbins)
    counts = counts/np.sum(counts)
    newbins = bins[1:]
#     subset = np.array([i for i in range(4,len(bins))])-1
#     logcounts = np.log(counts[subset])

    #plt.semilogy(newbins, counts, 'o', label='Numerical result')
    return counts, newbins
