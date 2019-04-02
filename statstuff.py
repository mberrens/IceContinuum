# Code for roughness statistics
import numpy as np
import copy
import imagestuff as ims
from scipy.interpolate import griddata
#import importlib; importlib.reload(ims)



def pWeibull(r, sigma, eta):
    ''' Weibull function '''
    from numpy import exp
    mu = 1-r
    ret = 2*eta/sigma**2/mu**3 * \
        (((mu**(-2)-1)/sigma**2)**(eta-1)) * \
        exp(-((mu**(-2)-1)/sigma**2)**eta)
    return ret

def pWeibullr(r, sigma, eta):
    ''' Weibull function times r '''
    return pWeibull(r, sigma, eta)*r

def pGaussian(r, sigma):
    ''' Gaussian function '''
    return pWeibull(r, sigma, 1)    

def pGaussianr(r, sigma):
    ''' Gaussian function times r '''
    return pWeibullr(r, sigma, 1)

def bimodal(r, sigma1, sigma2, N):
    ''' Bimodal Gaussian function '''
    pdf1 = pWeibull(r,sigma1,1.0)
    pdf2 = pWeibull(r,sigma2,1.0)
    return (1-N)*pdf1 + N*pdf2 

def bimodalr(r, sigma1, sigma2, N):
    ''' Bimodal Gaussian function times r'''
    pdf1 = pWeibullr(r,sigma1,1.0)
    pdf2 = pWeibullr(r,sigma2,1.0)
    return (1-N)*pdf1 + N*pdf2 


def bimodalfunc(r, sigma1, sigma2, N):
    ''' Bimodal Gaussian function '''
    pdf1 = pWeibullr(r,sigma1,1.0)
    pdf2 = pWeibullr(r,sigma2,1.0)
    return (1-N)*pdf1 + N*pdf2 


def sigma2meanr(sigma):
    ''' Converting sigma to <r> 
        Usage: 
        
        sigmalist = np.linspace(.01,.9)
        meanr = sigma2meanr(sigmalist)
        plt.figure()
        plt.plot(sigmalist,meanr,'o')
        plt.grid(True)        
    '''
    p = np.array([ 4.57899291e-01, -2.27236062e+00,  4.72080621e+00, -5.09338608e+00,
        2.57626515e+00,  1.77811151e-01, -8.38705493e-01,  1.49765369e-02,
        4.98822342e-01,  3.87112620e-05, -3.41914362e-07])
    meanr = np.polyval(p,sigma)
    return meanr

def R_squar(y,yfit):
    SS_res = np.sum((y-yfit)**2)
    SS_tot = np.sum((y-np.mean(y))**2)
    return 1-SS_res/SS_tot

def makehistogram(\
                  nsegments,nx1list,nx2list,ny1list,ny2list,dx,dy,solution,\
                  accumlist, newrbins):

    # Setting up null arrays
    hbins_accum = []
    meanrsub_accum = []
    zsigma_accum = []
    Z2_accum = []
    Zsquared_accum = []
    rsub_accum = []
    meanrsublist = []
    Zsigmalist = []
    Z2list = []

    # Now, to evaluate the roughness ... First step is to flatten each panel via rotation
    # Here we explicitly flip the y-coordinate (to make it a right-handed system) so we don't have to invert on the fly

    for isegment in range(0,nsegments):

        # Extract this segment
        nx1=nx1list[isegment]; nx2=nx2list[isegment]; nxsegment = nx2-nx1+1
        ny1=ny1list[isegment]; ny2=ny2list[isegment]; nysegment = ny2-ny1+1
        surf_xseg = np.linspace(0,(nxsegment-1)*dx,nxsegment); 
        surf_yseg = np.linspace(0,(nysegment-1)*dy,nysegment); 
        surf_xseggrid, surf_yseggrid = np.meshgrid(surf_xseg,surf_yseg) # 1st index is y, 2nd is x
        surf_zseggrid = copy.copy(np.flipud(solution[ny1:ny2+1,nx1:nx2+1])) # This flips the y-coordinate

        # Fit a plane to the data and adjust data to start at the origin
        m = ims.polyfit2d(\
                      surf_xseggrid.reshape(nysegment*nxsegment), \
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
        mp = ims.polyfit2d(\
                      surf_xseggridp.reshape(nysegment*nxsegment), \
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
        surf_zseggrid_theory_long = ims.polyval2d(\
                      surf_xseggrid.reshape(nysegment*nxsegment), \
                      surf_yseggrid.reshape(nysegment*nxsegment), \
                      m)
        surf_zseggrid_theory = surf_zseggrid_theory_long.reshape(nysegment,nxsegment)
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

        # Now we get the roughness
        dzsub_dx = np.diff(sub_zseggrid,axis=1)/np.diff(sub_xseggrid,axis=1)
        dzsub_dy = np.diff(sub_zseggrid,axis=0)/np.diff(sub_yseggrid,axis=0)
        Zsquared = dzsub_dx[1:,:]**2+dzsub_dy[:,1:]**2
        rsub = 1.0 - 1/np.sqrt(1+Zsquared)
        mu = 1-rsub
        phi = np.arccos(mu)
        Zplus = Zsquared**.5
        Z = np.hstack((Zplus,-Zplus)) # Need +/- to generate a two-sided distribution
        thismeanrsub = np.round(np.mean(rsub)*1000)/1000; meanrsublist.append(thismeanrsub)
        thissigma = np.round(np.std(Z)*100)/100; Zsigmalist.append(thissigma)
        thismeanZ2 = np.mean(Zsquared); Z2list.append(thismeanZ2)

        # Numerical distribution functions
        rsub_long = np.reshape(rsub,np.size(rsub))
        hist = np.histogram(rsub_long,bins=newrbins)
        rbins = hist[1][0:-1]
        rbins1 = hist[1][1:]
        hbins = hist[0] 

        # Normalizing ... this might be wrong
        norm = np.trapz(hbins,rbins)
        hbins = hbins/norm

        # Defining the analytical distribution function bins
        rwidth = rbins1-rbins
        rbinsW = (rbins+rwidth/2.0)        

        # Accumulate the binned data
        if isegment in accumlist:
            hbins_accum.append(hbins)
            print ('Accumulating ...', isegment+1, 'out of', len(accumlist))

    # Combine the histograms of individual segments
    hbins_total = np.sum((hbins_accum),axis=0)/len(accumlist)
    norm = np.trapz(hbins_total,np.log(rbinsW)); print('Norm =', norm)
    hbins_total = hbins_total/norm

    # Get out
    return hbins_total, rbinsW


def makehistogram_heights(\
                  nsegments,nx1list,nx2list,ny1list,ny2list,dx,dy,solution,\
                  accumlist, newzbins):

    # Setting up null arrays
    hbins_accum = []
    z_accum = []
    zlist = []

    # Looping over all the segments
    for isegment in range(0,nsegments):

        # Extract this segment
        nx1=nx1list[isegment]; nx2=nx2list[isegment]; nxsegment = nx2-nx1+1
        ny1=ny1list[isegment]; ny2=ny2list[isegment]; nysegment = ny2-ny1+1
        surf_xseg = np.linspace(0,(nxsegment-1)*dx,nxsegment); 
        surf_yseg = np.linspace(0,(nysegment-1)*dy,nysegment); 
        surf_xseggrid, surf_yseggrid = np.meshgrid(surf_xseg,surf_yseg) # 1st index is y, 2nd is x
        surf_zseggrid = copy.copy(np.flipud(solution[ny1:ny2+1,nx1:nx2+1])) # This flips the y-coordinate

        # Fit a plane to the data and adjust data to start at the origin
        m = ims.polyfit2d(\
                      surf_xseggrid.reshape(nysegment*nxsegment), \
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
        mp = ims.polyfit2d(\
                      surf_xseggridp.reshape(nysegment*nxsegment), \
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
        surf_zseggrid_theory_long = ims.polyval2d(\
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
        
        # Numerical distribution functions
        hist = np.histogram(sub_zseggrid_long-zreference,bins=newzbins)
        zbins = hist[1][0:-1]
        zbins1 = hist[1][1:]
        hbins = hist[0] 
        norm = np.trapz(hbins,zbins)
        hbins = hbins/norm

        # Defining the analytical distribution function bins
        zwidth = zbins1-zbins
        zbinsW = (zbins+zwidth/2.0)        

        # Accumulate the binned data
        if isegment in accumlist:
            hbins_accum.append(hbins)
            print ('Accumulating ...', isegment+1, 'out of', len(accumlist))

    # Combine the histograms of individual segments
    hbins_total = np.sum((hbins_accum),axis=0)/len(accumlist)
    norm = np.trapz(hbins_total,zbinsW); print('Norm =', norm)
    hbins_total = hbins_total/norm

    # Get out
    return hbins_total, zbinsW


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
