

# .. Built-in modules
import pickle
import numpy as np
import scipy.io as spio
from netCDF4 import Dataset, num2date, date2index
from scipy.io.netcdf import NetCDFFile as DS
from scipy.interpolate import interp2d
import array



def load_ssp_nu(datafile, nu):
    
    # Load the moments from a netcdf file & return the unbundled arrays
    dnu = nu[1] - nu[0]
    with DS(datafile,'r') as nc:
    
        inu = np.where(np.logical_and(\
                              nc.variables['wnum_list'][:] > nu[0]-2*dnu, 
                              nc.variables['wnum_list'][:] < nu[-1]+2*dnu))[0]
        Npmomarray = nc.variables['Npmomarray'][:,inu].astype('int32')
        w0_mesh    = nc.variables['w0_mesh'][:,inu].astype('float64')
        qext_mesh  = nc.variables['qext_mesh'][:,inu].astype('float64')
        reff = nc.variables['reff_list'][:].astype('float64'); reff = reff[:,0]
        wnum_vec  = nc.variables['wnum_list'][inu].astype('float64')
        # wnum_mesh  = nc.variables['wnum_mesh'][:,inu].astype('float64')
        # reff_mesh  = nc.variables['reff_mesh'][:,inu].astype('float64')
        
        
        # Set up empty output arrays
        Nnu = nu.size
        Nreff = reff.size
        qext = np.zeros((Nreff, Nnu))
        w0 = np.zeros((Nreff, Nnu))
        NPmom_fp = np.zeros((Nreff, Nnu))
    
        # Interpolate qext, w0, get an interpolated number of moments!
        fq = interp2d(reff, wnum_vec, qext_mesh.T)
        fw = interp2d(reff, wnum_vec, w0_mesh.T)
        fNP = interp2d(reff, wnum_vec, Npmomarray.T)
    
        for i in range(Nreff):
            qext[i,:] = fq(reff[i], nu)[:,0]
            w0[i,:] = fw(reff[i], nu)[:,0]
            NPmom_fp[i,:] = fNP(reff[i], nu)[:,0]
        
        # Use floor so we never interpolate between a moment and 0.
        NPmom = np.floor(NPmom_fp).astype(int) 
        NPmom_max = np.max(NPmom)
        pmomarray  = nc.variables['pmomarray'][:,inu,:NPmom_max]
        pmomarray = pmomarray.astype('float64')
        
        # Loop over all the moments to do the same
        Pmom = np.zeros((Nreff, Nnu, NPmom_max));
        for j in range( NPmom_max):
            f = interp2d(reff, wnum_vec, pmomarray[:,:,j].T)
            for i in range(Nreff):
                Pmom[i,:,j] = f(reff[i], nu)[:,0]

    return (NPmom, Pmom, reff, w0, qext)


def getsolarbeam_IR (wnum=None, solarbeam_IR_file=None):

    #print solarbeam_IR_file
    
    kurucz = np.loadtxt(solarbeam_IR_file) #'kurucz.dat')
    beam = np.interp(wnum,kurucz[:,0],kurucz[:,1])/1000;
    return (beam)



# # # # # # # #      LOAD SURFACE ALBEDO    # # # # # # # # # # # #  # # 
def get_surface_albedo_from_file(surfEmissDataFile):

    albedoData = np.loadtxt(surfEmissDataFile, comments='%')
    nu_surf_albedo = albedoData[:, 1]
    surf_albedo = 1-albedoData[:, 2]
    
    return nu_surf_albedo, surf_albedo


def get_surface_albedo_IR(wnum = None, surfEmissDataFile = None):

    Mammoth = np.loadtxt(surfEmissDataFile) #'Mammoth.dat')
    emissivity = np.interp(wnum, Mammoth[:,1], Mammoth[:,2])
    emissivity[emissivity>1.] = 1.
    emissivity[emissivity<0.] = 0.
    #for i in range(len(emissivity)):
    #    if emissivity[i]>1.:
    #        emissivity[i] = 1.
    #    elif emissivity[i]<0.:
    #        emissivity[i] = 0.
    #emissivity = min(interp1(Mammoth(:,2),Mammoth(:,3),wnum,'linear','extrap'),1);
    #emissivity = max(emissivity,0);
    surface_albedo = 1. - emissivity;
    return surface_albedo

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 



def load_od_gas(odfile):
    '''
    Purpose:
      Load in the matlab generated file because python is
      too slow for cubic interp and gives small differences
      But the python way is saved for reference in extras.py
    '''
    
    odinfo         = spio.loadmat(odfile)
    date           = odinfo['date']
    nu             = odinfo['nu'][0]
    rads           = odinfo['rads']
    rad_above      = odinfo['rad_above'][0]
    tsc            = odinfo['tsc']
    view_angle     = odinfo['view_angle']
    date_above_bef = odinfo['date_above_bef']
    date_above_aft = odinfo['date_above_aft']
    Bctc           = odinfo['Bc_tsc']
    dt_dtau        = odinfo['dt_dtau']
    # radClr    = odinfo['radClr'];
    # odlyr     = odinfo['odlyr']
    # print('Loaded od file ' + odfile)
    
    return date, view_angle, nu, rads, tsc, rad_above, \
           date_above_bef, date_above_aft, Bctc, dt_dtau


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def load_profiles(prof_file):
    '''
    Load in the profile file
    '''
    
    # .. Load in the profile
    with Dataset(prof_file, 'r', format='NETCDF4_CLASSIC') as nc:    
        z = np.double(nc['z'][:].data)
        P = np.double(nc['P'][:].data)
        T = np.double(nc['T'][:].data)
        h2o = np.double(nc['h2o'][:].data)
            
    return z, P, T, h2o


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def load_surface_temperatures(metfname, beg_datetime, end_datetime):
    '''
    Load the surface temperatures from a file
    '''
    with Dataset(metfname, 'r', format= "NETCDF4") as nc:
        itime = np.logical_and( \
          date2index(beg_datetime, nc.variables['time'], select='after'), \
          date2index(end_datetime, nc.variables['time'], select='before'))
        
        surf_time = num2date(nc.variables['time'][itime],
                             nc.variables['time'].units)
        surf_temp = nc.variables['temp_mean'][itime] + 273.15
        
        #ikeep = np.logical_and(np.where(surf_time>=beg_datetime)[0],
        #                       np.where(surf_time<=end_datetime)[0])
    
    return surf_time, surf_temp # surf_time[ikeep], surf_temp[ikeep]


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def load_surface_temperatures_for_datetime(metfname, thisdatetime, z, T):
    '''
    Load the surface temperatures from a file
    '''
    with Dataset(metfname, 'r', format= "NETCDF4") as nc:
        itime = date2index(thisdatetime, nc.variables['time'],
                           select='after')
        surf_time = num2date(nc.variables['time'][:],
                             nc.variables['time'].units)
        dd = surf_time[itime-1:itime+1] - thisdatetime
        dmin = [d.days*24*60 + d.seconds/60 for d in dd]
        wt = np.flipud(np.abs(dmin))
        wt = wt/np.sum(wt)
        surf_temp = wt[0] * nc.variables['temp_mean'][itime-1] + \
                    wt[1] * nc.variables['temp_mean'][itime] + 273.15

    i1km = np.where(z<=1)[0][-1]
    Tnew = np.zeros(i1km)
    Tnew[0] = surf_temp      # + 0 m
    Tnew[1:i1km] = np.interp(z[1:i1km], [z[0], z[i1km]], [Tnew[0], T[i1km]])
    
    return Tnew


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def get_prof_obsolete(prof_dir, prof_file, dir_lblrtm, z_toa, bwn, ewn):
    '''
    Load in the profile file
    '''
    
    
    # .. Load in the profile
    nc = Dataset(prof_dir + prof_file, 'r', format='NETCDF4_CLASSIC')
    
    zm = nc['z'][:]
    nlyr_toa = (np.abs(zm - z_toa)).argmin()
    itop = nlyr_toa + 1
    
    # .. Units for ozone are jchar = C for g/kg
    units = dict({('tm','km'), ('pm','mb'), ('h2o','ppmv'), 
                  ('co2','ppmv'), ('o3','gm_kg'), ('hno3','ppmv')})

    
    # .. Set prof values from prof_file, as well as inputs bwn and ewn
    #    viewing angle set to zenith. To output optical depths from LBLRTM,
    #    iemit is set to 0 and imrg to 1.
    prof = {
        "v1": bwn,
        "v2": ewn,
        "zm": zm[:itop],
        "pm": nc['P'][:itop],
        "tm": nc['T'][:itop],
        "h2o": nc['h2o'][:itop],
        "co2": nc['co2'][:itop],
        "o3": nc['o3'][:itop],
        "hno3": nc['hno3'][:itop],
        "f11": nc['f11'][:itop],
        "f12": nc['f12'][:itop],
        "f113": nc['f113'][:itop],
        "units": units,
        "zangle": 0,
        "fnametape5": dir_lblrtm + "TAPE5" ,
        "model": 0,
        "modelExtra": 3,
        "iod": 0 ,
        "iatm": 1,
        "ipunch": 1,
        "iemit": 0,
        "imrg": 1,
            }
    
    # .. Add this later?
    # if do_refl
    #   prof.surf_refl = surf_refl ;
    

    return prof, nlyr_toa


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def get_prof_pickle(prof_dir, prof_file, dir_lblrtm, z_toa, bwn, ewn):
    '''
    Load in the profile file that was pickled.
    '''
    
    # .. Load in the profile
    pstuff = pickle.load(open(prof_dir + prof_file, 'rb'))
    
    zm = pstuff.zm
    nlyr_toa = (np.abs(zm - z_toa)).argmin()
    itop = nlyr_toa + 1
    
    # .. Set prof values from prof_file, as well as inputs bwn and ewn
    #    viewing angle set to zenith. To output optical depths from LBLRTM,
    #    iemit is set to 0 and imrg to 1.
    prof = {
        "v1": bwn,
        "v2": ewn,
        "zm": pstuff.zm[:itop],
        "pm": pstuff.pm[:itop],
        "tm": pstuff.tm[:itop],
        "h2o": pstuff.h2o[:itop],
        "co2": pstuff.co2[:itop],
        "o3": pstuff.o3[:itop],
        "hno3": pstuff.hno3[:itop],
        "f11": pstuff.f11[:itop],
        "f12": pstuff.f12[:itop],
        "f113": pstuff.f113[:itop],
        "units": pstuff.units, 
        "zangle": 0,
        "fnametape5": dir_lblrtm + "TAPE5" ,
        "model": 0,
        "modelExtra": 3,
        "iod": 0 ,
        "iatm": 1,
        "ipunch": 1,
        "iemit": 0,
        "imrg": 1,
            }
    
    # .. Add this later?
    # if do_refl
    #   prof.surf_refl = surf_refl ;
    

    return prof, nlyr_toa

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def load_profiles_pickle(prof_file):
    pstuff = pickle.load(open(prof_file, 'rb'))
        
    return pstuff.zm, pstuff.pm, pstuff.tm, pstuff.h2o



def load_cloud_layers(cloud_layer_file, z, thisdatetime):
    
    with Dataset(cloud_layer_file, 'r', format = "NETCDF4") as nc:
        cloud_mask = nc['cloud_mask'][:].T
        height = nc['range'][:].data
        alt0 = nc['altitude'][0].data
        
        # .. Get the times
        mask_date = num2date(nc['time'][:], nc['time'].Description)
        imask = (np.abs(mask_date - thisdatetime)).argmin()
        
    
        # 0=>no cloud, 1=>ice, 2=>liquid, 3=>unknown, probably ice
        # Ice below 120 m is probably an artifact if there is not ice above,
        # so if there is cloud below 120 m, we will ignore it, unless
        # there is also cloud betwen 120 and 200 m
            
        icld_all = cloud_mask[:,imask].data != 0

        # .. If no cloud, set variables and return them now            
        if not np.any(icld_all):
            cloud_layer = array.array('i')
            has_ice = array.array('i')
            has_liq = array.array('i')
            return cloud_layer, has_liq, has_ice

        alt = alt0 + height[icld_all]
        mask = cloud_mask[icld_all,imask].data
        alt_liq = alt[mask==2]
        alt_ice = alt[np.logical_or(mask==1, mask==3)]
        
        if not np.all(alt[alt<=200] >= 120):
            alt_ice = alt_ice[alt_ice>=120]
            alt_liq = alt_liq[alt_liq>=120]
            alt = alt[alt>=120]
            
        
        
        # .. To make things easy, if there is any cloud within
        #    a model atmospheric layer, set the entire layer to cloudy
        #    Remember we are going top down
        #    Only try the range with cloud (+/- 30 m)
        alt = alt/1000; alt_liq = alt_liq/1000; alt_ice = alt_ice/1000
        iz1 = np.where(z >= alt[-1])[0]
        if np.any(iz1):
            iz1 = iz1[-1]
        else:
            iz1 = 0
        iz2 = np.where(z <= alt[0])[0]
        if np.any(iz2):
            iz2 = iz2[0]+1
        else:
            iz2 = len(z)-3
        
        if iz2 >= len(z)-2:
            iz2 = len(z)-3
            if iz2 < iz1:
                print('pause here!')

        Npossible = iz2 - iz1 + 1
        cloud_layer = array.array('i',(0 for i in range(Npossible)))
        has_ice = array.array('i',(0 for i in range(Npossible)))
        has_liq = array.array('i',(0 for i in range(Npossible)))
        i = 0
        for iz in range(iz1,iz2+1):
            if np.any((alt > z[iz+1]) * (alt <= z[iz])):
                cloud_layer[i] = iz
                if np.any((alt_liq > z[iz+1]) * (alt_liq <= z[iz])):
                    has_liq[i] = 1
                if np.any((alt_ice > z[iz+1]) * (alt_ice <= z[iz])):
                    has_ice[i] = 1
                i += 1
                
        cloud_layer = cloud_layer[:i]
        has_ice = has_ice[:i]
        has_liq = has_liq[:i]

    
    return cloud_layer, has_liq, has_ice
