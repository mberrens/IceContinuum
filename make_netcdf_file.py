

import numpy as np
from netCDF4 import Dataset
#from netCDF4 import stringtochar

def make_netcdf_file(ret, outputfile, doSlowRetrieval, doFastRetrieval,
                     mradhastruevalues):
    '''
    Purpose: Save retrieval variables to netcdf file.
    Outputs (unless specified these are x Nspecs, or number of spectra):
      fast_Xret
      Xa
      Sa
      nu
      height
      datenumber
      hour
      viewAngle
      iscloud:        1=>cloud exists, 0=>no cloud found
      cldbase_liq
      cldtop__liq
      cldbase_ice
      cldtop__ice
      Nclds
      maxIters
      Ra
      Rf
      Xfin:            [4 x Nspecs]
      Xmin:            [4 x Nspecs]
      Xret:            [4 x Nspecs]
      chi2s
      rmsDiff
      rmsDiffret
      h2o
      temp
      Se
      Rm
      imin
      rmsDiffa
      Niters
      Nrmsiters
      convFlag
    Optional (only when true values in measured radiance file:
      trueX           ('spectrum','cloudprop'))
      trueCloudBase   ('spectrum'))
      trueCloudTop    ('spectrum'))

    '''
    
    # Use netcdf4 to save the results
    with Dataset(outputfile, 'w', format='NETCDF4_CLASSIC') as nc:
      # nc = Dataset(outputfile, 'w', format='NETCDF4_CLASSIC')
    
      # .. Create Dimensions
      nc.createDimension('spectrum', len(ret.datenumber))
      nc.createDimension('maxIterationsFast', ret.fast_Xret.shape[1])
      nc.createDimension('cloudprop', len(ret.Xa)) 
      nc.createDimension('cloudpropsq', len(ret.Xa)*len(ret.Xa)) 
      nc.createDimension('wavenumber', len(ret.nu))
      nc.createDimension('z', len(ret.height))
      nc.createDimension('maxIterations', ret.chi2s.shape[1])
      nc.createDimension('ncharOD', len(ret.odfiles[0]))
      nc.createDimension('ncharProf', len(ret.profFiles[0]))
      nc.createDimension('Nweights', 2)
      nc.createDimension('NprofFiles', len(ret.profFiles))
      nc.createDimension('oneval', 1)  
      
      # .. Create variables
      nc_fast_Xret    = nc.createVariable('fast_Xret', np.float32,('spectrum','maxIterationsFast'))
      nc_Xa           = nc.createVariable('Xa',np.float32,('cloudprop'))
      nc_Sa           = nc.createVariable('Sa',np.float32,('cloudprop'))
      nc_nu           = nc.createVariable('nu',np.float32,('wavenumber'))
      nc_height       = nc.createVariable('height',np.float32,('z'))
      nc_datenumber   = nc.createVariable('datenumber',np.float64,('spectrum'))
      nc_hour         = nc.createVariable('hour',np.float64,('spectrum'))
      nc_viewAngle    = nc.createVariable('viewAngle',np.float64,('spectrum'))
      nc_iscloud      = nc.createVariable('iscloud','i2',('spectrum'))
      nc_cldbase_liq  = nc.createVariable('cldbase_liq',np.float32,('spectrum')) 
      nc_cldtop__liq  = nc.createVariable('cldtop__liq',np.float32,('spectrum')) 
      nc_cldbase_ice  = nc.createVariable('cldbase_ice',np.float32,('spectrum')) 
      nc_cldtop__ice  = nc.createVariable('cldtop__ice',np.float32,('spectrum')) 
      nc_Nclds        = nc.createVariable('Nclds',np.float32,('spectrum')) 
      nc_maxIters     = nc.createVariable('maxIters','i2',('spectrum'))
      nc_imin         = nc.createVariable('imin','i2',('spectrum')) 
      nc_rmsDiffa     = nc.createVariable('rmsDiffa',np.float32,('spectrum')) 
      nc_rmsDiffret   = nc.createVariable('rmsDiffret',np.float32,('spectrum')) 
      nc_Niters       = nc.createVariable('Niters','i2',('spectrum')) 
      nc_Nrmsiters    = nc.createVariable('Nrmsiters','i2',('spectrum')) 
      nc_convFlag     = nc.createVariable('convFlag','i2',('spectrum')) 
      nc_h2o          = nc.createVariable('h2o',np.float32,('spectrum','z'))
      nc_temp         = nc.createVariable('temp',np.float32,('spectrum','z'))
      nc_Se           = nc.createVariable('Se',np.float32,('spectrum','wavenumber'))
      nc_Rm           = nc.createVariable('Rm',np.float32,('spectrum','wavenumber'))
      nc_Ra           = nc.createVariable('Ra',np.float32,('spectrum','wavenumber'))
      nc_Rf           = nc.createVariable('Rf',np.float32,('spectrum','wavenumber'))
      nc_Xfin         = nc.createVariable('Xfin',np.float32,('spectrum','cloudprop')) 
      nc_Xmin         = nc.createVariable('Xmin',np.float32,('spectrum','cloudprop')) 
      nc_S            = nc.createVariable('S',np.float32,('spectrum','cloudpropsq'))
      nc_Xret         = nc.createVariable('Xret',np.float32,('spectrum','cloudprop')) 
      nc_chi2s        = nc.createVariable('chi2s',np.float32,('spectrum','maxIterations'))
      nc_rmsDiff      = nc.createVariable('rmsDiff',np.float32,('spectrum','maxIterations'))
      # Put the following back later?
      #nc_odFiles      = nc.createVariable('odFiles','S1',('NprofFiles','ncharOD'))
      #nc_profFiles    = nc.createVariable('profFiles','S1',('NprofFiles','ncharProf'))
      nc_iodfile      = nc.createVariable('iodfile',np.float32,('spectrum','Nweights'))
      nc_iprofFile    = nc.createVariable('iprofFile',np.float32,('spectrum','Nweights'))
      nc_timeWeight   = nc.createVariable('timeWeight',np.float32,('spectrum','Nweights'))
      nc_lat          = nc.createVariable('lat',np.float32,('spectrum')) #('oneval'))
      nc_lon          = nc.createVariable('lon',np.float32,('spectrum')) #('oneval'))
      nc_alt          = nc.createVariable('alt',np.float32,('spectrum')) #('oneval'))
      
          
      # .. Assign values      
      nc_fast_Xret[:]   = ret.fast_Xret
      nc_Xa[:]          = ret.Xa 
      nc_Sa[:]          = ret.Sa 
      nc_nu[:]          = ret.nu
      nc_height[:]      = ret.height 
      nc_datenumber[:]  = ret.datenumber
      nc_hour[:]        = ret.hour
      nc_viewAngle[:]   = ret.viewAngle
      nc_iscloud[:]     = ret.iscloud
      nc_cldbase_liq[:] = ret.cldbase_liq 
      nc_cldtop__liq[:] = ret.cldtop__liq 
      nc_cldbase_ice[:] = ret.cldbase_ice 
      nc_cldtop__ice[:] = ret.cldtop__ice 
      nc_Nclds[:]       = ret.Nclds
      nc_maxIters[:]    = ret.maxIters 
      nc_Ra[:]          = ret.Ra 
      nc_Rf[:]          = ret.Rf 
      nc_Xfin[:]        = ret.Xfin
      nc_Xmin[:]        = ret.Xmin
      nc_Xret[:]        = ret.Xret
      nc_chi2s[:]       = ret.chi2s 
      nc_rmsDiff[:]     = ret.rmsDiff
      nc_rmsDiffret[:]  = ret.rmsDiffret
      nc_h2o[:]         = ret.h2o 
      nc_temp[:]        = ret.temp 
      nc_Se[:]          = ret.Se 
      nc_Rm[:]          = ret.Rm 
      nc_imin[:]        = ret.imin 
      nc_rmsDiffa[:]    = ret.rmsDiffa 
      nc_Niters[:]      = ret.Niters 
      nc_Nrmsiters[:]   = ret.Nrmsiters 
      nc_convFlag[:]    = ret.convFlag 

      # The covariance matrix has an extra dimension, so stack up the rows
      # horizontally and save that
      nc_S[:]=np.hstack([ret.S[:,:,0],ret.S[:,:,1],ret.S[:,:,2],ret.S[:,:,3]])
      
      
      if ret.retrieveCloudHts:
          nc_cldHtMlev  = nc.createVariable('cldHtMlev',np.float32,('spectrum'))  
          nc_rmsDiffSlice = nc.createVariable('rmsDiffSlice',np.float32,('spectrum'))  
          nc_cldHtSlice = nc.createVariable('cldHtSlice',np.float32,('spectrum'))  
          nc_cldHtSlice[:]  = ret.cldHtSlice 
          nc_rmsDiffSlice[:] = ret.rmsDiffSlice 
          nc_cldHtMlev[:] = ret.cldHtMlev 
  
      # .. Put this back later?
      #sformo = 'S'+str(len(ret.odfiles[0]))
      #sformp = 'S'+str(len(ret.profFiles[0]))
      #for ifl in range(len(ret.profFiles)):
      #    nc_odFiles[ifl]   = stringtochar(np.array(ret.odfiles[ifl], sformo)) 
      #    nc_profFiles[ifl] = stringtochar(np.array(ret.profFiles[ifl], sformp)) 
        
      nc_iodfile[:]     = ret.iodfile
      nc_iprofFile[:]   = ret.iprofFile
      nc_timeWeight[:]  = ret.timeWeight
      nc_lat[:]         = ret.lat
      nc_lon[:]         = ret.lon
      nc_alt[:]         = ret.alt
      
      if mradhastruevalues:
          nc_trueX = nc.createVariable('trueX',np.float32,('spectrum','cloudprop')) 
          nc_trueCloudBase = nc.createVariable('trueCloudBase',np.float32,('spectrum')) 
          nc_trueCloudTop = nc.createVariable('trueCloudTop',np.float32,('spectrum')) 
          nc_trueX[:] = ret.trueX
          nc_trueCloudBase[:] = ret.trueCloudBaseLiq 
          nc_trueCloudTop[:] = ret.trueCloudTopLiq 
    
    #nc.close()
