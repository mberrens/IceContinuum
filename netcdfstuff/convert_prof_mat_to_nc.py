#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 14:36:46 2019

@author: prowe
"""

import numpy as np
import scipy.io as spio
from netCDF4 import Dataset
from netCDF4 import date2num 
import time
import os
from datetime import datetime


# # # # # # #     Get file names     # # # # # # #
def get_files(directory, sample_filename, i1, i2, fstr):
    files = os.listdir(directory)
    files.sort()
    files = filter(lambda x: len(x)==len(sample_filename), files)
    file_n_date = map(lambda f: (f, datetime.strptime(f[i1:i2], fstr)), files)
    return tuple(file_n_date)



# # # # # # #     Profile     # # # # # # #
class Prof():
    __slots__ = ('file',
                 'date',
                 'time',
                 'zm',
                 'tm',
                 'pm',
                 'rh',
                 'h2o',
                 'co2',
                 'o3',
                 'f11',
                 'f12',
                 'f113',
                 'hno3',
                 'units',
                 'model_extra')
        
    def __init__(self, prof_file, prof_date):
        '''
        Load in the profile file and set some values
        '''
        
        # .. Load in the profile
        pstuff = spio.loadmat(prof_file)
        prof0 = pstuff['prof']          # prof0 never changes
        
        # .. Set prof values from prof_file, as well as inputs bwn and ewn
        #    viewing angle set to zenith. To output optical depths from LBLRTM,
        #    iemit is set to 0 and imrg to 1.
        self.file = prof_file
        self.date = prof_date
        self.zm = np.double(prof0['zm'][0][0][:])
        self.pm = np.double(prof0['pm'][0][0][:])
        self.tm = np.double(prof0['tm'][0][0][:])
        self.h2o = np.double(prof0['h2o'][0][0][:])
        self.rh = np.double(prof0['rh'][0][0][:])
        self.co2 = np.double(prof0['co2'][0][0][:])
        self.o3 = np.double(prof0['o3'][0][0][:])
        self.hno3 = np.double(prof0['hno3'][0][0][:])
        self.f11 = np.double(prof0['f11'][0][0][:])
        self.f12 = np.double(prof0['f12'][0][0][:])
        self.f113 = np.double(prof0['f113'][0][0][:])
        
        # units = prof0['units'][0][0][0][0],
        
        # .. Hard-writing these
        units = dict({('zm','km'), ('tm','K'), ('pm','mb'), ('h2o','ppmv'), 
                      ('co2','ppmv'), ('o3','gm_kg'), ('hno3','ppmv'),
                      ('f11','ppmv'), ('f12','ppmv'), ('f113','ppmv')})
        self.units = units

        
    def error_check(self):
        # .. Check for NaNs
        if np.any(np.isnan(self.zm)) or np.any(np.isnan(self.pm)) or \
           np.any(np.isnan(self.co2)) or np.any(np.isnan(self.h2o)) or \
           np.any(np.isnan(self.o3)) or np.any(np.isnan(self.f11)) or \
           np.any(np.isnan(self.f12)) or np.any(np.isnan(self.f11)) or \
           np.any(np.isnan(self.tm)):
               raise NameError('One or more values is NaN!')
               
        # .. Check for heights (zm) that do not monotonically increase or 
        #    pressures (pm) that do not monotonically decrease
        if (np.any(np.diff(self.zm)<=0)) or (np.any(np.diff(self.pm)>=0)):
            raise NameError('zs do not increase or Ps do not decrease.')


    
    def write_to_netcdf_file(self, outputfile):
        with Dataset(outputfile, 'w', format='NETCDF4_CLASSIC') as nc:    
            # .. Create Dimensions
            nc.createDimension('level', len(self.zm)) 
            nc.createDimension('time', 1)
            nc.createDimension('const', 1)
            # lat = nc.createDimension('lat', 1)
            # lon = nc.createDimension('lon', 1) 
         
            # .. Create variables
            nc_time = nc.createVariable('time', np.float64, ('time',))
            nc_z = nc.createVariable('z', np.float32, ('level'))
            nc_T = nc.createVariable('T', np.float32, ('level'))
            nc_P = nc.createVariable('P', np.float32, ('level'))
            nc_rh = nc.createVariable('rh', np.float32, ('level'))
            nc_h2o = nc.createVariable('h2o', np.float32, ('level'))
            nc_co2 = nc.createVariable('co2', np.float32, ('level'))
            nc_o3 = nc.createVariable('o3', np.float32, ('level'))
            nc_f11 = nc.createVariable('f11', np.float32, ('level'))
            nc_f12 = nc.createVariable('f12', np.float32, ('level'))
            nc_f113 = nc.createVariable('f113', np.float32, ('level'))
            nc_hno3 = nc.createVariable('hno3', np.float32, ('level'))
            nc_model_extra = nc.createVariable('model_extra',np.int8,('const'))
        
            
            # .. Global attributes
            nc.filename = self.file
            nc.history = 'Created ' + time.ctime(time.time())
            
            # .. Variable attributes
            nc_time.units = 'hours since 0001-01-01 00:00:00'
            nc_time.calendar = 'gregorian'
            nc_z.units = self.units['zm']
            nc_T.units = self.units['tm']
            nc_P.units = self.units['pm']
            nc_rh.units = '%'
            nc_h2o.units = self.units['h2o']
            nc_co2.units = self.units['co2']
            nc_o3.units = self.units['o3']
            nc_hno3.units  = self.units['hno3']
            nc_f11.units = self.units['f11']
            nc_f12.units = self.units['f12']
            nc_f113.units = self.units['f113']
        
            # .. Assign values   
            nc_time[:] = date2num(self.date, units = nc_time.units, 
                                  calendar = nc_time.calendar) 
            nc_z[:] = self.zm
            nc_T[:] = self.tm
            nc_P[:] = self.pm
            nc_rh[:] = self.rh
            nc_h2o[:] = self.h2o
            nc_co2[:] = self.co2
            nc_o3[:] = self.o3
            nc_f11[:] = self.f11
            nc_f12[:] = self.f12
            nc_f113[:] = self.f113
            nc_hno3[:] = self.hno3
            nc_model_extra[:] = self.model_extra       


out_dir = '/Users/prowe/Projects/NSF_AERI_cloud_CRI/SouthPole/profFiles/'
prof_dir = '/Users/prowe/Projects/NSF_AERI_cloud_CRI/SouthPole/profFiles/mat/'
sample_file = 'prof20010101_0831.mat'
prof_file_n_dates = get_files(prof_dir, sample_file,  4, 16, '%Y%m%d_%H%M')


for file, prof_date in prof_file_n_dates:    
    print('Working on', file)
    outfile = file[:-4] + '.nc'
    prof = Prof(prof_dir + file, prof_date)
    prof.model_extra = 3
    prof.write_to_netcdf_file(out_dir + outfile)