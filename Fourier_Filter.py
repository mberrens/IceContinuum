
# coding: utf-8

# In[158]:


import numpy as np

import copy 
import os
import time
from sys import platform
from scipy.interpolate import griddata

def FT(grid,x,y):


    # Fourier transform it
    grid_FT = np.fft.fft2(grid)
    grid_FTshift = np.fft.fftshift(grid_FT)

    # And get the frequencies
    Ny, Nx = np.shape(grid)
    dx = x[1]-x[0]
    dy = y[1]-y[0]
    kx = np.fft.fftfreq(Nx)*np.pi/dx
    ky = np.fft.fftfreq(Ny)*np.pi/dy
    kxshift = np.fft.fftshift(kx)
    kyshift = np.fft.fftshift(ky)
    return grid_FTshift, kxshift, kyshift


def IFT(sollast_FTshift_filtered):

    # Un-shift it
    sollast_FT_filtered = np.fft.ifftshift(sollast_FTshift_filtered)

    # Inverse Fourier transform
    sollast_FT_filtered_IFT = np.fft.ifft2(sollast_FT_filtered)

    return sollast_FT_filtered_IFT


# # Plot it
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(xgrid, ygrid, np.real(sollast_FT_filtered_IFT))
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_title('real')

# # Plot it
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(xgrid, ygrid, np.real(sollast_FT_filtered_IFT-sollast))
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_title('real difference')

# # Plot it
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(xgrid, ygrid, np.imag(sollast_FT_filtered_IFT))
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_title('imag')

