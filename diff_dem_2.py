#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 08:06:07 2025

@author: bigett
"""

import numpy as np 
import xarray as xr 
# import rioxarray
import matplotlib.pyplot as plt 
# import glob 
# import pandas as pd 

import pytopocomplexity as tc
import xdem
# from osgeo import gdal
# import numpy as np
import rasterio
import os
from matplotlib.widgets import MultiCursor

ds = xr.open_dataset("stack_xdem.nc", engine="netcdf4")
# ds = xr.open_dataset("stack_xdem_sent2.nc", engine="netcdf4")

# ds['diff_elev'] = ds['elevation'] - ds['elevation_50m']


# %%
def calculate_CWTMexHat(pathin, pathout = './cache_explore/', replace = False, lambdaa = 5):
    
    if not os.path.exists('./cache_explore/'):
        os.makedirs('./cache_explore/')
    
    f = os.path.join(pathout, pathin.split('.')[0].split('/')[-1]) + 'TC' + 'CWTMexHat_' + str(lambdaa) +'_m.tif'
    if  not os.path.exists(f) or replace == True:
        cwt = tc.CWTMexHat(Lambda=lambdaa)
        Z, result = cwt.analyze(pathin)
        cwt.export_result(f)
    else:
        print('Feature already processed')
    with rasterio.open(f) as dataset:
        Z=dataset.read(1)
        
    Z[Z == -9999] = np.nan
    return Z


def perf(obs, pred):
    arr = (obs == pred)
    size = len(arr)* len(arr[0])
    return sum(sum(arr))/size


def x_diff(mat,  h):
    diff = np.zeros_like(mat)
    diff[:, 1:-1] = (mat [:, :-2] - mat [:, 2:])/(2*h)
    return diff

ds['x_diff'] = (('y','x'), x_diff(ds.diff_elev.values, 1.5))

def y_diff(mat,  h):
    diff = np.zeros_like(mat)
    diff[1:-1, :] = (mat [:-2, :] - mat [2:, :])/(2*h)
    return diff

ds['y_diff'] = (('y','x'), y_diff(ds.diff_elev.values, 1.5))
 
th = 0.01




slope50 = (ds.slope_horn_50m.values > 45) & ds.mask


# %%



fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2,3 , sharex = 1, sharey = 1)


x_value = (ds.x_diff.values) * (np.sin(ds.aspect_horn.values*np.pi/180))
y_value = (ds.y_diff.values) * (np.cos(ds.aspect_horn.values*np.pi/180))


# mat1 = ((x_value**2 + y_value**2)**0.5 <.58) & slope50 #.30
    
x_value = (ds.x_diff.values) * (np.sin(ds.aspect_horn.values*np.pi/180)**2) ######### top 1 
y_value = (ds.y_diff.values) * (np.cos(ds.aspect_horn.values*np.pi/180)**2) ######### top 1 


ax1.imshow(x_value, vmin = -.5, vmax = .5)
ax4.imshow(y_value, vmin = -.5, vmax = .5)


mat2= ((x_value + y_value)< 0.112) & ((x_value + y_value)> -.58) & slope50

# ax2.imshow(mat1, vmin = -.5, vmax = .5)
ax2.imshow(ds.snow_patches>1)
# ax5.imshow(mat2, vmin = -.5, vmax = .5)
ax5.imshow(ds.aspect_horn_50m)

ax3.imshow((mat2).astype('int')  - (~slope50)) ######### top 1

ax3.set_title('prediction') ######### top 1


ax6.imshow(ds.spot.isel(time = 0), cmap = 'grey')
ax6.set_title('SPOT7')
# ax6.imshow(ds.spot.isel(time = 0), cmap = 'managua')

ax1.set_title('x_value')
ax4.set_title('y_value')

ax2.set_title('snow w/ skimage')
ax5.set_title('aspect')

cursor = MultiCursor(fig.canvas, (ax1, ax2, ax3, ax4, ax5, ax6), color='cyan',lw=1, horizOn=True, vertOn=True)
plt.show()

# plt.imshow(ds.snow_patches.values>0, alpha= 0.4, cmap = 'grey' )

# ds['prediction'] = (['y','x'], mat2)
# %%
snow_obs_sample = mat2[700:950, 1250:1500]
snow_pred_sample = (ds.snow_patches.values>0)[700:950, 1250:1500]


arr = (snow_obs_sample == snow_pred_sample)


h1 = arr * ds.slope_horn_diff.values[700:950, 1250:1500]
h2 = ~arr * ds.slope_horn_diff.values[700:950, 1250:1500]

l1 = arr * ds.slope_horn.values[700:950, 1250:1500]
l2 = ~arr * ds.slope_horn.values[700:950, 1250:1500]


h11 = h1.values.reshape(1, -1)[0]
h12 = h2.values.reshape(1, -1)[0]

l11 = l1.values.reshape(1, -1)[0]
l12 = l2.values.reshape(1, -1)[0]

plt.figure()
plt.hist(h11, bins = 50)
plt.title('aspect prediction true')
plt.figure()
plt.hist(h12, bins = 50)
plt.title('aspect prediction false')

plt.figure()
plt.hist(l11, bins = 50)
plt.title('slope prediction true')
plt.figure()
plt.hist(l12, bins = 50)
plt.title('slope prediction false')

# %%

dem = ds.diff_elev.values

TPI5 = xdem.terrain.topographic_position_index(dem, 5)
TPI15 = xdem.terrain.topographic_position_index(dem, 15)
TPI25 = xdem.terrain.topographic_position_index(dem, 25)
TPI35 = xdem.terrain.topographic_position_index(dem, 35)

# %%



fig, ((ax5, ax15, axu), (ax25, ax35, axb )) = plt.subplots(2,3, sharex = True, sharey=1)

ax5.imshow(TPI5 < 0, vmax = 5, vmin = -2)
ax15.imshow(TPI15 < 0, vmax = 5, vmin = -2)
ax25.imshow(TPI25 < 0, vmax = 5, vmin = -2)
# ax35.imshow(TPI35, vmax = 5, vmin = -2)

TPI_th = ((TPI5 < 0) | (TPI15 < 0) | (TPI25 < 0)) & slope50

ax35.imshow(TPI_th.astype('int')-(~slope50))
axu.imshow(ds.spot.isel(time = 0).values)

axb.imshow(mat2.astype('int')-(~slope50))

cursor = MultiCursor(fig.canvas, (ax5, ax15, ax25, ax35, axu, axb), color='cyan',lw=1, horizOn=True, vertOn=True)
plt.show()

# %%

mex5 = calculate_CWTMexHat('/home/bigett/Bureau/03_verte/dem_high_pass_1-5_50m_verte.tif', lambdaa = 5)
mex10 = calculate_CWTMexHat('/home/bigett/Bureau/03_verte/dem_high_pass_1-5_50m_verte.tif', lambdaa = 10)
mex15 = calculate_CWTMexHat('/home/bigett/Bureau/03_verte/dem_high_pass_1-5_50m_verte.tif', lambdaa = 15)
mex25 = calculate_CWTMexHat('/home/bigett/Bureau/03_verte/dem_high_pass_1-5_50m_verte.tif', lambdaa = 25)
# %%

fig, ((ax5, ax15, axu), (ax25, ax35, axb )) = plt.subplots(2,3, sharex = True, sharey=1)

ax5.imshow(mex5, vmax = 5, vmin = -2)
ax15.imshow(mex10 < 0, vmax = 5, vmin = -2)
ax25.imshow(mex15 < 0, vmax = 5, vmin = -2)
ax35.imshow(mex25, vmax = 5, vmin = -2)

TPI_th = ((TPI5 < 0) | (TPI15 < 0) | (TPI25 < 0)) & slope50

# ax35.imshow(TPI_th.astype('int')-(~slope50))
axu.imshow(ds.spot.isel(time = 0).values)

axb.imshow(mat2.astype('int')-(~slope50))

cursor = MultiCursor(fig.canvas, (ax5, ax15, ax25, ax35, axu, axb), color='cyan',lw=1, horizOn=True, vertOn=True)
plt.show()
# %%
x_value = (ds.x_diff.values) * (np.sin(ds.aspect_horn.values*np.pi/180))
y_value = (ds.y_diff.values) * (np.cos(ds.aspect_horn.values*np.pi/180))
mat1 = ((x_value**2 + y_value**2)**0.5 <.6) & slope50
    
x_value = (ds.x_diff.values) * (np.sin(ds.aspect_horn.values*np.pi/180)**2) ######### top 1 
y_value = (ds.y_diff.values) * (np.cos(ds.aspect_horn.values*np.pi/180)**2) ######### top 1 
mat2= ((x_value + y_value)< 0.112) & ((x_value + y_value)> -0.58) & slope50

snow_obs = (ds.snow_patches >0) & slope50
snow_pred = mat1 & mat2 & slope50

snow_obs_sample = snow_obs[700:950, 1250:1500]
snow_pred_sample = snow_pred[700:950, 1250:1500]

per = perf(snow_obs_sample, snow_pred_sample)
 
print(per)


# %%
from tqdm import tqdm

list_score = []
for i in tqdm(np.arange(0.5, 0.7, 0.01)):
    for j in tqdm(np.arange(0.10, 0.12, 0.002)):
        for k in np.arange(0.5, 0.7, 0.02):
            x_value = (ds.x_diff.values[700:950, 1250:1500]) * (np.sin(ds.aspect_horn.values[700:950, 1250:1500]*np.pi/180))
            y_value = (ds.y_diff.values[700:950, 1250:1500]) * (np.cos(ds.aspect_horn.values[700:950, 1250:1500]*np.pi/180))
            mat1 = ((x_value**2 + y_value**2)**0.5 <i) & slope50[700:950, 1250:1500]
                
            x_value = (ds.x_diff.values[700:950, 1250:1500]) * (np.sin(ds.aspect_horn.values[700:950, 1250:1500]*np.pi/180)**2) ######### top 1 
            y_value = (ds.y_diff.values[700:950, 1250:1500]) * (np.cos(ds.aspect_horn.values[700:950, 1250:1500]*np.pi/180)**2) ######### top 1 
            mat2= ((x_value + y_value)< j) & ((x_value + y_value)> -k) & slope50[700:950, 1250:1500]
            
            snow_obs = (ds.spot.isel(time = 0).values[700:950, 1250:1500] > 3200) & slope50[700:950, 1250:1500]
            snow_pred = mat1 & mat2 & slope50[700:950, 1250:1500]
            
            snow_obs_sample = snow_obs
            snow_pred_sample = snow_pred
            
            per = perf(snow_obs_sample, snow_pred_sample)
            
            list_score.append([i,j,k, per])












# %%
snow_obs = (ds.spot.isel(time = 0).values > 3200) & slope50
snow_pred = ((x_value + y_value)< 0.07) & ((x_value + y_value)> -0.44) & slope50

snow_obs_sample = snow_obs[700:950, 1250:1500]
snow_pred_sample = snow_pred[700:950, 1250:1500]

per = perf(snow_obs_sample, snow_pred_sample)

print(per)

fig, (ax1, ax2, ax3) = plt.subplots(1,3 , sharex = 1, sharey = 1)
ax1.imshow(snow_obs_sample)
ax2.imshow(snow_pred_sample)
ax3.imshow(ds.spot.isel(time = 2).values[700:950, 2800:3000])

ax1.set_title('obs')
ax2.set_title('pred')
ax3.set_title('SPOT')

cursor = MultiCursor(fig.canvas, (ax1, ax2, ax3), color='b',lw=1, horizOn=True, vertOn=True)
plt.show()


# %%
list_per = []
for i in np.arange(1, 3, 0.1):
    x_value = (ds.x_diff.values) * abs(np.sin(ds.aspect_horn.values*np.pi/180)**i)
    y_value = (ds.y_diff.values) * abs(np.cos(ds.aspect_horn.values*np.pi/180)**i)

    snow_obs = (ds.spot.isel(time = 2).values > 3200) & slope50
    snow_pred = ((x_value + y_value)< 0.07) & ((x_value + y_value)> -0.44) & slope50

    snow_obs_sample = snow_obs[700:950, 1250:1500]
    snow_pred_sample = snow_pred[700:950, 1250:1500]

    list_per.append(perf(snow_obs_sample, snow_pred_sample))
    print(i)

plt.plot( np.arange(1, 3, 0.1), list_per)

# %%


fig, (ax1, ax2, ax3) = plt.subplots(1,3 , sharex = 1, sharey = 1)
ax1.imshow(snow_obs_sample)
ax2.imshow(snow_pred_sample)
ax3.imshow(ds.spot.isel(time = 0).values[700:950, 1250:1500])

ax1.set_title('obs')
ax2.set_title('pred')
ax3.set_title('SPOT')

cursor = MultiCursor(fig.canvas, (ax1, ax2, ax3), color='b',lw=1, horizOn=True, vertOn=True)
plt.show()


# %%






