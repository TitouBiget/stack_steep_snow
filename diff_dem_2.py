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
slope50 = (ds.slope_horn_50m.values > 45) & ds.mask



# %%
def Confusion_matrix(snow_obs_sample, snow_pred_sample, plot = True):
    arr= snow_obs_sample.astype('int') - snow_pred_sample.astype('int') 
    arr2 = snow_obs_sample.astype('int') + snow_pred_sample.astype('int') 
    size = len(snow_obs_sample)* len(snow_obs_sample[0])
    
    arr_22 = sum(sum((arr2 == 0)))/size
    arr_11 = sum(sum((arr2 == 2)))/size
    
    arr_21 = sum(sum((arr == 1)))/size
    arr_12 = sum(sum((arr == -1)))/size
    
    
    conf_matrix = np.array([[arr_11, arr_21] , [arr_12, arr_22]])
    
    if plot == True:
        plt.figure()
        plt.title(f'Total: {size}, prediction right: {arr_11 + arr_22}, prediction false: {arr_12 + arr_21}')
        plt.imshow(conf_matrix)
        plt.xticks([0,1], ['True', 'False'])
        plt.yticks([0,1], ['True', 'False'])
        plt.ylabel('Predicted')
        plt.xlabel('Observed')
        plt.text(0, 0 , s = f'{arr_11:.0%}', horizontalalignment='center', verticalalignment='center', fontsize = 15)
        plt.text(0, 1 , s = f'{arr_12:.0%}', horizontalalignment='center', verticalalignment='center', fontsize = 15)
        plt.text(1, 0 , s = f'{arr_21:.0%}', horizontalalignment='center', verticalalignment='center', fontsize = 15)
        plt.text(1, 1 , s = f'{arr_22:.0%}', horizontalalignment='center', verticalalignment='center', fontsize = 15)
    return conf_matrix


def perf(obs, pred):
    arr = (obs == pred)
    size = len(arr)* len(arr[0])
    return (sum(sum(arr))/size)


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

def submatrix(mat, lim = (0,0,0,0)):
    xmin, xmax, ymin, ymax = lim
    return mat[xmin:xmax, ymin:ymax]



# %%

x_value = (ds.x_diff.values) * (np.sin(ds.aspect_horn.values*np.pi/180)**2) ######### top 1 for spot 7
y_value = (ds.y_diff.values) * (np.cos(ds.aspect_horn.values*np.pi/180)**2) ######### top 1 for spot 7

mat2= ((x_value + y_value)< 1.4) & ((x_value + y_value)> -.9) & slope50

if True:
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2,3 , sharex = 1, sharey = 1)
    
    
    # x_value = (ds.x_diff.values) * (np.sin(ds.aspect_horn.values*np.pi/180))
    # y_value = (ds.y_diff.values) * (np.cos(ds.aspect_horn.values*np.pi/180))
    
    
    # mat1 = ((x_value**2 + y_value**2)**0.5 <.58) & slope50 #.30
        
    
    
    ax1.imshow(x_value, vmin = -.5, vmax = .5)
    ax4.imshow(y_value, vmin = -.5, vmax = .5)
    
    
    # mat2= ((x_value + y_value)< 0.112) & ((x_value + y_value)> -.58) & slope50 ######### top 1 for spot 7
    
    
    ax2.imshow(ds.x_diff, vmin = -.5, vmax = .5)
    # ax2.imshow(ds.snow_patches>1)
    # ax5.imshow(mat2, vmin = -.5, vmax = .5)
    ax5.imshow(ds.y_diff, vmin = -.5, vmax = .5)
    
    ax3.imshow((mat2).astype('int')  - (~slope50)) ######### top 1
    
    ax3.set_title('prediction') ######### top 1
    
    
    ax6.imshow(ds.spot.isel(time = 1), cmap = 'grey')
    ax6.set_title('PLE 2021_05_23')

    
    # ax6.imshow(ds.spot.isel(time = 0), cmap = 'managua')
    
    ax1.set_title('x_value')
    ax4.set_title('y_value')
    
    ax2.set_title("x'")
    ax5.set_title("y'")
    
    cursor = MultiCursor(fig.canvas, (ax1, ax2, ax3, ax4, ax5, ax6), color='cyan',lw=1, horizOn=True, vertOn=True)
    plt.show()

zone1 = (700,950,1250,1500) #[700:950, 1250:1500]
zone2 = (780,975,780,1100) #[780:975, 780:1100]
zone3 = (950,1060,1820,2220) #[950:1060, 1820:2220]


c1 = Confusion_matrix( submatrix(mat2.values, zone1), submatrix(ds.snow_patches.values>0, zone1), 1)
p1 = perf( submatrix(mat2.values, zone1), submatrix(ds.snow_patches.values>0, zone1))
rd1 = perf( submatrix((np.random.rand(2250,2250) > 0.5).astype(int), zone1), submatrix(ds.snow_patches.values>0, zone1))
one1 = perf( submatrix(np.ones((2250,2250)), zone1), submatrix(ds.snow_patches.values>0, zone1))
zero1 = perf( submatrix(np.zeros((2250,2250)), zone1), submatrix(ds.snow_patches.values>0, zone1))

c2 = Confusion_matrix( submatrix(mat2.values, zone2), submatrix(ds.snow_patches.values>0, zone2), 1)
p2 = perf( submatrix(mat2.values, zone2), submatrix(ds.snow_patches.values>0, zone2))
rd2 = perf( submatrix(((np.random.rand(2195,2320) > 0.5).astype(int)), zone2), submatrix(ds.snow_patches.values>0, zone2))
one2 = perf( submatrix(np.ones((2250,2250)), zone2), submatrix(ds.snow_patches.values>0, zone2))
zero2 = perf( submatrix(np.zeros((2250,2250)), zone2), submatrix(ds.snow_patches.values>0, zone2))

c3 = Confusion_matrix( submatrix(mat2.values, zone3), submatrix(ds.snow_patches.values>0, zone3), 1)
p3 = perf( submatrix(mat2.values, zone3), submatrix(ds.snow_patches.values>0, zone3))
rd3 = perf( submatrix(((np.random.rand(3200,3000) > 0.5).astype(int)), zone3), submatrix(ds.snow_patches.values>0, zone3))
one3 = perf( submatrix(np.ones((3200,3200)), zone3), submatrix(ds.snow_patches.values>0, zone3))
zero3 =  perf( submatrix(np.zeros((3200,3200)), zone3), submatrix(ds.snow_patches.values>0, zone3))

# print('c1', c1[0,0], c1[1,1], ' rd1', rd1[0,0], rd1[1,1])
# print('c2', c2[0,0], c2[1,1], ' rd2', rd2[0,0], rd2[1,1])
# print('c3', c3[0,0], c3[1,1], ' rd3', rd3[0,0], rd3[1,1])

avg1 = (c1[0,1] + c2[0,1] + c3[0,1])/3
avg2 = (c1[1,0] + c2[1,0] + c3[1,0])/3

print(avg1 + avg2)

print('perf ratio:')
print("Zone 1: p1 ", p1, " rd1 ", rd1, " one1 ", one1, " zero1 ", zero1)
print("Zone 2: p2 ", p2, " rd2 ", rd2, " one1 ", one2, " zero2 ", zero2)
print("Zone 3: p3 ", p3, " rd3 ", rd3, " one3 ", one3, " zero3 ", zero3)

# Confusion_matrix( np.ones((195,320)), (ds.snow_patches.values>0)[780:975, 780:1100])

# plt.imshow

# %%


from tqdm import tqdm

list_score = []



x_value = (ds.x_diff.values) * (np.sin(ds.aspect_horn.values*np.pi/180)**2) ######### top 1 for spot 7
y_value = (ds.y_diff.values) * (np.cos(ds.aspect_horn.values*np.pi/180)**2) ######### top 1 for spot 7

zone1 = (700,950,1250,1500) #[700:950, 1250:1500]
zone2 = (780,975,780,1100) #[780:975, 780:1100]
zone3 = (950,1060,1820,2220) #[950:1060, 1820:2220]

l = len(np.arange(0, 3, 0.1))
heatmap = np.zeros((l,l))

for i, j in enumerate(np.arange(0, 3, 0.1)):
    for m, k in enumerate(np.arange(0, 3, 0.1)):


        mat2= ((x_value + y_value)< j) & ((x_value + y_value)> -k) & slope50

        c1 = Confusion_matrix( submatrix(mat2.values, zone1), submatrix(ds.snow_patches.values>0, zone1), False)
        rd1 = Confusion_matrix( submatrix((np.random.rand(2250,2250) > 0.5).astype(int), zone1), submatrix(ds.snow_patches.values>0, zone1), False)
        
        c2 = Confusion_matrix( submatrix(mat2.values, zone2), submatrix(ds.snow_patches.values>0, zone2), False)
        rd2 = Confusion_matrix( submatrix(((np.random.rand(2195,2320) > 0.5).astype(int)), zone2), submatrix(ds.snow_patches.values>0, zone2), False)
        
        c3 = Confusion_matrix( submatrix(mat2.values, zone3), submatrix(ds.snow_patches.values>0, zone3), False)
        rd3 = Confusion_matrix( submatrix(((np.random.rand(3200,3000) > 0.5).astype(int)), zone3), submatrix(ds.snow_patches.values>0, zone3), False)
        
        
        # print('c1', c1[0,0], c1[1,1], ' rd1', rd1[0,0], rd1[1,1])
        # print('c2', c2[0,0], c2[1,1], ' rd2', rd2[0,0], rd2[1,1])
        # print('c3', c3[0,0], c3[1,1], ' rd3', rd3[0,0], rd3[1,1])
        
        avg1 = (c1[0,1] + c2[0,1] + c3[0,1])/3
        avg2 = (c1[1,0] + c2[1,0] + c3[1,0])/3
        
        heatmap[i,m] = avg1 + avg2
    print(i)

plt.imshow(heatmap, vmax = 0.28)
# plt.xticks(range(l), labels= np.arange(0, 3, 0.1))
# plt.yticks(range(l), labels= np.arange(0, 3, 0.1))
           

# %%
snow_obs_sample = mat2[700:950, 1250:1500]
snow_pred_sample = (ds.snow_patches.values>0)[700:950, 1250:1500]


arr = (snow_obs_sample == snow_pred_sample)


h1 = arr * ds.slope_horn_diff.values[700:950, 1250:1500]
h2 = ~arr * ds.slope_horn_diff.values[700:950, 1250:1500]

h11 = h1.values.reshape(1, -1)[0]
h12 = h2.values.reshape(1, -1)[0]

plt.figure()
plt.hist(h11, bins = 50)
plt.title('aspect prediction true')
plt.figure()
plt.hist(h12, bins = 50)
plt.title('aspect prediction false')
# %% confusion_matrix

Confusion_matrix( mat2[700:950, 1250:1500], (ds.snow_patches.values>0)[700:950, 1250:1500])

# %%

l1 = arr * ds.slope_horn.values[700:950, 1250:1500]
l2 = ~arr * ds.slope_horn.values[700:950, 1250:1500]

l11 = l1.values.reshape(1, -1)[0]
l12 = l2.values.reshape(1, -1)[0]

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
import pywt

# Load image
original = ds.diff_elev.values

# Wavelet transform of image, and plot approximation and details
titles = ['Approximation', ' Horizontal detail',
          'Vertical detail', 'Diagonal detail']
coeffs2 = pywt.dwt2(original, 'bior1.1')
LL, (LH, HL, HH) = coeffs2
fig = plt.figure(figsize=(12, 3))
ax1 = fig.add_subplot(2, 4 , 1)  # equivalent but more general
for i, a in enumerate([LL, LH, HL, HH]):
    ax = fig.add_subplot(2, 4, i + 1, sharex = ax1, sharey = ax1)
    ax.imshow(a<0, interpolation="nearest", cmap=plt.cm.gray)
    ax.set_title(titles[i], fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
ax = fig.add_subplot(2, 4 , 5, sharex = ax1, sharey = ax1) 
ax.imshow(ds.spot.isel(time=0).values[::2, ::2])

fig.tight_layout()
plt.show()
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






