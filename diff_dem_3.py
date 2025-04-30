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

# import pytopocomplexity as tc
# import xdem
# from osgeo import gdal
# import numpy as np
# import rasterio
# import os
from matplotlib.widgets import MultiCursor
from tqdm import tqdm
import pandas as pd
import rioxarray


ds = xr.open_dataset("stack_xdem.nc", engine="netcdf4")
# ds = xr.open_dataset("stack_xdem_sent2.nc", engine="netcdf4")

# ds['diff_elev'] = ds['elevation'] - ds['elevation_50m']
slope50 = (ds.slope_horn_50m.values > 45) & ds.mask

# df = ds.to_dataframe()


ds['diff_elev'] = ds['elevation'] - ds['elevation_50m']

# %%
def Confusion_matrix(snow_obs_sample, snow_pred_sample, plot = True, return_size = False, fscore = False):
    arr= snow_obs_sample.astype('int') - snow_pred_sample.astype('int') 
    arr2 = snow_obs_sample.astype('int') + snow_pred_sample.astype('int') 
    lim = snow_obs_sample >= 0
    size = sum(sum(lim))
    
    arr_22 = sum(sum((arr2 == 0) & lim))/size
    arr_11 = sum(sum((arr2 == 2) & lim))/size
    
    arr_21 = sum(sum((arr == 1) & lim))/size
    arr_12 = sum(sum((arr == -1) & lim))/size
    
    
    conf_matrix = np.array([[arr_11, arr_21] , [arr_12, arr_22]])
    
    if plot == True:
        plt.figure()
        plt.title(f'Total: {size}, prediction right: {arr_11 + arr_22:.3}, prediction false: {arr_12 + arr_21:.3}')
        plt.imshow(conf_matrix)
        plt.xticks([0,1], ['True', 'False'])
        plt.yticks([0,1], ['True', 'False'])
        plt.ylabel('Predicted')
        plt.xlabel('Observed')
        plt.tight_layout()
        plt.text(0, 0 , s = f'{arr_11:.0%}', horizontalalignment='center', verticalalignment='center', fontsize = 15)
        plt.text(0, 1 , s = f'{arr_12:.0%}', horizontalalignment='center', verticalalignment='center', fontsize = 15)
        plt.text(1, 0 , s = f'{arr_21:.0%}', horizontalalignment='center', verticalalignment='center', fontsize = 15)
        plt.text(1, 1 , s = f'{arr_22:.0%}', horizontalalignment='center', verticalalignment='center', fontsize = 15)
    if (return_size == True) & (fscore == False):
        return conf_matrix, size
    elif (fscore == True) & (return_size == False):
        return conf_matrix, (2*arr_11*size/(size*(2*arr_11+arr_21+arr_12)))
    elif (fscore == True) & (return_size == True):
        return conf_matrix, size,(2*arr_11*size/(size*(2*arr_11+arr_21+arr_12)))
    else:
        return conf_matrix

def normalize_array(arr, max_val = 255):
    a1 = arr + abs(np.nanmin(arr))
    a2 = (a1/np.nanmax(a1))*max_val
    return a2.astype('int8')

def perf(obs, pred, mode = 'fscore'):
    if mode == 'fscore':
        arr= obs.astype('int') - pred.astype('int') 
        arr2 = obs.astype('int') + pred.astype('int') 
        lim = obs >= 0
        size = sum(sum(lim))
        
        arr_11 = sum(sum((arr2 == 2) & lim))/size
        
        arr_21 = sum(sum((arr == 1) & lim))/size
        arr_12 = sum(sum((arr == -1) & lim))/size
        return (2*arr_11*size/(size*(2*arr_11+arr_21+arr_12)))
    else:
        arr = (obs == pred)
        size = len(arr)* len(arr[0])
        return (sum(sum(arr))/size)

def submatrix(mat, lim = (0,0,0,0)):
    xmin, xmax, ymin, ymax = lim
    return mat[xmin:xmax, ymin:ymax]

def x_diff(mat,  h):
    diff = np.zeros_like(mat)
    diff[:, 1:-1] = (mat [:, :-2] - mat [:, 2:])/(2*h)
    return diff

def y_diff(mat,  h):
    diff = np.zeros_like(mat)
    diff[1:-1, :] = (mat [:-2, :] - mat [2:, :])/(2*h)
    return diff

ds['x_diff'] = (('y','x'), x_diff(ds.diff_elev.values, 1.5))
ds['y_diff'] = (('y','x'), y_diff(ds.diff_elev.values, 1.5))
ds['x_diff_2'] = (('y','x'), x_diff(ds.x_diff.values, 1.5))
ds['y_diff_2'] = (('y','x'), y_diff(ds.y_diff.values, 1.5))
ds['x_diff_y_diff'] = (('y','x'), y_diff(ds.x_diff.values, 1.5))



if 0:
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2,3 , sharex = 1, sharey = 1)
        
    ax1.imshow(ds.x_diff, vmin = -.5, vmax = .5)
    ax4.imshow(ds.y_diff, vmin = -.5, vmax = .5)
        
    ax2.imshow(ds.x_diff_2, vmin = -.2, vmax = .2)
    ax5.imshow(ds.y_diff_2, vmin = -.2, vmax = .2)
    
    ax3.imshow(ds.x_diff_y_diff, vmin = -.2, vmax = .2) ######### top 1    
    ax6.imshow(ds.spot.isel(time = 2), cmap = 'grey')
    
    ax6.set_title('PLE 2021_05_23')
    ax3.set_title('d/dxdy')
    
    ax1.set_title('d/dx')
    ax4.set_title('d/dy')
    
    ax2.set_title('d/dx²')
    ax5.set_title('d/dy²')
    
    cursor = MultiCursor(fig.canvas, (ax1, ax2, ax3, ax4, ax5, ax6), color='cyan',lw=1, horizOn=True, vertOn=True)
    plt.show()

# %%

x_value = (ds.x_diff.values) * (np.sin(ds.aspect_horn.values*np.pi/180)**2) ######### top 1 for spot 7
y_value = (ds.y_diff.values) * (np.cos(ds.aspect_horn.values*np.pi/180)**2) ######### top 1 for spot 7

mat2= ((x_value + y_value)< .6) & ((x_value + y_value)> -1) & slope50

if True:
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2,3 , sharex = 1, sharey = 1)
    
    
    # x_value = (ds.x_diff.values) * (np.sin(ds.aspect_horn.values*np.pi/180))
    # y_value = (ds.y_diff.values) * (np.cos(ds.aspect_horn.values*np.pi/180))
    
    
    # mat1 = ((x_value**2 + y_value**2)**0.5 <.58) & slope50 #.30
        
    
    
    ax2.imshow(x_value, vmin = -.5, vmax = .5)
    ax5.imshow(y_value, vmin = -.5, vmax = .5)
    
    
    # mat2= ((x_value + y_value)< 0.112) & ((x_value + y_value)> -.58) & slope50 ######### top 1 for spot 7
    
    
    ax1.imshow(ds.x_diff, vmin = -.5, vmax = .5)
    # ax2.imshow(ds.snow_patches_2>1)
    # ax5.imshow(mat2, vmin = -.5, vmax = .5)
    ax4.imshow(ds.y_diff, vmin = -.5, vmax = .5)
    
    ax3.imshow((mat2).astype('int') - (~slope50)) ######### top 1
    
    ax3.set_title('prediction') ######### top 1
    
    
    ax6.imshow(ds.spot.isel(time = 1), cmap = 'grey')
    ax6.set_title('PLE 2021_05_23')

    
    # ax6.imshow(ds.spot.isel(time = 0), cmap = 'managua')
    
    ax1.set_title('x_value')
    ax4.set_title('y_value')
    
    ax2.set_title("d/dx")
    ax5.set_title("d/dy")
    
    # ax1.grid(1)
    # ax2.grid(1)
    # ax3.grid(1)
    # ax4.grid(1)
    # ax5.grid(1)
    # ax6.grid(1)
    
    cursor = MultiCursor(fig.canvas, (ax1, ax2, ax3, ax4, ax5, ax6), color='cyan',lw=1, horizOn=True, vertOn=True)
    plt.tight_layout()
    plt.show()
    

zone1 = (700,950,1250,1500) #[700:950, 1250:1500]
zone2 = (780,975, 780,1080) #[780:975, 780:1080]
zone3 = (950,1060,1820,2220) #[950:1060, 1820:2220]
zone4 = (1050,1200, 450,550) #[1050:1200, 450:550]

mat_zeros = np.zeros_like(mat2)

mat_zeros[700:950, 1250:1500] = 1
mat_zeros[780:975, 780:1080] = 1
mat_zeros[950:1060, 1820:2220] = 1
mat_zeros[1050:1200, 450:550] = 1

ax3.imshow(mat_zeros, alpha = 0.4, cmap = 'grey') ######### top 1


# c1 = Confusion_matrix( submatrix(mat2.values, zone1), submatrix(ds.snow_patches_2.values>0, zone1), 1)
p1 = perf( submatrix(mat2.values, zone1), submatrix(ds.snow_patches_2.values>0, zone1))
rd1 = perf( submatrix((np.random.rand(2250,2250) > 0.5).astype(int), zone1), submatrix(ds.snow_patches_2.values>0, zone1))
one1 = perf( submatrix(np.ones((2250,2250)), zone1), submatrix(ds.snow_patches_2.values>0, zone1))
zero1 = perf( submatrix(np.zeros((2250,2250)), zone1), submatrix(ds.snow_patches_2.values>0, zone1))

# c2 = Confusion_matrix( submatrix(mat2.values, zone2), submatrix(ds.snow_patches_2.values>0, zone2), 1)
p2 = perf( submatrix(mat2.values, zone2), submatrix(ds.snow_patches_2.values>0, zone2))
rd2 = perf( submatrix(((np.random.rand(2195,2320) > 0.5).astype(int)), zone2), submatrix(ds.snow_patches_2.values>0, zone2))
one2 = perf( submatrix(np.ones((2250,2250)), zone2), submatrix(ds.snow_patches_2.values>0, zone2))
zero2 = perf( submatrix(np.zeros((2250,2250)), zone2), submatrix(ds.snow_patches_2.values>0, zone2))

# c3 = Confusion_matrix( submatrix(mat2.values, zone3), submatrix(ds.snow_patches_2.values>0, zone3), 1)
p3 = perf( submatrix(mat2.values, zone3), submatrix(ds.snow_patches_2.values>0, zone3))
rd3 = perf( submatrix(((np.random.rand(3200,3000) > 0.5).astype(int)), zone3), submatrix(ds.snow_patches_2.values>0, zone3))
one3 = perf( submatrix(np.ones((3200,3200)), zone3), submatrix(ds.snow_patches_2.values>0, zone3))
zero3 =  perf( submatrix(np.zeros((3200,3200)), zone3), submatrix(ds.snow_patches_2.values>0, zone3))

# c4 = Confusion_matrix( submatrix(mat2.values, zone4), submatrix(ds.snow_patches_2.values>0, zone4), 1)
p4 = perf( submatrix(mat2.values, zone4), submatrix(ds.snow_patches_2.values>0, zone4))
rd4 = perf( submatrix(((np.random.rand(3200,3000) > 0.5).astype(int)), zone3), submatrix(ds.snow_patches_2.values>0, zone3))
one4 = perf( submatrix(np.ones((3200,3200)), zone4), submatrix(ds.snow_patches_2.values>0, zone4))
zero4 =  perf( submatrix(np.zeros((3200,3200)), zone4), submatrix(ds.snow_patches_2.values>0, zone4))


# avg1 = (c1[0,1] + c2[0,1] + c3[0,1])/3
# avg2 = (c1[1,0] + c2[1,0] + c3[1,0])/3

# print(avg1 + avg2)

print('perf ratio:')
print("Zone 1: p1 ", p1, " rd1 ", rd1, " one1 ", one1, " zero1 ", zero1)
print("Zone 2: p2 ", p2, " rd2 ", rd2, " one1 ", one2, " zero2 ", zero2)
print("Zone 3: p3 ", p3, " rd3 ", rd3, " one3 ", one3, " zero3 ", zero3)
print("Zone 4: p4 ", p4, " rd4 ", rd4, " one4 ", one4, " zero4 ", zero4)
# %%

ds['ple_corr_score'] =  (["y", "x"], rioxarray.open_rasterio('/home/bigett/Bureau/images_pleiades/formater/DSM/A1_dsm_quality_correlation_score2154_1,5m_aligned.tif').to_dataset('band')[1].values)

ds['ple_dsm'] =  (["y", "x"], rioxarray.open_rasterio('/home/bigett/Bureau/images_pleiades/formater/DSM/A2_dsm_denoised_2154_1,5m_aligned.tif').to_dataset('band')[1].values)
# %%
corr_perf = []
list_size = []
list_tpi = []
list_rock_obs = []
list_rock_pred = []
list_snow_obs = []
list_snow_pred = []
list_fscore = []

list_corr = np.arange(0, 254, 2)
for th_corr in tqdm(list_corr):
    condi_asp = (ds.aspect_horn_50m.values < 225) & (ds.aspect_horn_50m.values > 135)
    condi_calc = (100 *( ds.ple_corr_score.values <= th_corr) + ( 100* ~slope50.values) + (100 * ~condi_asp))
    
    pred_gene = mat2.values.astype('int') - condi_calc
    obs_gene = (ds.snow_patches_1.values>0).astype('int') - condi_calc
    
    c, size, f = Confusion_matrix( pred_gene, obs_gene, 0, True, True)
    
    tot_tpi = abs(np.nansum(np.nansum((condi_calc < 100).astype('int') * ds.TPI.values)))
    
    corr_perf.append( c[0,0] + c[1,1])
    list_size.append(size)
    list_tpi.append(tot_tpi/size)
    list_rock_obs.append(sum(sum(obs_gene==0)))
    list_rock_pred.append(sum(sum(pred_gene==0)))
    list_snow_obs.append(sum(sum(obs_gene==1)))
    list_snow_pred.append(sum(sum(pred_gene==1)))
    list_fscore.append(f)


arr = np.array([corr_perf, list_size, list_tpi, list_rock_obs, list_rock_pred, list_snow_obs, list_snow_pred, list_fscore])
arr = arr.T
df_perf = pd.DataFrame(arr, index = list_corr)
plt.plot(df_perf[7], 'b')
plt.xlabel('Minimum quality correlation score')
plt.yticks(color = 'b')
plt.ylabel('Prediction score', color = 'b')
plt.twinx()
plt.plot(df_perf[2], 'r')
plt.yticks(color = 'r')
plt.ylabel('Sample TPI', color = 'r')
plt.tight_layout()

plt.figure()
plt.plot(df_perf[3]/df_perf[5], 'b')
plt.xlabel('Minimum quality correlation score')
plt.yticks(color = 'b')
plt.ylabel('obs rock/snow', color = 'b')
plt.twinx()
plt.plot(df_perf[4]/df_perf[6], 'r')
plt.yticks(color = 'r')
plt.ylabel('pre rock/snow', color = 'r')
plt.tight_layout()
# %%



th_corr = 200
condi_asp = (ds.aspect_horn_50m.values < 225) & (ds.aspect_horn_50m.values > 135)
condi_calc = (100 *( ds.ple_corr_score.values <= th_corr) + ( 100* ~slope50.values) + (100 * ~condi_asp))

pred_gene = mat2.values.astype('int') - condi_calc
obs_gene = (ds.snow_patches_1.values>0).astype('int') - condi_calc

if 1:
    arr= obs_gene.astype('int') - pred_gene.astype('int') 
    arr2 = obs_gene.astype('int') + pred_gene.astype('int') 
    lim = obs_gene >= 0
    size = sum(sum(lim))
    
    arr_22 = ((arr2 == 0) & lim).astype('int') * 1
    arr_11 = ((arr2 == 2) & lim).astype('int') * 2
    
    arr_21 = ((arr == 1) & lim).astype('int') * 3
    arr_12 = ((arr == -1) & lim).astype('int') * 4
    

    
    

conf_map = arr_22 + arr_11 + arr_21 + arr_12 

fig, ((ax2, ax1), (ax3, ax4)) = plt.subplots(2, 2, sharex = 1, sharey = 1)
bax = ax2.imshow(ds.ple_corr_score.values)
cax = ax1.imshow(conf_map, cmap='gnuplot')
cb = fig.colorbar(cax, ticks = [0,1,2,3,4], shrink = 0.7)
cb.ax.set_yticklabels(['Nan','FF','TT','FT', 'TF'])

ax3.imshow(pred_gene, vmin = -1)
ax3.set_title('prediction')

ax4.imshow(obs_gene, vmin = -1)
ax4.set_title('obs')

cursor = MultiCursor(fig.canvas, (ax1, ax2, ax3, ax4), color='cyan',lw=1, horizOn=True, vertOn=True)

plt.show()
# %%
df = ds[['snow_patches_2', 'rocks_2', 'elevation','slope_horn', 'slope_horn_50m', 'aspect_horn',
         'aspect_horn_50m', 'prcurv', 'plcurv', 'x_diff', 'y_diff']].to_dataframe()


df_sud = df[(df['aspect_horn_50m']>135) & (df['aspect_horn_50m']<225) ]

df_sud_patches = df_sud.groupby(df_sud.snow_patches_2).mean()
df_sud_patches_sizes = df_sud.groupby(df_sud.snow_patches_2).count()['rocks_2'].values
df_sud_patches['patch_size'] = df_sud_patches_sizes

# %%

if True:
    list_prcurv= []
    list_plcurv= []
    list_xdiff = []
    list_ydiff = []
    list
    for patch in tqdm(df_sud_patches.index.values):    
        list_prcurv.append(df_sud[df_sud['snow_patches_2'] == patch]['prcurv'].values.mean())
        list_plcurv.append(df_sud[df_sud['snow_patches_2'] == patch]['plcurv'].values.mean())
        list_xdiff.append(abs(df_sud[df_sud['snow_patches_2'] == patch]['x_diff'].values).max())
        list_ydiff.append(abs(df_sud[df_sud['snow_patches_2'] == patch]['y_diff'].values).max())

    df_mean_curv_sud = pd.DataFrame(zip(df_sud_patches.index.values, list_prcurv, list_plcurv, list_xdiff, list_ydiff))
    df_mean_curv_sud.to_csv('df_mean_curv_sud_2025_04_07.csv')
# %%


plt.scatter(df_mean_curv_sud[3], df_mean_curv_sud[1], color = 'red')
plt.scatter(df_mean_curv_sud[3], df_mean_curv_sud[2], color = 'blue')

# plt.hist(df_mean_curv_sud[2]/ df_mean_curv_sud[1], bins = 1000)

# %%



if True:
    list_slope= []
    for patch in tqdm(df_sud_patches.index.values):
        num = 0
        den = 0
        
        for slope in df_sud[df_sud['snow_patches_2'] == patch]['slope_horn'].values:
            num = num+ (np.deg2rad(slope)*1.5/np.cos(np.deg2rad(slope)))
            den = den+ (1.5/np.cos(np.deg2rad(slope)))
        # print((num/den)*180/3.14159265)
        list_slope.append((num/den)*180/3.14159265)
    
    df_mean_slope_sud = pd.DataFrame(zip(df_sud_patches.index.values, list_slope))
    df_mean_slope_sud.to_csv('df_sud_patches_2025_04_07.csv')

# plt.hist(df_sud_patches.slope_horn.values, label = 'bedrock', color = 'blue', bins = 20)
plt.hist(df_mean_slope_sud[1].values, alpha = 0.4, label = 'snow_meth_1', color = 'yellow', bins = 20)
plt.legend()

if True:
    list_slope= []
    for patch in tqdm(df_sud_patches.index.values):
        num = 0
        den = 0
        
        for slope in df_sud[df_sud['snow_patches_2'] == patch]['slope_horn'].values:
            num = num+ 1.5*np.tan(np.deg2rad(slope))
            den = den+ 1.5
        # print((num/den)*180/3.14159265)
        list_slope.append(np.rad2deg(np.arctan(num/den)))
    
    df_mean_slope_sud = pd.DataFrame(zip(df_sud_patches.index.values, list_slope))
    # df_mean_slope_sud.to_csv('df_sud_patches_2025_04_07.csv')

# plt.hist(df_sud_patches.slope_horn.values, label = 'bedrock', color = 'blue', bins = 20)
plt.hist(df_mean_slope_sud[1].values, alpha = 0.4, label = 'snow', color = 'red', bins = 20)
plt.legend()
# %%



if True:
    list_slope= []
    for patch1 in tqdm(df_sud_patches.index.values):
        
        patch = df_sud[df_sud['snow_patches_2'] == patch1].elevation
        max_elev= np.nanmax(patch.values)
        min_elev= np.nanmin(patch.values)
        dz = max_elev - min_elev
        
        y_max, x_max = patch[patch.values == max_elev].index.values[0]
        y_min, x_min = patch[patch.values == min_elev].index.values[0]
        dl = ((y_max-y_min)**2+(x_max-x_min)**2)**0.5

        # print((num/den)*180/3.14159265)
        list_slope.append(np.rad2deg(np.arctan(dz/dl)))
    
    df_mean_slope_sud = pd.DataFrame(zip(df_sud_patches.index.values, list_slope))
    df_mean_slope_sud.to_csv('df_sud_patches_2025_04_07.csv')

plt.hist(df_sud_patches.slope_horn.values, label = 'bedrock', color = 'blue', bins = 20)
plt.hist(df_mean_slope_sud[1].values, alpha = 0.4, label = 'snow', color = 'red', bins = 20)
plt.legend()
# %%


from tqdm import tqdm

list_score = []



x_value = (ds.x_diff.values) * (np.sin(ds.aspect_horn.values*np.pi/180)**2) ######### top 1 for spot 7
y_value = (ds.y_diff.values) * (np.cos(ds.aspect_horn.values*np.pi/180)**2) ######### top 1 for spot 7

zone1 = (700,950,1250,1500) #[700:950, 1250:1500]
zone2 = (780,975,780,1100) #[780:975, 780:1100]
zone3 = (950,1060,1820,2220) #[950:1060, 1820:2220]
zone4 = (1050,1200, 450,550) #[950:1060, 1820:2220]

l = len(np.arange(0, 3, 0.1))
heatmap = np.zeros((l,l))

for i, j in enumerate(np.arange(0, 3, 0.1)):
    for m, k in enumerate(np.arange(0, 3, 0.1)):

        

        mat2= ((x_value + y_value)< j) & ((x_value + y_value)> -k) & slope50

        c1 = perf( submatrix(mat2.values, zone1), submatrix(ds.snow_patches_2.values>0, zone1))
        
        c2 = perf( submatrix(mat2.values, zone2), submatrix(ds.snow_patches_2.values>0, zone2))
        
        c3 = perf( submatrix(mat2.values, zone3), submatrix(ds.snow_patches_2.values>0, zone3))

        c4 = perf( submatrix(mat2.values, zone4), submatrix(ds.snow_patches_2.values>0, zone4))
        
        
        # print('c1', c1[0,0], c1[1,1], ' rd1', rd1[0,0], rd1[1,1])
        # print('c2', c2[0,0], c2[1,1], ' rd2', rd2[0,0], rd2[1,1])
        # print('c3', c3[0,0], c3[1,1], ' rd3', rd3[0,0], rd3[1,1])
        

        heatmap[i,m] = (c1+c2+c3+c4)/4
    print(i)

plt.imshow(heatmap, vmin = 0.8)
plt.xlabel('borne sup')
plt.ylabel('borne inf')

# plt.xticks(range(l), labels= np.arange(0, 3, 0.1))
# plt.yticks(range(l), labels= np.arange(0, 3, 0.1))
# %%
from tqdm import tqdm

list_score = []



x_value = (ds.x_diff.values) * (np.sin(ds.aspect_horn.values*np.pi/180)**2) ######### top 1 for spot 7
y_value = (ds.y_diff.values) * (np.cos(ds.aspect_horn.values*np.pi/180)**2) ######### top 1 for spot 7


condi_asp = (ds.aspect_horn_50m.values < 225) & (ds.aspect_horn_50m.values > 135)
condi_calc = (( 100* ~slope50.values) + (100 * ~condi_asp))

obs_gene = (ds.snow_patches_1.values>0).astype('int') - condi_calc    

l = len(np.arange(0, 3, 0.1))
heatmap = np.zeros((l,l)).astype('float')

for i, j in enumerate(np.arange(0, 3, 0.1)):
    for m, k in enumerate(np.arange(0, 3, 0.1)):

        

        mat2= ((x_value + y_value)< j) & ((x_value + y_value)> -k) & slope50
        pred_gene = mat2.values.astype('int') - condi_calc


        

        heatmap[i,m] = perf( pred_gene>0, obs_gene>0)
    print(i)

plt.imshow(heatmap, vmin = 0.8, cmap = 'grey')
plt.xlabel('borne sup')
plt.ylabel('borne inf')




# %%

fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3,3, sharex =1 , sharey = 1)

list_ax = np.array([[ax1, ax2, ax3], [ax4, ax5, ax6], [ax7, ax8, ax9]])

AOI = (677,744, 1393,1461) #[1393:1461, 677:744]


# mat2= ((x_value + y_value)< 0.85) & ((x_value + y_value)> -1.56) & slope50

for index_i, i in enumerate(np.arange(0.1, 1.2, 0.4)):
    for index_j, j in enumerate(np.arange(1.3, 2.4, 0.4)):
        mat2= ((x_value + y_value)< i) & ((x_value + y_value)> -j) & slope50
        list_ax[index_i, index_j].imshow(submatrix(mat2.values, AOI), cmap = 'grey')
        list_ax[index_i, index_j].set_title(f'lower lim = {-j:.2}, upper lim = {i:.2}')
        
ax5.imshow(submatrix(ds.spot.isel(time = 1).values, AOI),  cmap = 'grey')
ax5.set_title("Pleiades")

fig.supxlabel('lower lim')
fig.supylabel('upper lim')
fig.tight_layout()

cursor = MultiCursor(fig.canvas, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9), color='blue',lw=1, horizOn=True, vertOn=True)
plt.show()

# %%
path = '/home/bigett/codes/machine_learning/modele_1/data_training/'
import rasterio

for ind, zone in enumerate([zone1, zone2, zone3, zone4]):
    
    img =  submatrix(ds.snow_patches_2.values>0,zone).astype('int')*255
    with rasterio.open(path +'snow'+ str(ind)  + '.tif', 'w', width = len(img[0]), height = len(img), count = 1, dtype = 'int8') as dst:
        dst.write(img, 1)
        img2 =  submatrix(x_value + y_value,zone).astype('int')
    
    img1 = normalize_array(submatrix(ds.diff_elev.values,zone).astype('int')) #high_pass_elev
    img2 = normalize_array(submatrix(x_value + y_value,zone).astype('int')) # x_values + y_values
    img3 = normalize_array(submatrix(ds.slope_zevenberg.values,zone).astype('int'))
    img4 = normalize_array(submatrix(ds.TPI.values,zone).astype('int'))
    img5 = normalize_array(submatrix(ds.curv.values,zone).astype('int'))
    img6 = normalize_array(submatrix(ds.roughness.values,zone).astype('int'))
    img7 = normalize_array(submatrix(ds.rugosity.values,zone).astype('int'))
    img8 = normalize_array(submatrix(ds.FR.values,zone).astype('int'))


    with rasterio.open(path +'stack'+ str(ind)  + '.tif', 'w', width = len(img[0]), height = len(img), count = 8, dtype = 'int8') as dst:
        dst.write(img1, 1)
        dst.write(img2, 2)
        dst.write(img3, 3)
        dst.write(img4, 4)
        dst.write(img5, 5)
        dst.write(img6, 6)
        dst.write(img7, 7)
        dst.write(img8, 8)

zone_val_1 = (1182, 1337, 149, 293)
zone_val_2 = (1185, 1410, 2750, 3070)
zone_val_2 = (1151, 1433, 2746, 3076)

path = '/home/bigett/codes/machine_learning/modele_1/data_validation/'

for ind, zone in enumerate([zone_val_1, zone_val_2]):
    
    img =  (submatrix(ds.spot.isel(time = 2).values, zone).astype('int')/np.nanmax(submatrix(ds.spot.isel(time = 2).values, zone).astype('int')))*255
    with rasterio.open(path +'ple'+ str(ind)  + '.tif', 'w', width = len(img[0]), height = len(img), count = 1, dtype = 'int8') as dst:
        dst.write(img, 1)
    
    img1 = normalize_array(submatrix(ds.diff_elev.values,zone).astype('int')) #high_pass_elev
    img2 = normalize_array(submatrix(x_value + y_value,zone).astype('int')) # x_values + y_values
    img3 = normalize_array(submatrix(ds.slope_zevenberg.values,zone).astype('int'))
    img4 = normalize_array(submatrix(ds.TPI.values,zone).astype('int'))
    img5 = normalize_array(submatrix(ds.curv.values,zone).astype('int'))
    img6 = normalize_array(submatrix(ds.roughness.values,zone).astype('int'))
    img7 = normalize_array(submatrix(ds.rugosity.values,zone).astype('int'))
    img8 = normalize_array(submatrix(ds.FR.values,zone).astype('int'))
    with rasterio.open(path +'stack'+ str(ind)  + '.tif', 'w', width = len(img[0]), height = len(img), count = 8, dtype = 'int8') as dst:
        dst.write(img1, 1)
        dst.write(img2, 2)
        dst.write(img3, 3)
        dst.write(img4, 4)
        dst.write(img5, 5)
        dst.write(img6, 6)
        dst.write(img7, 7)
        dst.write(img8, 8)
        
        
# %%