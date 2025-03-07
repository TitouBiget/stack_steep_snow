"""
Script to explore data on Tacul slopes
S. Filhol & T. Biget, Feb 2025

0. combine all rasters into a dataset 
"""

import numpy as np 
import xarray as xr 
import rioxarray
import matplotlib.pyplot as plt 
import glob 
import pandas as pd 
import gis_tools as gs
import richdem as rd
import os


import func

from osgeo import gdal
import numpy as np
import rasterio


def calculate_slope(pathin, pathout = './cache_explore/'):
    
    if not os.path.exists('./cache_explore/'):
        os.makedirs('./cache_explore/')
    
    f = os.path.join(pathout, pathin.split('.')[0].split('/')[-1]) + '_slope.tif'
    if  not os.path.exists(f):
        gdal.DEMProcessing(f, pathin, 'slope')
    else:
        print('Feature already processed')
    with rasterio.open(f) as dataset:
        slope=dataset.read(1)
    
    slope[slope == -9999] = np.nan
    return slope



# slope = calculate_slope('/home/bigett/Bureau/03_verte/dem_crop_filled.tif')

def calculate_aspect(pathin, pathout = './cache_explore/'):
    
    if not os.path.exists('./cache_explore/'):
        os.makedirs('./cache_explore/')
    
    f = os.path.join(pathout, pathin.split('.')[0].split('/')[-1]) + '_aspect.tif'
    if  not os.path.exists(f):
        gdal.DEMProcessing(f, pathin, 'aspect')
    else:
        print('Feature already processed')
    with rasterio.open(f) as dataset:
        aspect=dataset.read(1)
    
    aspect[aspect == -9999] = np.nan
    return aspect



def calculate_TRI(pathin, pathout = './cache_explore/'):
    
    if not os.path.exists('./cache_explore/'):
        os.makedirs('./cache_explore/')
    
    f = os.path.join(pathout, pathin.split('.')[0].split('/')[-1]) + '_TRI.tif'
    if  not os.path.exists(f):
        gdal.DEMProcessing(f, pathin, 'TRI')
    else:
        print('Feature already processed')
    with rasterio.open(f) as dataset:
        TRI=dataset.read(1)
    
    TRI[TRI == -9999] = np.nan
    return TRI


def gaussian_curvature(Z, neighbor=None):
    '''
    **K = gaussian_curvature(Z)**\n
    Function to calculate the gaussian curvature of a 2D matrix\n
    see  http://en.wikipedia.org/wiki/Gaussian_curvature 
        
    Parameters
    ==========
    **Z** - 2D matrix of elevation
    **neighbor** - number of neighboring pixel to consider (see numpy.gradient() function help)
    
    Returns
    =======
    **K** - 2D matrix of curvature value
    '''
    if neighbor is None:
        neighbor=1
    Zy, Zx = np.gradient(Z,neighbor)                                                     
    Zxy, Zxx = np.gradient(Zx,neighbor)                                                  
    Zyy, _ = np.gradient(Zy)                                                    
    K = (Zxx * Zyy - (Zxy ** 2)) / (1 + (Zx ** 2) + (Zy **2)) ** 2
    return K
    


# Combine rasters to dataset


def compute_dem_param(dem_file, params=['slope', 'aspect', 'svf', 'gcurv', 'TRI'] ):
    """
    Function to compute and derive DEM parameters: slope, aspect, sky view factor

    Args:
        dem_file (str): path to raster file (geotif). Raster must be in local cartesian coordinate system (e.g. UTM)

    Returns:
        dataset: x, y, elev, slope, aspect, svf


    """   
    print(f"\n---> Extracting DEM parameters ({', '.join(params)})")
    ds = rioxarray.open_rasterio(dem_file).to_dataset('band')
    ds = ds.rename({1: 'elevation'})
    # dx = ds.x.diff('x').median().values
    # dy = ds.y.diff('y').median().values

    print('Computing slope and aspect ...')

    if 'slope' or 'aspect' in params:
        slope, aspect = calculate_slope(dem_file), calculate_aspect(dem_file)

        if 'slope' in params:
            ds['slope'] = (["y", "x"], np.flip(slope,0))
            ds.slope.attrs = {'units': 'rad'}

        if 'aspect' in params:
            aspect = np.flip(aspect, 0)
            ds['aspect'] = (["y", "x"], np.deg2rad(aspect))
            ds['aspect_cos'] = (["y", "x"], np.cos(np.deg2rad(aspect)))
            ds['aspect_sin'] = (["y", "x"], np.sin(np.deg2rad(aspect)))
            ds.aspect.attrs = {'units': 'rad'}
            ds.aspect_cos.attrs = {'units': 'cosinus'}
            ds.aspect_sin.attrs = {'units': 'sinus'}
        
        if 'gcurv' in params:
            gcurv = gaussian_curvature(ds.elevation.values)
            ds['gcurv'] = (["y", "x"], gcurv)

        if 'TRI' in params:
            TRI = calculate_TRI(dem_file)
            ds['TRI'] = (["y", "x"], TRI)

    # if 'svf' in params:
    #     print('Computing svf ...')
    #     svf = viewf.viewf(np.double(dem_arr), dx)[0]
    #     ds['svf'] = (["y", "x"], svf)
    #     ds.svf.attrs = {'units': 'ratio', 'standard_name': 'svf', 'long_name': 'Sky view factor'}

    ds.attrs = dict(description="DEM input parameters to TopoSub",
                   author="mettools, https://github.com/ArcticSnow/mettools")
    ds.x.attrs = {'units': 'm'}
    ds.y.attrs = {'units': 'm'}
    ds.elevation.attrs = {'units': 'm'}

    return ds


flist = glob.glob('/home/bigett/Bureau/03_verte/IMG_*_crop.tif')
date_list = [(lambda x: pd.to_datetime(x.split('/')[-1][8:16]))(x) for x in flist] 

# open and combine all spot images to one dataset
lis = []
for file in flist:
    dd = xr.open_dataset(file, engine='rasterio')
    lis.append(dd.band_data.isel(band=0))
ds= xr.concat(lis, dim="time").to_dataset()
ds['time'] = date_list
ds = ds.sortby('time')
ds = ds.rename({'band_data':'spot'})

tmp = compute_dem_param('/home/bigett/Bureau/03_verte/dem_03_verte.tif', params=['slope', 'aspect', 'gcurv', 'TRI'] )

# ds = ds.assign(elevation= tmp.elevation)

ds['elevation'] = (('y','x'), tmp.elevation.values)
ds['slope'] = (('y','x'), tmp.slope.values)
ds['aspect'] = (('y','x'), tmp.aspect.values)
ds['aspect_sin'] = (('y','x'), tmp.aspect_sin.values)
ds['aspect_cos'] = (('y','x'), tmp.aspect_cos.values)
ds['gcurv'] = (["y", "x"], tmp.gcurv.values)
ds['TRI'] = (["y", "x"], tmp.TRI.values)

#ds['svf'] = tmp.svf
tmp = None

tmp50 = compute_dem_param('/home/bigett/Bureau/03_verte/dem_03_verte_smooth_50m.tif', params=['slope', 'aspect'] )

ds['elevation_50m'] = (('y','x'), tmp50.elevation.values)
ds['slope_50m'] = (('y','x'), tmp50.slope.values)
ds['aspect_50m'] = (('y','x'), tmp50.aspect.values)
ds['aspect_sin_50m'] = (('y','x'), tmp50.aspect_sin.values)
ds['aspect_cos_50m'] = (('y','x'), tmp50.aspect_cos.values)
ds['gcurv_50m'] = (('y','x'), gaussian_curvature(tmp50.elevation.values))

#ds['svf'] = tmp.svf
tmp50 = None


 
tmp2 = xr.open_dataset('/home/bigett/Bureau/03_verte/mask_AOI_03_verte.tif', engine='rasterio')
ds['mask'] = tmp2.band_data.isel(band=0) >= 1


# extract smoothed DEM
car_length = [2,4,8,16]
kernels = [(lambda x: func.kernel_square(x))(x) for x in car_length]

for i, kernel in enumerate(kernels):
    ds[f'smooth_{int(car_length[i]*1.5)}'] = (("y", "x"),func.smooth(ds.elevation.values, kernel))


# from skimage.filters import sobel

# fig, ax = plt.subplots(1,2,sharex=True, sharey=True)
# ax[0].imshow(ds.spot.isel(time=3).where(ds.mask), interpolation='nearest', aspect='auto', cmap=plt.cm.grey)

# ax[1].imshow(ds.spot.isel(time=3).where(ds.mask)>=2000, interpolation='nearest', aspect='auto')

# plt.show()
if True:
    from skimage.filters import sobel
    from skimage.measure import label
    from skimage.segmentation import slic, join_segmentations, watershed
    from skimage.color import label2rgb
    from skimage.filters import threshold_otsu, threshold_niblack, threshold_sauvola
    
    snow = ds.spot.isel(time=2).where(ds.mask).values 
    snow = (snow/np.nanmax(snow))
    
    # Make segmentation using edge-detection and watershed.
    edges = sobel(snow)
    
    
    # Identify some background and foreground pixels from the intensity values.
    # These pixels are used as seeds for watershed.
    markers = np.zeros_like(snow*0)
    
    markers[snow <0.65] = 1
    markers[snow >0.75] = 2
    
    markers = markers.astype('int')
    
    
    elev_map = sobel(snow)
    ws = watershed(elev_map, markers)
    snow_patches = label(ws == 2)
    rocks = (ws == 1)
    
    
    # Show the segmentations.
    fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(9, 5), sharex=True, sharey=True)
    ax = axes.ravel()
    
    ax[0].imshow(edges, cmap='gray')
    # ax[0].set_title(f'sobel, th = {th}')
    ax[0].set_title('sobel, 0.65, 0.75')
    
    color1 = label2rgb(snow_patches, image=snow, bg_label=0)
    ax[1].imshow(color1)
    ax[1].set_title('Sobel+Watershed+label COLORED')
    
    ax[2].imshow(snow, cmap=plt.cm.grey)
    ax[2].set_title('Original Image')
    plt.show()
    
    ds['snow_patches'] = (('y','x'), snow_patches)
    ds['rocks'] = (('y','x'), rocks)
    
 # %%   
df = ds[['snow_patches', 'rocks','slope', 'aspect_50m', 'gcurv_50m', 'TRI']].to_dataframe()



plt.figure()

df_sud = df[(df['aspect_50m']>135*np.pi/180) & (df['aspect_50m']<225*np.pi/180)]
# plt.scatter(df_sud['slope'][df_sud['rocks'] == False], df_sud['snow_patches'][df_sud['rocks']== False])
# plt.scatter(df_sud['slope'][df_sud['rocks'] == True], df_sud['snow_patches'][df_sud['rocks'] == True])
(df_sud['slope']).hist(bins = 1000, color = 'red', label = 'total')
(df_sud['slope'][df_sud['rocks'] == True]).hist(bins = 1000,  color = 'green', label = 'rocks')
(df_sud['slope'][df_sud['rocks'] == False]).hist(bins = 1000,  color = 'blue', label = 'snow')
plt.title('Slope distribution SUD')
plt.legend()
plt.tight_layout()

plt.figure()
df_nord = df[(df['aspect_50m']>315*np.pi/180) | (df['aspect_50m']<45*np.pi/180)]
(df_nord['slope']).hist(bins = 1000, color = 'red', label = 'total')
(df_nord['slope'][df_nord['rocks'] == True]).hist(bins = 1000,  color = 'green', label = 'rocks')
(df_nord['slope'][df_nord['rocks'] == False]).hist(bins = 1000,  color = 'blue',  label = 'snow')
plt.title('Slope distribution NORD')
plt.legend()
plt.tight_layout()


# %%
 

# plt.figure()
# plt.scatter(df_nord['slope'][df_nord['rocks'] == True], df_nord['gcurv'][df_nord['rocks'] == True], alpha = 0.2)
# plt.scatter(df_nord['slope'][df_nord['rocks'] == False], df_nord['gcurv'][df_nord['rocks'] == False], alpha = 0.2)


# %%
 


plt.figure()
plt.scatter(df_nord['slope'][df_nord['rocks'] == True], df_nord['TRI'][df_nord['rocks'] == True], alpha = 0.05, label = 'rocks')
plt.scatter(df_nord['slope'][df_nord['rocks'] == False], df_nord['TRI'][df_nord['rocks'] == False], alpha = 0.05,  label = 'snow')
plt.xlabel('Slope')
plt.ylabel('Terrain Ruggedness Index')
plt.title('TRI/ Slope NORD')
plt.legend()
plt.tight_layout()


plt.figure()
plt.scatter(df_sud['slope'][df_sud['rocks'] == True], df_sud['TRI'][df_sud['rocks'] == True], alpha = 0.05, label = 'rocks')
plt.scatter(df_sud['slope'][df_sud['rocks'] == False], df_sud['TRI'][df_sud['rocks'] == False], alpha = 0.05,  label = 'snow')
plt.xlabel('Slope')
plt.ylabel('Terrain Ruggedness Index')
plt.title('TRI/ Slope  SUD')
plt.legend()
plt.tight_layout()


# %%




plt.figure()
plt.scatter(df_nord['slope'][df_nord['rocks'] == True], df_nord['gcurv_50m'][df_nord['rocks'] == True], alpha = 0.05, label = 'rocks')
plt.scatter(df_nord['slope'][df_nord['rocks'] == False], df_nord['gcurv_50m'][df_nord['rocks'] == False], alpha = 0.05,  label = 'snow')
plt.xlabel('Slope')
plt.ylabel('Gaussian curvature on 50m smoothed demp')
plt.title('gcurv_50m/ Slope  NORD')
plt.legend()
plt.tight_layout()

plt.figure()
plt.scatter(df_sud['slope'][df_sud['rocks'] == True], df_sud['gcurv_50m'][df_sud['rocks'] == True], alpha = 0.05, label = 'rocks')
plt.scatter(df_sud['slope'][df_sud['rocks'] == False], df_sud['gcurv_50m'][df_sud['rocks'] == False], alpha = 0.05,  label = 'snow')
plt.xlabel('Slope')
plt.ylabel('Gaussian curvature on 50m smoothed dem')
plt.title('gcurv_50m/ Slope SUD')
plt.legend()
plt.tight_layout()

# %%

#################################################################################################################""""""""""""
df_sud_patches = df_sud.groupby(df_sud.snow_patches).mean()

df_sud_patches_sizes = df_sud.groupby(df_sud.snow_patches).count()['rocks'].values

df_sud_patches['patch_size'] = df_sud_patches_sizes

######################################################################

df_nord_patches = df_nord.groupby(df_nord.snow_patches).mean()

df_nord_patches_sizes = df_nord.groupby(df_nord.snow_patches).count()['rocks'].values

df_nord_patches['patch_size'] = df_nord_patches_sizes
#################################################################################################################""""""""""""

# %%


plt.figure()

# plt.scatter(df_sud['slope'][df_sud['rocks'] == False], df_sud['snow_patches'][df_sud['rocks']== False])
(df_sud_patches['slope']).hist(bins = 1000, color = 'red', label = 'total')
(df_sud_patches['slope'][df_sud_patches['rocks'] == True]).hist(bins = 1000,  color = 'green', label = 'rocks')
(df_sud_patches['slope'][df_sud_patches['rocks'] == False]).hist(bins = 1000,  color = 'blue', label = 'snow')
plt.title('Patches mean slope distribution SUD')
plt.legend()
plt.tight_layout()

plt.figure()
df_nord = df[(df['aspect_50m']>315*np.pi/180) | (df['aspect_50m']<45*np.pi/180)]
(df_nord_patches['slope']).hist(bins = 1000, color = 'red', label = 'total')
(df_nord_patches['slope'][df_nord_patches['rocks'] == True]).hist(bins = 1000,  color = 'green', label = 'rocks')
(df_nord_patches['slope'][df_nord_patches['rocks'] == False]).hist(bins = 1000,  color = 'blue',  label = 'snow')
plt.title('Patches mean slope distribution NORD')
plt.legend()
plt.tight_layout()




# %%

##################à faire courbe histogramme

plt.figure()
(df_sud['slope'][df_sud['rocks'] == True]/df_sud['slope']).hist(bins = 10,  color = 'green', label = 'rocks')
(df_sud['slope'][df_sud['rocks'] == False]/df_sud['slope']).hist(bins = 10,  color = 'blue', label = 'snow')
plt.legend()
plt.tight_layout()



# %%
 


plt.figure()
plt.scatter(df_nord_patches['slope'][df_nord_patches.index.values != 0], df_nord_patches['patch_size'][df_nord_patches.index.values != 0], alpha = 0.5,  label = 'snow')
plt.xlabel('Slope')
plt.ylabel('patch size')
plt.title('patch_size/ Slope NORD')
plt.legend()
plt.tight_layout()


plt.figure()
plt.scatter(df_sud_patches['slope'][df_sud_patches.index.values != 0], df_sud_patches['patch_size'][df_sud_patches.index.values != 0], alpha = 0.5,  label = 'snow')
plt.xlabel('Slope')
plt.ylabel('patch size')
plt.title('patch_size/ Slope SUD')
plt.legend()
plt.tight_layout()


# %%




plt.figure()
plt.scatter(df_nord['slope'][df_nord['rocks'] == True], df_nord['gcurv_50m'][df_nord['rocks'] == True], alpha = 0.05, label = 'rocks')
plt.scatter(df_nord['slope'][df_nord['rocks'] == False], df_nord['gcurv_50m'][df_nord['rocks'] == False], alpha = 0.05,  label = 'snow')
plt.xlabel('Slope')
plt.ylabel('Gaussian curvature on 50m smoothed demp')
plt.title('gcurv_50m/ Slope  NORD')
plt.legend()
plt.tight_layout()

plt.figure()
plt.scatter(df_sud['slope'][df_sud['rocks'] == True], df_sud['gcurv_50m'][df_sud['rocks'] == True], alpha = 0.05, label = 'rocks')
plt.scatter(df_sud['slope'][df_sud['rocks'] == False], df_sud['gcurv_50m'][df_sud['rocks'] == False], alpha = 0.05,  label = 'snow')
plt.xlabel('Slope')
plt.ylabel('Gaussian curvature on 50m smoothed dem')
plt.title('gcurv_50m/ Slope SUD')
plt.legend()
plt.tight_layout()