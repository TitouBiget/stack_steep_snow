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
# import gis_tools as gs
import xdem as xdem
import os
import pytopocomplexity as tc


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

def calculate_CWTMexHat(pathin, pathout = './cache_explore/', replace = True, lambdaa = 4):
    
    if not os.path.exists('./cache_explore/'):
        os.makedirs('./cache_explore/')
    
    f = os.path.join(pathout, pathin.split('.')[0].split('/')[-1]) + 'TC' + 'CWTMexHat' + '.tif'
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

def calculate_FracD(pathin, pathout = './cache_explore/', replace = True, ws = 4):
    
    if not os.path.exists('./cache_explore/'):
        os.makedirs('./cache_explore/')
    
    f = os.path.join(pathout, pathin.split('.')[0].split('/')[-1]) + 'TC' + 'FracD' + '.tif'
    if  not os.path.exists(f) or replace == True:
        fa = tc.FracD(window_size=10)
        Z, result = fa.analyze(pathin)
        fa.export_result(f)
    else:
        print('Feature already processed')
    with rasterio.open(f) as dataset:
        Z=dataset.read(1)
        
    Z[Z == -9999] = np.nan
    return Z



def calculate_RugosityIndex(pathin, pathout = './cache_explore/', replace = False, ws = 50):
    
    if not os.path.exists('./cache_explore/'):
        os.makedirs('./cache_explore/')
    
    f = os.path.join(pathout, pathin.split('.')[0].split('/')[-1]) + 'TC' + 'RugosityIndex' + '.tif'
    if  not os.path.exists(f) or replace == True:
        ri = tc.FracD(window_size=ws)
        Z, result = ri.analyze(pathin)
        ri.export_result(f)
    else:
        print('Feature already processed')
    with rasterio.open(f) as dataset:
        Z=dataset.read(1)
        
    Z[Z == -9999] = np.nan
    return Z


def calculate_TPI(pathin, pathout = './cache_explore/', replace = True, ws = 20):
    
    if not os.path.exists('./cache_explore/'):
        os.makedirs('./cache_explore/')
    
    f = os.path.join(pathout, pathin.split('.')[0].split('/')[-1]) + 'TC' + 'TPI' + '.tif'
    f_abs = os.path.join(pathout, pathin.split('.')[0].split('/')[-1]) + 'TC' + 'TPI_abs' + '.tif'
    if  not os.path.exists(f) or replace == True:
        tpi = tc.TPI(window_size=ws)
        Z, TPI, TPIabs, window_m = tpi.analyze(pathin)
        tpi.export_result(f, f_abs)
    else:
        print('Feature already processed')
    with rasterio.open(f) as dataset:
        Z=dataset.read(1)
        
    Z[Z == -9999] = np.nan
    return Z


    

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


def compute_dem_param(dem_file, params=['slope', 'aspect', 'svf', 'gcurv', 'TRI', 'TPI', 'RugosityIndex', 'FracD', 'CWTMexHat'] ):
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

    dem = xdem.DEM(dem_file)
    slope_horn = xdem.terrain.slope(dem)
    slope_zevenberg = xdem.terrain.slope(dem, method="ZevenbergThorne")
    diff_slope = slope_horn - slope_zevenberg
    
    aspect_horn = xdem.terrain.aspect(dem)
    aspect_zevenberg = xdem.terrain.aspect(dem, method="ZevenbergThorne")

    diff_aspect = aspect_horn - aspect_zevenberg
    diff_aspect_mod = np.minimum(diff_aspect % 360, 360 - diff_aspect % 360)
    
    
    curv = xdem.terrain.curvature(dem)
    TPI = xdem.terrain.topographic_position_index(dem)
    TRI = xdem.terrain.terrain_ruggedness_index(dem)
    roughness = xdem.terrain.roughness(dem)
    rugosity = xdem.terrain.rugosity(dem)
    FR = xdem.terrain.fractal_roughness(dem)
    
    prcurv = xdem.terrain.profile_curvature(dem)
    plcurv = xdem.terrain.planform_curvature(dem)
    
    ds['slope_horn'] = (["y", "x"], slope_horn.data.data)
    ds['slope_zevenberg'] = (["y", "x"], slope_zevenberg.data.data)
    ds['diff_slope'] = (["y", "x"], diff_slope.data.data)
    ds['aspect_horn'] = (["y", "x"], aspect_horn.data.data)
    ds['aspect_zevenberg'] = (["y", "x"], aspect_zevenberg.data.data)
    ds['diff_aspect'] = (["y", "x"], diff_aspect.data.data)
    ds['diff_aspect_mod'] = (["y", "x"], diff_aspect_mod.data.data)
    ds['curv'] = (["y", "x"], curv.data.data)
    ds['prcurv'] = (["y", "x"], prcurv.data.data)
    ds['plcurv'] = (["y", "x"], plcurv.data.data)   
    ds['TPI'] = (["y", "x"], TPI.data.data)
    ds['TRI'] = (["y", "x"], TRI.data.data)
    ds['roughness'] = (["y", "x"], roughness.data.data)
    ds['rugosity'] = (["y", "x"], rugosity.data.data)
    ds['FR'] = (["y", "x"], FR.data.data)
              

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

tmp = compute_dem_param('/home/bigett/Bureau/03_verte/dem_03_verte.tif')#, params=['slope', 'aspect', 'gcurv', 'TRI'] )


ds['elevation'] = (('y','x'), tmp.elevation.values)
ds['slope_horn'] = (('y','x'), tmp.slope_horn.values)
ds['slope_zevenberg'] = (('y','x'), tmp.slope_zevenberg.values)
ds['diff_slope'] = (["y", "x"], tmp.diff_slope.values)
ds['aspect_horn'] = (('y','x'), tmp.aspect_horn.values)
ds['aspect_zevenberg'] = (('y','x'), tmp.aspect_zevenberg.values)
ds['diff_aspect'] = (["y", "x"], tmp.diff_aspect.values)
ds['diff_aspect_mod'] = (["y", "x"], tmp.diff_aspect_mod.values)
ds['curv'] = (["y", "x"], tmp.curv.values)
ds['plcurv'] = (["y", "x"], tmp.plcurv.values)
ds['prcurv'] = (["y", "x"], tmp.prcurv.values)
ds['TPI'] = (["y", "x"], tmp.TPI.values)
ds['TRI'] = (["y", "x"], tmp.TRI.values)
ds['roughness'] = (["y", "x"], tmp.roughness.values)
ds['rugosity'] = (["y", "x"], tmp.rugosity.values)
ds['FR'] = (["y", "x"], tmp.FR.values)

tmp = None


tmp50 = compute_dem_param('/home/bigett/Bureau/03_verte/dem_03_verte_smooth_50m.tif', params=['slope', 'aspect'] )

ds['elevation_50m'] = (('y','x'), tmp50.elevation.values)
ds['slope_horn_50m'] = (('y','x'), tmp50.slope_horn.values)
ds['slope_zevenberg_50m'] = (('y','x'), tmp50.slope_zevenberg.values)
ds['diff_slope_50m'] = (["y", "x"], tmp50.diff_slope.values)
ds['aspect_horn_50m'] = (('y','x'), tmp50.aspect_horn.values)
ds['aspect_zevenberg_50m'] = (('y','x'), tmp50.aspect_zevenberg.values)
ds['diff_aspect_50m'] = (["y", "x"], tmp50.diff_aspect.values)
ds['diff_aspect_mod_50m'] = (["y", "x"], tmp50.diff_aspect_mod.values)
ds['curv_50m'] = (["y", "x"], tmp50.curv.values)
ds['plcurv_50m'] = (["y", "x"], tmp50.plcurv.values)
ds['prcurv_50m'] = (["y", "x"], tmp50.prcurv.values)
ds['TPI_50m'] = (["y", "x"], tmp50.TPI.values)
ds['TRI_50m'] = (["y", "x"], tmp50.TRI.values)
ds['roughness_50m'] = (["y", "x"], tmp50.roughness.values)
ds['rugosity_50m'] = (["y", "x"], tmp50.rugosity.values)
ds['FR_50m'] = (["y", "x"], tmp50.FR.values)

tmp50 = None


 
tmp2 = xr.open_dataset('/home/bigett/Bureau/03_verte/mask_AOI_03_verte.tif', engine='rasterio')
ds['mask'] = tmp2.band_data.isel(band=0) >= 1


# extract smoothed DEM
car_length = [2,4,8,16]
kernels = [(lambda x: func.kernel_square(x))(x) for x in car_length]

for i, kernel in enumerate(kernels):
    ds[f'smooth_{int(car_length[i]*1.5)}'] = (("y", "x"),func.smooth(ds.elevation.values, kernel))


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

ds.to_netcdf("stack_xdem.nc")
