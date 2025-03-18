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
import richdem as rd
import os
import pytopocomplexity as tc

import func

from osgeo import gdal
import numpy as np
import rasterio













def plot_contours_matplotlib(x, y, threshold=0.05, start=0.1, end=1, size=0.1, title=None):
    # Compute the 2D histogram
    hist, x_edges, y_edges = np.histogram2d(x, y, bins=len(x)//500)
    
    # Normalize the density values to [0, 1]
    hist_normalized = (hist - np.min(hist)) / (np.max(hist) - np.min(hist))
    
    # Compute the mask for the threshold
    mask = hist_normalized >= threshold
    
    # Find indices for x and y ranges
    x_indices = np.where(mask.any(axis=1))[0]
    y_indices = np.where(mask.any(axis=0))[0]
    xmin, xmax = x_edges[x_indices[0]], x_edges[x_indices[-1]]
    ymin, ymax = y_edges[y_indices[0]], y_edges[y_indices[-1]]
    
    # Create the contour levels
    levels = np.arange(start, end + size, size)
    
    # Plot the contour
    fig, ax = plt.subplots()
    contour = ax.contourf(x_edges[:-1], y_edges[:-1], hist_normalized.T, levels=levels, cmap='viridis')
    
    # Add contour lines
    ax.contour(x_edges[:-1], y_edges[:-1], hist_normalized.T, levels=levels, colors='black', linewidths=0)
    
    # Add colorbar
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label('Density')
    
    # Set title and labels
    ax.set_title(title if title else '')
    ax.set_xlabel(x.name)
    ax.set_ylabel(y.name)
    # ax.set_facecolor('')
    
    # Set axis limits
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    
    # Show plot
    plt.show()












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
    
def derivate(array, h):
    """
    Compute the first derivative of an array using the second-order central difference method.

    Parameters:
    array (numpy.ndarray): The input array of function values.
    h (float): The spacing between the points.

    Returns:
    numpy.ndarray: The array of first derivatives.
    """
    # Ensure the input is a numpy array
    array = np.asarray(array)
    
    # Initialize an array to store the derivatives
    derivatives = np.zeros_like(array)
    
    # Compute the derivatives using the central difference method
    for i in range(1, len(array) - 1):
        derivatives[i] = (array[i + 1] - array[i - 1]) / (2 * h)
    
    # Handle the boundaries (optional, depending on your needs)
    derivatives[0] = (array[1] - array[0]) / h
    derivatives[-1] = (array[-1] - array[-2]) / h
    
    return derivatives

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
            
        if 'TPI' in params:
            TPI = calculate_TPI(dem_file)
            ds['TPI'] = (["y", "x"], TPI)
            
        # if 'RugosityIndex' in params:
        #     RugosityIndex = calculate_RugosityIndex(dem_file)
        #     ds['RugosityIndex'] = (["y", "x"], RugosityIndex)
            
        # if 'FracD' in params:
        #     FracD = calculate_FracD(dem_file)
        #     ds['FracD'] = (["y", "x"], FracD)
            
        if 'CWTMexHat' in params:
            CWTMexHat = calculate_CWTMexHat(dem_file)
            ds['CWTMexHat'] = (["y", "x"], CWTMexHat)

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

# ds = ds.assign(elevation= tmp.elevation)

ds['elevation'] = (('y','x'), tmp.elevation.values)
ds['slope'] = (('y','x'), tmp.slope.values)
ds['aspect'] = (('y','x'), tmp.aspect.values)
ds['aspect_sin'] = (('y','x'), tmp.aspect_sin.values)
ds['aspect_cos'] = (('y','x'), tmp.aspect_cos.values)
ds['gcurv'] = (["y", "x"], tmp.gcurv.values)
ds['TRI'] = (["y", "x"], tmp.TRI.values)
ds['TPI'] = (["y", "x"], tmp.TPI.values)
# ds['RugosityIndex'] = (["y", "x"], tmp.RugosityIndex.values)
# ds['FracD'] = (["y", "x"], tmp.FracD.values)
ds['CWTMexHat'] = (["y", "x"], tmp.CWTMexHat.values)

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
plt.imshow(ds.elevation)



 # %%   skimage parameters
ds['cos_slope'] = (('y','x'), np.sin(ds.slope.values * np.pi/180) )
import skimage.filters as skif
ds['butterworth'] = (('y','x'), skif.butterworth(ds.elevation.values, order=7.0, cutoff_frequency_ratio=0.003))
ds['farid'] = (('y','x'), skif.farid(ds.elevation.values))
ds['frangi'] = (('y','x'), skif.frangi(ds.elevation.values))
# ds['gabor'] = (('y','x'), skif.gabor(ds.elevation.values, 1))
ds['hessian'] = (('y','x'), skif.hessian(ds.elevation.values))
ds['meijering'] = (('y','x'), skif.meijering(ds.elevation.values))
ds['prewitt'] = (('y','x'), skif.prewitt(ds.elevation.values))
# ds['rank_order'] = (('y','x'), skif.rank_order(ds.elevation.values))
ds['roberts'] = (('y','x'), skif.roberts(ds.elevation.values))
ds['sato'] = (('y','x'), skif.sato(ds.elevation.values))
# ds['wiener'] = (('y','x'), skif.wiener(ds.elevation.values))
# ds['LPIFilter2D'] = (('y','x'), skif.LPIFilter2D(ds.elevation.values))
ds['difference_of_gaussians'] = (('y','x'), skif.difference_of_gaussians(ds.elevation.values, 2))
# ds[''] = (('y','x'), skif.(ds.elevation.values))
# %%


df = ds[['snow_patches', 'rocks', 'elevation', 'elevation_50m','slope', 'slope_50m', 'aspect_50m', 'gcurv_50m', 
         'TRI', 'TPI', 'CWTMexHat', 'butterworth','farid', 'frangi', 'hessian','meijering', 'prewitt', 'roberts', 
         'sato', 'difference_of_gaussians']].to_dataframe()

df['cos_slope'] = np.sin(df['slope'] * np.pi/180) 

df_sud = df[(df['aspect_50m']>135*np.pi/180) & (df['aspect_50m']<225*np.pi/180) ]#& (df['slope']>50)]


df_sud_patches = df_sud.groupby(df_sud.snow_patches).mean()

df_sud_patches_sizes = df_sud.groupby(df_sud.snow_patches).count()['rocks'].values

df_sud_patches['patch_size'] = df_sud_patches_sizes
print(len(df_sud[df_sud['butterworth'] > df_sud['butterworth'][df_sud['rocks'] == 1].max()]))

# %% 

for param in ['butterworth','farid', 'frangi', 'hessian','meijering', 'prewitt', 'roberts', 'sato', 'difference_of_gaussians']:
    y = df_sud[param]
    plt.figure()
    plt.scatter(df_sud['slope'][df_sud['rocks'] == 1], (y[df_sud['rocks'] == 1]), alpha = 0.1 )
    plt.scatter(df_sud['slope'][df_sud['rocks'] == 0], (y[df_sud['rocks'] == 0]) , alpha = 0.1 )
    plt.title(param)
    plt.ylabel(param)
    plt.xlabel(param)



# %%

# for freq in np.arange(0.0060, 0.012, 0.0002):
    
ds['diff_elev'] = ds['elevation'] - ds['elevation_50m']
ds['butterworth'] = (('y','x'), skif.butterworth(ds.CWTMexHat.values, order=2.0, cutoff_frequency_ratio=0.5, high_pass= False))



df = ds[['slope','aspect_50m', 'rocks', 'butterworth']].to_dataframe()

df_sud = df[(df['aspect_50m']>135*np.pi/180) & (df['aspect_50m']<225*np.pi/180) ]#& (df['slope']>50)]

print(len(df_sud[df_sud['butterworth'] > df_sud['butterworth'][df_sud['rocks'] == 1].max()]))


plt.figure()

y = df_sud['butterworth']

plt.scatter(df_sud['slope'][df_sud['rocks'] == 1], (y[df_sud['rocks'] == 1]), alpha = 0.1 )
plt.scatter(df_sud['slope'][df_sud['rocks'] == 0], (y[df_sud['rocks'] == 0]) , alpha = 0.1 )

# %%

y = df_sud['slope_50m'] - df_sud['slope']
plt.figure()

plt.scatter(df_sud['slope'][df_sud['rocks'] == 1], (y[df_sud['rocks'] == 1]), alpha = 0.1 )
plt.scatter(df_sud['slope'][df_sud['rocks'] == 0], (y[df_sud['rocks'] == 0]) , alpha = 0.1 )



# %%

plt.figure()

y = df_sud_patches['cos_slope']
# y = df_sud_patches['elevation_50m'] - df_sud_patches['elevation']
# y = df_sud_patches['gcurv_50m'] 

plt.scatter(df_sud_patches['patch_size'][df_sud_patches['rocks'] == 0], (y[df_sud_patches['rocks'] == 0]) )
plt.xscale('log')

plt.figure()
plot_contours_matplotlib(df_sud_patches['patch_size'][df_sud_patches['rocks'] == 0], np.abs(y[df_sud_patches['rocks'] == 0]) , size=0.2)
# %%
##############!!! BUTTERWORTH kinda works


plt.figure()

y = df_sud['butterworth']

plt.scatter(df_sud['slope'][df_sud['rocks'] == 1], (y[df_sud['rocks'] == 1]), alpha = 0.1 )
plt.scatter(df_sud['slope'][df_sud['rocks'] == 0], (y[df_sud['rocks'] == 0]) , alpha = 0.1 )

print(len(df_sud[df_sud['butterworth'] > df_sud['butterworth'][df_sud['rocks'] == 1].max()]))



# %%


plt.figure()
# plt.scatter(df_sud['slope'][df_sud['rocks'] == False], df_sud['snow_patches'][df_sud['rocks']== False])
# plt.scatter(df_sud['slope'][df_sud['rocks'] == True], df_sud['snow_patches'][df_sud['rocks'] == True])
(df_sud['slope']).hist(bins = 1000, color = 'red', label = 'total')
(df_sud['slope'][df_sud['rocks'] == True]).hist(bins = 1000,  color = 'green', label = 'rocks')
(df_sud['slope'][df_sud['rocks'] == False]).hist(bins = 1000,  color = 'blue', label = 'snow')
plt.title('Slope distribution SUD')
plt.legend()
plt.tight_layout()

# plt.figure()
df_nord = df[(df['aspect_50m']>315*np.pi/180) | (df['aspect_50m']<45*np.pi/180)]
# (df_nord['slope']).hist(bins = 1000, color = 'red', label = 'total')
# (df_nord['slope'][df_nord['rocks'] == True]).hist(bins = 1000,  color = 'green', label = 'rocks')
# (df_nord['slope'][df_nord['rocks'] == False]).hist(bins = 1000,  color = 'blue',  label = 'snow')
# plt.title('Slope distribution NORD')
# plt.legend()
# plt.tight_layout()
# %%
df_sud_slope_count = df_sud.dropna().astype(int)
df_sud_slope_count_rocks = df_sud_slope_count[df_sud_slope_count['rocks'] == True].groupby(df_sud_slope_count.slope).count()
df_sud_slope_count_snow = df_sud_slope_count[df_sud_slope_count['rocks'] == False].groupby(df_sud_slope_count.slope).count()

diff_rocks = derivate(df_sud_slope_count_rocks['band'].values, h = 1)
diff_snow = derivate(df_sud_slope_count_snow['band'].values, h = 1)

plt.plot(df_sud_slope_count_rocks['band'].values/diff_rocks , label = 'rocks')
plt.plot(df_sud_slope_count_snow['band'].values/diff_snow, label = 'snow')
plt.legend()
# plt.twinx()
plt.plot(df_sud_slope_count_snow/df_sud_slope_count_rocks)
# %%
 
# y = df_sud['slope_50m'] - df_sud['slope']
x = df_sud['TRI']

# x = df_sud['slope_50m'] - df_sud['slope']

y = df_sud['elevation_50m'] - df_sud['elevation']

plt.figure()
plt.scatter(x[df_sud['rocks'] == True], y[df_sud['rocks'] == True], alpha = 0.05, label = 'rocks')
plt.scatter(x[df_sud['rocks'] == False], y[df_sud['rocks'] == False], alpha = 0.05,  label = 'snow')
plt.xlabel('Slope')
plt.ylabel('Terrain Ruggedness Index')
plt.title('TRI/ Slope  SUD')
plt.legend()
plt.tight_layout()




plot_contours_matplotlib(x[df_sud['rocks'] == 1], y[df_sud['rocks'] == 1], size=0.2)



# %%

x = df_sud['slope']
y = df_sud['elevation_50m'] - df_sud['elevation']
plot_contours_matplotlib(x[df_sud['rocks'] == 0], y[df_sud['rocks'] == 0], size=0.2)

# %%


plt.figure()
plt.scatter(df_sud['gcurv_50m'][df_sud['rocks'] == True], df_sud['CWTMexHat'][df_sud['rocks'] == True], alpha = 0.5, label = 'rocks')
plt.scatter(df_sud['gcurv_50m'][df_sud['rocks'] == False], df_sud['CWTMexHat'][df_sud['rocks'] == False], alpha = 0.5,  label = 'snow')
plt.xlabel('gcurv_50m')
plt.ylabel('CWTMexHat')
plt.title('gcurv_50m/ CWTMexHat  SUD')
plt.legend()
plt.tight_layout()

# %%



plt.figure()
plt.scatter(df_sud['TPI'][df_sud['rocks'] == True], df_sud['TRI'][df_sud['rocks'] == True], alpha = 0.5, label = 'rocks')
plt.scatter(df_sud['TPI'][df_sud['rocks'] == False], df_sud['TRI'][df_sud['rocks'] == False], alpha = 0.5,  label = 'snow')
plt.xlabel('TPI')
plt.ylabel('TRI')
plt.title('TRI/ TPI  SUD')
plt.legend()
plt.tight_layout()
# %%

plt.figure()
plt.scatter(df_sud['gcurv_50m'][df_sud['rocks'] == True], df_sud['TPI'][df_sud['rocks'] == True], alpha = 0.5, label = 'rocks')
plt.scatter(df_sud['gcurv_50m'][df_sud['rocks'] == False], df_sud['TPI'][df_sud['rocks'] == False], alpha = 0.5,  label = 'snow')
plt.xlabel('gcurv_50m')
plt.ylabel('CWTMexHat')
plt.title('gcurv_50m/ CWTMexHat  SUD')
plt.legend()
plt.tight_layout()

# %%

plt.figure()
plt.scatter(df_sud['gcurv_50m'][df_sud['rocks'] == True], df_sud['TPI'][df_sud['rocks'] == True], alpha = 0.5, label = 'rocks')
plt.scatter(df_sud['gcurv_50m'][df_sud['rocks'] == False], df_sud['TPI'][df_sud['rocks'] == False], alpha = 0.5,  label = 'snow')
plt.xlabel('gcurv_50m')
plt.ylabel('CWTMexHat')
plt.title('gcurv_50m/ TPI  SUD')
plt.legend()
plt.tight_layout()


# %%

for paramx in ['gcurv_50m', 'TRI', 'TPI', 'CWTMexHat']:
    for paramy in ['gcurv_50m', 'TRI', 'TPI', 'CWTMexHat']:
        plt.figure()
        plt.scatter(df_sud[paramx][df_sud['rocks'] == True], df_sud[paramy][df_sud['rocks'] == True], alpha = 0.5, label = 'rocks')
        plt.scatter(df_sud[paramx][df_sud['rocks'] == False], df_sud[paramy][df_sud['rocks'] == False], alpha = 0.5,  label = 'snow')
        plt.xlabel(paramx)
        plt.ylabel(paramy)
        plt.title(paramx + '|' +paramy)
        plt.legend()
        # plt.tight_layout()
# %%

plt.figure()
plt.scatter(df_sud['TPI'][df_sud['rocks'] == True], df_sud['CWTMexHat'][df_sud['rocks'] == True], alpha = 0.5, label = 'rocks')
plt.scatter(df_sud['TPI'][df_sud['rocks'] == False], df_sud['CWTMexHat'][df_sud['rocks'] == False], alpha = 0.5,  label = 'snow')
plt.xlabel(paramx)
plt.ylabel(paramy)
plt.title(paramx + '|' +paramy)
plt.legend()

# %%

x = (df_sud['TPI'] * df_sud['gcurv_50m']) 
y = (df_sud['CWTMexHat']) 

plt.figure()
plt.scatter(x[df_sud['rocks'] == True], y[df_sud['rocks'] == True], alpha = 0.05, label = 'rocks')
plt.scatter(x[df_sud['rocks'] == False], y[df_sud['rocks'] == False], alpha = 0.05,  label = 'snow')
plt.xlabel(paramx)
plt.ylabel(paramy)
plt.title(paramx + paramy)
plt.legend()
# plt.tight_layout()
    


# %%

plt.scatter(df_sud['slope'][df_sud['rocks'] == True], df_sud['gcurv_50m'][df_sud['rocks'] == True], alpha = 0.05, label = 'rocks')
plt.scatter(df_sud['slope'][df_sud['rocks'] == False], df_sud['gcurv_50m'][df_sud['rocks'] == False], alpha = 0.05,  label = 'snow')
plt.xlabel('Slope')
plt.ylabel('Gaussian curvature on 50m smoothed dem')
plt.title('gcurv_50m/ Slope SUD')
plt.legend()
plt.tight_layout()



# %%


plt.figure()

# plt.scatter(df_sud['slope'][df_sud['rocks'] == False], df_sud['snow_patches'][df_sud['rocks']== False])
(df_sud_patches['slope']).hist(bins = 1000, color = 'red', label = 'total')
(df_sud_patches['slope'][df_sud_patches['rocks'] == True]).hist(bins = 1000,  color = 'green', label = 'rocks')
(df_sud_patches['slope'][df_sud_patches['rocks'] == False]).hist(bins = 1000,  color = 'blue', label = 'snow')
plt.title('Patches mean slope distribution SUD')
plt.legend()
plt.tight_layout()


# %%
plt.figure()
plt.scatter(df_sud_patches['slope'][df_sud_patches.index.values != 0], df_sud_patches['patch_size'][df_sud_patches.index.values != 0], alpha = 0.5,  label = 'snow')
plt.xlabel('Slope')
plt.ylabel('patch size')
plt.title('patch_size/ Slope SUD')
plt.legend()
plt.tight_layout()


# %%

plt.figure()
plt.scatter(df_sud['slope'][df_sud['rocks'] == True], df_sud['gcurv_50m'][df_sud['rocks'] == True], alpha = 0.05, label = 'rocks')
plt.scatter(df_sud['slope'][df_sud['rocks'] == False], df_sud['gcurv_50m'][df_sud['rocks'] == False], alpha = 0.05,  label = 'snow')
plt.xlabel('Slope')
plt.ylabel('Gaussian curvature on 50m smoothed dem')
plt.title('gcurv_50m/ Slope SUD')
plt.legend()
plt.tight_layout()