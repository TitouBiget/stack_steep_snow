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



# %%
    
ds = xr.open_dataset("stack_xdem.nc", engine="netcdf4")
 # %%   
df = ds[['snow_patches', 'rocks','elevation', 'slope_horn', 'slope_zevenberg', 'diff_slope', 'aspect_horn', 'aspect_horn_50m',
         'diff_slope', 'aspect_horn', 'TPI', 'TRI', 'roughness', 'rugosity', 'FR', 'curv_50m', 'prcurv_50m', 'plcurv_50m']].to_dataframe()


df_sud = df[(df['aspect_horn_50m']>135*np.pi/180) & (df['aspect_horn_50m']<225*np.pi/180) & (df['slope_horn']>50)]

# %%


# plt.scatter(df_sud['slope'][df_sud['rocks'] == False], df_sud['snow_patches'][df_sud['rocks']== False])
# plt.scatter(df_sud['slope'][df_sud['rocks'] == True], df_sud['snow_patches'][df_sud['rocks'] == True])
(df_sud['slope_horn']).hist(bins = 1000, color = 'red', label = 'total')
(df_sud['slope_horn'][df_sud['rocks'] == True]).hist(bins = 300,  color = 'green', label = 'rocks')
(df_sud['slope_horn'][df_sud['rocks'] == False]).hist(bins = 300,  color = 'blue', label = 'snow')
plt.title('Slope distribution SUD')
plt.legend()
plt.tight_layout()

# %%
 

plt.figure()
plt.scatter(df_sud['TPI'][df_sud['rocks'] == True], df_sud['plcurv_50m'][df_sud['rocks'] == True], alpha = 0.2)
plt.scatter(df_sud['TPI'][df_sud['rocks'] == False], df_sud['plcurv_50m'][df_sud['rocks'] == False], alpha = 0.2)



