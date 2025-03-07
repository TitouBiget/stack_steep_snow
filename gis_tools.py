from pyproj import Transformer
# from topocalc import gradient
# from topocalc import viewf
import cv2
import numpy as np
import xarray as xr
import rioxarray
import xarray
# from pysheds.grid import Grid

def convert_pts(xs,ys, src='EPSG:4326', tgt='EPSG:2154'):
    """
    Simple function to convert a list fo point from one projection to another oen using PyProj with projection code

    Args:
        xs (array): 1D array with X-coordinate expressed in the source EPSG
        ys (array): 1D array with Y-coordinate expressed in the source EPSG
        src (int): source projection code
        tgt (int): target projection code

    Returns:
        array: Xs 1D arrays of the point coordinates expressed in the target projection
        array: Ys 1D arrays of the point coordinates expressed in the target projection
    """
    print(f'Convert coordinates from {src} to {tgt}')
    trans = Transformer.from_crs(src, tgt, always_xy=True)
    Xs, Ys = trans.transform(xs, ys)
    return Xs, Ys

def convert_epsg_pts(xs,ys, epsg_src=4326, epsg_tgt=2154):
    """
    Simple function to convert a list fo poitn from one EPSG projection to another oen using PyProj

    Args:
        xs (array): 1D array with X-coordinate expressed in the source EPSG
        ys (array): 1D array with Y-coordinate expressed in the source EPSG
        epsg_src (int): source projection EPSG code
        epsg_tgt (int): target projection EPSG code

    Returns:
        array: Xs 1D arrays of the point coordinates expressed in the target projection
        array: Ys 1D arrays of the point coordinates expressed in the target projection
    """
    print('Convert coordinates from EPSG:{} to EPSG:{}'.format(epsg_src, epsg_tgt))
    trans = Transformer.from_crs("epsg:{}".format(epsg_src), "epsg:{}".format(epsg_tgt), always_xy=True)
    Xs, Ys = trans.transform(xs, ys)
    return Xs, Ys


def compute_dem_param(dem_file, params=['slope', 'aspect', 'svf'] ):
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
    dx = ds.x.diff('x').median().values
    dy = ds.y.diff('y').median().values
    dem_arr = np.flip(ds.elevation.values, 0)

    print('Computing slope and aspect ...')
    if 'slope' or 'aspect' in params:
        slope, aspect = gradient.gradient_d8(dem_arr, dx, dy)

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

    if 'svf' in params:
        print('Computing svf ...')
        svf = viewf.viewf(np.double(dem_arr), dx)[0]
        ds['svf'] = (["y", "x"], svf)
        ds.svf.attrs = {'units': 'ratio', 'standard_name': 'svf', 'long_name': 'Sky view factor'}

    ds.attrs = dict(description="DEM input parameters to TopoSub",
                   author="mettools, https://github.com/ArcticSnow/mettools")
    ds.x.attrs = {'units': 'm'}
    ds.y.attrs = {'units': 'm'}
    ds.elevation.attrs = {'units': 'm'}

    return ds


def laplacian_of_gaussian(x, y, sigma):
    '''
    Laplacian of Gaussian operator for curvature computation smoothing noise.
    https://homepages.inf.ed.ac.uk/rbf/HIPR2/log.htm

    :param x: array of x-coordinate. Should be centered on 0.
    :param y: array of y-coordinate. Should be centered on 0.
    :param sigma: float, standard deviation setting the with of the bell curve
    :return: array containing the operator
    '''
    op = -1/(np.pi * sigma **4)*(1-((x**2 + y**2)/(2*sigma**2)))*np.exp(-((x**2 + y**2)/(2*sigma**2)))
    return op

def laplacian_of_gaussian_kernel(n, sigma):
    """
    Function to compute the kernel of Laplacian of Gaussian. Kernel is square.
    Args:
        n: number of pixel
        sigma:

    Returns:

    """
    x = np.arange(-n,n)
    y = np.arange(-n,n)
    Xs, Ys = np.meshgrid(x,y)
    kernel = laplacian_of_gaussian(Xs,Ys,sigma)
    #print('Kernel sum = ',np.sum(kernel))
    return kernel

def plot_kernel(kernel):
    plt.figure()
    plt.imshow(kernel)
    plt.colorbar()

def convolve_array(arr, kernel):
    """
    Function to convolve a kernel to a 2D array using OpenCV

    Args:
        arr (float): 2D array
        kernel (float): kernel to convolve. 2D array

    Returns:
        Convolved arrau
    """
    res = cv2.filter2D(arr, -1, kernel)
    return res

def compute_laplacian_of_gaussian(arr, n, sigma, normed=True):
    kernel = laplacian_of_gaussian_kernel(n,sigma)
    res = convolve_array(arr, kernel)
    if normed:
        res = res/(np.max(np.abs(res)))
    return res


def sample_dataarray(da, Xs, Ys, interpolation_method='cubic'):
    """
    Function to sample a DataArray for a list of points of coordinates (Xs,Ys)

    Args:
        da: DataArray (2D or more)
        Xs (float): array of X-coordinates of the points
        Ys (float): array of Y-coordinates of the points

    Returns:
        samples (numpy array) values
    """
    sample = np.diagonal(topo.curv_2.interp({'x':df.x.values, 'y':df.y.values}, method=interpolation_method).values)
    return sample



# Class to delineate hydrological catchment

class Catchment:
    """
    Class to derive watershed boundaries based on a DEM and an exit point

    """
    def __init__(self, dem_fname, acc_thres=200):
        """
        Initialization of the class
        Args:
            dem_fname (str): filename of the DEM raster
            acc_thres (float): tunning parameter for computing channeling (see pysheds documentation)
        """
        self.fname = dem_fname
        self.grid = Grid.from_raster(self.fname)
        self.dem = self.grid.read_raster(self.fname)

        self.dem_xr = xr.open_dataset(self.fname, engine='rasterio')
        self.dirmap = None
        self.acc_thres = acc_thres


    def prepare_dem(self, dirmap = (64, 128, 1, 2, 4, 8, 16, 32)):
        self.dirmap = dirmap

        # Condition DEM
        # ----------------------
        # Fill pits in DEM
        self.pit_filled_dem = self.grid.fill_pits(self.dem)

        # Fill depressions in DEM
        self.flooded_dem = self.grid.fill_depressions(self.pit_filled_dem)

        # Resolve flats in DEM
        self.inflated_dem = self.grid.resolve_flats(self.flooded_dem)

        # Compute flow directions
        # -------------------------------------
        self.fdir = self.grid.flowdir(self.inflated_dem, dirmap=self.dirmap)
        self.acc = self.grid.accumulation(self.fdir, dirmap=self.dirmap)



    def delineate_catchment(self, x_exit, y_exit):

        # Snap pour point to high accumulation cell
        x_snap, y_snap = self.grid.snap_to_mask(self.acc > self.acc_thres, (x_exit, y_exit))

        # Delineate the catchment
        self.catch = self.grid.catchment(x=x_snap, y=y_snap, fdir=self.fdir, dirmap=self.dirmap,
                               xytype='coordinate')

        # Crop and plot the catchment
        # ---------------------------
        # Clip the bounding box to the catchment
        self.grid.clip_to(self.catch)
        #clipped_catch = self.grid.view(self.catch)
        self.branches = self.grid.extract_river_network(self.fdir, self.acc > self.acc_thres, dirmap=self.dirmap)
        self.dist = self.grid.distance_to_outlet(x=x_snap, y=y_snap, fdir=self.fdir, dirmap=self.dirmap,
                                       xytype='coordinate')
        print('--> Catchement delineated <--')



    def plot(self, x_exit, y_exit, label='La Bérarde'):
        ls = LightSource(azdeg=315, altdeg=45)

        fig, ax = plt.subplots(figsize=(8,6))
        fig.patch.set_alpha(0)
        plt.grid('on', zorder=0)

        dx = np.mean(np.diff(self.dem_xr.x))
        dy = np.mean(np.diff(self.dem_xr.y))
        print(f'dx={dx}, dy={dy}')

        plt.imshow(ls.hillshade(self.dem_xr.band_data.values[0,:,:], dx=dx, dy=dy), cmap=plt.cm.gray, extent=[self.dem_xr.x.min(), self.dem_xr.x.max(), self.dem_xr.y.min(), self.dem_xr.y.max()])

        im = ax.imshow(self.dist, extent=self.grid.extent, zorder=2,
                       cmap='cubehelix_r', alpha=0.5)


        plt.xlim(self.grid.bbox[0], self.grid.bbox[2])
        plt.ylim(self.grid.bbox[1], self.grid.bbox[3])
        ax.set_aspect('equal')

        for branch in self.branches['features']:
            line = np.asarray(branch['geometry']['coordinates'])
            plt.plot(line[:, 0], line[:, 1], c='k')

        # plot position of La Berarde
        plt.scatter(x_exit, y_exit, c='r')
        plt.text(x_exit, y_exit, label)

        plt.colorbar(im, ax=ax, label='Distance to outlet (cells)')
        plt.xlabel('East [m]')
        plt.ylabel('North [m]')
        plt.title('Flow Distance', size=14)