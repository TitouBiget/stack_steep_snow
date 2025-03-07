import numpy as np
# from topocalc import gradient
import cv2
import matplotlib.pyplot as plt


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


def kernel_square(nPix):
    """
    Function to defin a square kernel of equal value for performing averaging
    :param nPix: size of the kernel in pixel
    :return: kernel matrix
    """
    print("Averaging kernel of " + str(nPix) + " by " + str(nPix))
    kernel = np.empty([nPix, nPix])
    kernel.fill(1)
    kernel /= kernel.sum()   # kernel should sum to 1!  :)
    return kernel


def smooth(mat, kernel):
    """
    Function that produce a smoothed version of the 2D array
    :param mat: Array to smooth
    :param nPix: kernel array (output) from the function kernel_square()
    :return: smoothed array
    """
    r = cv2.filter2D(mat, -1, kernel)
    print("Smoothing done ...")
    return r

def compute_dem_param(ds, params=['slope', 'aspect', 'svf'] ):
    """
    Function to compute and derive DEM parameters: slope, aspect, sky view factor

    Args:
        dem_file (str): path to raster file (geotif). Raster must be in local cartesian coordinate system (e.g. UTM)

    Returns:
        dataset: x, y, elev, slope, aspect, svf

    """
    print(f"\n---> Extracting DEM parameters ({', '.join(params)})")
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
