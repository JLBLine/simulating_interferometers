'''Useful functions for simulating interferometric observations'''
from __future__ import division
from __future__ import print_function
from astropy.io import fits
from ephem import Observer
from numpy import *
from numpy import abs as np_abs
from numpy import exp as np_exp
import matplotlib.pyplot as plt

##Convert degrees/rads
D2R = pi/180.0
R2D = 180.0/pi
##Speed of light m/s
VELC = 299792458.0
##Latitude of the MWA
MWA_LAT = -26.7033194444
##Always set the kernel size to an odd value
##Makes all ranges set to zero at central values
KERNEL_SIZE = 31
##Rotational velocity of the Earth rad / sex
W_E = 7.292115e-5
##Sidereal seconds per solar seconds - ie if 1s passes on
##the clock, sky has moved by 1.00274 secs of angle
SOLAR2SIDEREAL = 1.00274

def enh2xyz(east,north,height,latitiude):
    '''Calculates local X,Y,Z using east,north,height coords,
    and the latitude of the array. Latitude must be in radians'''
    sl = sin(latitiude)
    cl = cos(latitiude)
    X = -north*sl + height*cl
    Y = east
    Z = north*cl + height*sl
    return X,Y,Z

def get_lm(ra=None,ra0=None,dec=None,dec0=None):
    '''Calculate l,m,n for a given phase centre ra0,dec0 and sky point ra,dec
    Enter angles in radians'''

    ##RTS way of doing it
    cdec0 = cos(dec0)
    sdec0 = sin(dec0)
    cdec = cos(dec)
    sdec = sin(dec)
    cdra = cos(ra-ra0)
    sdra = sin(ra-ra0)
    l = cdec*sdra
    m = sdec*cdec0 - cdec*sdec0*cdra
    n = sdec*sdec0 + cdec*cdec0*cdra
    return l,m,n

def get_uvw(x_lamb,y_lamb,z_lamb,dec,HA):
    '''Calculates u,v,w for a given '''
    u = sin(HA)*x_lamb + cos(HA)*y_lamb
    v = -sin(dec)*cos(HA)*x_lamb + sin(dec)*sin(HA)*y_lamb + cos(dec)*z_lamb
    w = cos(dec)*cos(HA)*x_lamb - cos(dec)*sin(HA)*y_lamb + sin(dec)*z_lamb
    return u,v,w

def find_closet_uv(u=None,v=None,u_range=None,v_range=None):
    '''Finds the closet values to u,v in the ranges u_range,v_range
    Returns the index of the closest values, and the offsets from
    the closest values'''
    u_resolution = u_range[1] - u_range[0]
    v_resolution = v_range[1] - v_range[0]
    ##Find the difference between the gridded u coords and the desired u
    u_offs = np_abs(u_range - u)
    ##Find out where in the gridded u coords the current u lives;
    ##This is a boolean array of length len(u_offs)
    u_true = u_offs < u_resolution/2.0
    ##Find the index so we can access the correct entry in the container
    u_ind = where(u_true == True)[0]

    ##Use the numpy abs because it's faster (np_abs)
    v_offs = np_abs(v_range - v)
    v_true = v_offs < v_resolution/2.0
    v_ind = where(v_true == True)[0]

    ##If the u or v coord sits directly between two grid points,
    ##just choose the first one ##TODO choose smaller offset?
    if len(u_ind) == 0:
        u_true = u_offs <= u_resolution/2
        u_ind = where(u_true == True)[0]
        #print('here')
        #print(u_range.min())
    if len(v_ind) == 0:
        v_true = v_offs <= v_resolution/2
        v_ind = where(v_true == True)[0]
    # print(u,v)
    u_ind,v_ind = u_ind[0],v_ind[0]

    u_offs = u_range - u
    v_offs = v_range - v

    u_off = -(u_offs[u_ind] / u_resolution)
    v_off = -(v_offs[v_ind] / v_resolution)

    return u_ind,v_ind,u_off,v_off

def image_gaussian(kernel_sig_x=None,kernel_sig_y=None,l_mesh=None,m_mesh=None,cell_reso=None):
    '''Takes desired properties for a kernel in u,v space (in pixel coords),
    and creates FT of this'''

    fiddle = 2*pi
    sig_l, sig_m = 1.0/(fiddle*cell_reso*kernel_sig_x), 1.0/(fiddle*cell_reso*kernel_sig_y)

    l_bit = l_mesh*l_mesh / (2*sig_l*sig_l)
    m_bit = m_mesh*m_mesh / (2*sig_m*sig_m)

    return np_exp(-(l_bit + m_bit))

def gaussian(sig_x=None,sig_y=None,gridsize=KERNEL_SIZE,x_offset=0,y_offset=0):
    '''Creates a gaussian array of a specified gridsize, with the
    the gaussian peak centred at an offset from the centre of the grid'''
    x_cent = int(gridsize / 2.0) + x_offset
    y_cent = int(gridsize / 2.0) + y_offset

    x = arange(gridsize)
    y = arange(gridsize)
    x_mesh, y_mesh = meshgrid(x,y)

    x_bit = (x_mesh - x_cent)*(x_mesh - x_cent) / (2*sig_x*sig_x)
    y_bit = (y_mesh - y_cent)*(y_mesh - y_cent) / (2*sig_y*sig_y)

    amp = 1 / (2*pi*sig_x*sig_y)
    gaussian = amp*np_exp(-(x_bit + y_bit))
    return gaussian

def image2kernel(image=None,cell_reso=None,u_off=0.0,v_off=0.0,l_mesh=None,m_mesh=None):
    '''Takes an input image array, and FTs to create a kernel
    Uses the u_off and v_off (given in pixels values), cell resolution
    and l and m coords to phase    shift the image, to create a kernel
    with the given u,v offset'''

    ##TODO WARNING - if you want to use a complex image for the kernel,
    ##may need to either take the complex conjugate, or flip the sign in
    ##the phase shift, or reverse the indicies in l_mesh, m_mesh. Or some
    ##combo of all!! J. Line 20-07-2016
    phase_shift_image =  image * np_exp(2.0j * pi*(u_off*cell_reso*l_mesh + v_off*cell_reso*m_mesh))
    #phase_shift_image =  image * np_exp(2j * pi*(u_off*l_mesh + v_off*m_mesh))

    ##FFT shift the image ready for FFT
    phase_shift_image = fft.ifftshift(phase_shift_image)
    ##Do the forward FFT as we define the inverse FFT for u,v -> l,m.
    ##Scale the output correctly for the way that numpy does it, and remove FFT shift
    recovered_kernel = fft.fft2(phase_shift_image) / (image.shape[0] * image.shape[1])
    recovered_kernel = fft.fftshift(recovered_kernel)
    #return recovered_kernel
    return recovered_kernel

def sample_image_coords(n2max=None,l_reso=None,num_samples=KERNEL_SIZE):
    '''Creates a meshgrid of l,m coords to give a specified
    size array which covers a given range of l and m - always
    the same range of l,m'''

    ##So this makes you sample at zero and all the way up to half a
    ##resolution element away from the edge of your range
    ##n2max is half the l range, ie want -n2max <= l <= n2max
    offset = n2max*l_reso / num_samples
    l_sample = linspace(-n2max*l_reso + offset, n2max*l_reso - offset, num_samples)
    m_sample = linspace(-n2max*l_reso + offset, n2max*l_reso - offset, num_samples)
    l_mesh, m_mesh = meshgrid(l_sample,m_sample)

    return l_mesh, m_mesh

def add_kernel(uv_array,u_ind,v_ind,kernel):
    '''Takes v by u sized kernel and adds it into
    a numpy array at the u,v point u_ind, v_ind
    Kernel MUST be odd dimensions for symmetry purposes'''
    ker_v,ker_u = kernel.shape
    width_u = int((ker_u - 1) / 2)
    width_v = int((ker_v - 1) / 2)
    array_subsec = uv_array[v_ind - width_v:v_ind+width_v+1,u_ind - width_u:u_ind+width_u+1]
    array_subsec += kernel

def grid(container=None,u_coords=None, v_coords=None, u_range=None, v_range=None,complexes=None, kernel='gaussian', kernel_params=[2.0,2.0]):
    '''A simple(ish) gridder - defaults to gridding with a gaussian '''
    for i in arange(len(u_coords)):
        u,v,comp = u_coords[i],v_coords[i],complexes[i]
        ##Find the difference between the gridded u coords and the current u
        ##Get the u and v indexes in the uv grdding container
        u_ind,v_ind,u_off,v_off = find_closet_uv(u=u,v=v,u_range=u_range,v_range=v_range)

        if kernel == 'gaussian':
            kernel_array = gaussian(sig_x=kernel_params[0],sig_y=kernel_params[1],gridsize=KERNEL_SIZE,x_offset=0,y_offset=0)
        else:
            kernel_array = array([[complex(1,0)]])
            # kernel_array = 1.0
        ##Multiply the kernal by the complex value
        data_kernel = kernel_array * comp
        ##Add the multiplied kernel-uvdata values to the grid
        add_kernel(container,u_ind,v_ind,data_kernel)

    return container
