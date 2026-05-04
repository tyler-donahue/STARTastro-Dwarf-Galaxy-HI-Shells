#import libraries
from photutils.aperture import EllipticalAperture
from photutils.aperture import EllipticalAnnulus
from photutils.aperture import aperture_photometry
from astropy.coordinates import SkyCoord
import astropy.units as u
import pyregion
import numpy as np
from photutils.centroids import centroid_com

from astropy.nddata import Cutout2D
import matplotlib.pyplot as plt

from photutils.segmentation import detect_sources
from astropy.visualization import ZScaleInterval, ImageNormalize
from matplotlib_scalebar.scalebar import ScaleBar

import sep

from matplotlib.patches import Ellipse
from astropy.visualization import simple_norm
from astropy.stats import SigmaClip
from photutils.background import Background2D, MedianBackground

#core to rim integration
def sum_core_to_rim(center, elongation, theta, image, arcsecs_per_pixel, axis_pix, radius_scalar=2):
    '''
    Calculates the flux over an ellipse using arcsecs
    Accepts: Center of Elipse in (x, y), elongation, theta (rads), image, arcsecs per pixel, semi-major axis (pixels),
    max scale for radius (int)
    Retuns: Array of radii, Array of summed flux at each radii
    '''

    #initialize flux array
    all_flux = []

    #initialize start of rim
    rim_start = 0

    #initialize radii
    radii = []

    #initilize radius
    radius = int(radius_scalar * axis_pix * arcsecs_per_pixel)
    rim_radius = int(axis_pix * arcsecs_per_pixel)

    #initialize step
    step = 1

    #loop from center to radius
    for arcsecs in range(1, radius, step):

        #initialize ellipse radius
        radius_a = arcsecs / arcsecs_per_pixel
        radius_b = radius_a / elongation

        #add radius_a to radii
        radii = np.append(radii, arcsecs)

        #initialize aperature
        aperature = EllipticalAperture(center, radius_a, radius_b, theta)

        #adjust background noise
        image = image - bkg_noise_offset(image)

        #calculate flux
        flux = aperture_photometry(np.abs(image), aperature)['aperture_sum'][0]

        #add flux to all_flux
        all_flux = np.append(all_flux, np.abs(flux))

        #check if index is at rim start
        if arcsecs >= rim_radius and rim_start == 0:

            #set start of rim
            rim_start = len(all_flux) - 1

    #calculate rim to core ratio
    
    #initialize rim to core ratio
    rim_core = 0

    #check if rim_start is valid
    if rim_start != 0:

        #calculate rim to core ratio
        rim_core = rim_core_ratio(all_flux, rim_start)
    
    #return radii, and all_flux
    return radii, all_flux, rim_core

def sum_core_to_rim_pix(center, elongation, theta, image, arcsecs_per_pixel, axis_pix, radius_scalar=4):
    '''
    Calculates the flux over an ellipse using pixels
    Accepts: Center of Elipse in (x, y), elongation, theta (rads), image, arcsecs per pixel, semi-major axis (pixels),
    max scale for radius (int)
    Retuns: Array of radii, Array of summed flux at each radii
    '''

    #initialize flux array
    all_flux = []

    #initialize start of rim
    rim_start = 0

    #initialize radii
    radii = []

    #initilize radius
    radius = int(radius_scalar * axis_pix)
    rim_radius = int(axis_pix)

    #initialize step
    step = 1

    #loop from center to radius
    for pixel in range(1, radius, step):

        #initialize ellipse radius
        radius_a = pixel * arcsecs_per_pixel
        radius_b = radius_a / elongation

        #add radius_a to radii
        radii = np.append(radii, radius_a)

        #initialize aperature
        aperature = EllipticalAperture(center, radius_a, radius_b, theta)
        
        #calculate flux
        flux = aperture_photometry(image, aperature)['aperture_sum'][0]

        #add flux to all_flux
        all_flux = np.append(all_flux, flux)

        #check if index is at start of rim
        if pixel >= rim_radius and rim_start == 0:
            
            #set start of rim index
            rim_start = len(all_flux) - 1

    #calculate rim to core ratio

    #initialize core to rim ratio
    rim_core = 0
    
    #check if rim_start is valid
    if rim_start != 0:

        #calculate rim to core ratio
        rim_core = rim_core_ratio(all_flux, rim_start)

    #set core radius to zero
    radii[0] = 0
    
    #return radii, and all_flux
    return radii, all_flux, rim_core

#create cutout around given center
def cut_mark_image(center, pixel_size_a, pixel_size_b, theta, image, ellipse_scale=2):
    '''
    Cuts image around center and creates ellipse around center
    Accepts: center (x, y), pixel height, pixel width, theta (rads), image
    Returns: cut image, ellipse defined around center
    '''

    #initialize parameters
    cutout_scale = 5
    #ellipse_scale = 2

    try:
        #cut image
        cutout_obj = Cutout2D(data=image, position=center, size=pixel_size_a*cutout_scale)
        cutout = cutout_obj.data 
    
        #center of cutout
        x_c, y_c = cutout_obj.input_position_cutout
    
        #define ellipse
        ellipse = Ellipse((x_c, y_c), width=pixel_size_a*ellipse_scale, height=pixel_size_b*ellipse_scale,
                       angle=np.degrees(theta), edgecolor='g', facecolor='none', lw=4)
    except:
        raise "Failed Cutout"

    #return cutout and region ellipse
    return cutout, ellipse

#plot fits image
def plot_fits(image, arcsec_per_pixel, distance, ax, ellipse=0):

    #initialize zscale
    zscale = ZScaleInterval()
    
    #determine zmin and zmax
    vmin, vmax = zscale.get_limits(image)

    #plot image
    ax.imshow(image, origin='lower', cmap='inferno', vmin=vmin, vmax=vmax)
    ax.set_xlabel("pixel")
    ax.set_ylabel("pixel")

    #check if ellipse is valid
    if ellipse != 0:
        #plot region ellipse
        ax.add_patch(ellipse)

    #add scalebar to image
    scalebar = ScaleBar(arcsecs_to_radius(arcsec_per_pixel, distance), units='kpc', dimension="astro-length", border_pad=.8, box_alpha=0, color='w')
    ax.add_artist(scalebar)

#calculates rim to core ratio
def rim_core_ratio(flux_array, rim_start=-2):
    '''
    Calculates rim to core ratio
    Accepts: Array of cumulative flux
    Returns: Ratio of rim to core
    '''

    #initialize core flux
    core = flux_array[0]

    #calculate rim flux
    rim = flux_array[-1] - flux_array[rim_start]

    #calculate rim to core ratio
    rim_core_ratio = rim / core

    #return rim to core ratio
    return rim_core_ratio

#determine background noise offset
def bkg_noise_offset(image, sigma=3.0):
    '''
    Calculates the median background noise
    Accepts: 2D numpy array
    Returns: Value of background noise
    '''

    #initialize sigma
    sigma_clip = SigmaClip(sigma=sigma)

    #initialize background estimator
    bkg_estimator = MedianBackground()

    #generate background noise
    bkg = Background2D(image, (50, 50), filter_size=(3, 3), sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)

    #return background noise offset
    return bkg.background_median

#subtract noise
def subtract_noise(image):
    '''
    Subtracts noise from image
    Accepts: Image
    Returns: Image with noise subtracted
    '''

    #calculate background noise
    bkg = bkg_noise_offset(image)

    #check if negative
    if bkg < 0:

        #return corrected image
        return image - bkg

    else:

        #return original image
        return image

#plot core to rim sums
def plot_core_rim(radii_hi, flux_hi, radii_ha, flux_ha, radii_irac, flux_irac, axs, galaxy, index, hi_rim, unit_ha):
    '''
    Plots flux vs radius of three objects
    Accepts: radius a, flux a, radius b, flux b, radius c, flux c, figure axis, object name, object position, units for ha flux
    Returns: None
    '''
    
    #set first y axis
    ax1 = axs

    #plot hi flux
    ax1.scatter(radii_hi, np.log10(flux_hi), marker='s', label="HI Flux")
    ax1.set_xlabel("Distance from Center (kpc)")
    ax1.set_ylabel("Flux: JY/B*M/S", color='b')
    ax1.tick_params(axis='y', labelcolor='b')

    #create secondary y axis for ha
    ax2 = ax1.twinx()

    #create tertiary y axis for irac
    ax3 = ax1.twinx()

    #determine ha flux units
    if unit_ha == 1:
        units = "Counts/s"
    else:
        units = "erg/s/cm^2/A"

    #plot ha flux
    ax2.scatter(radii_ha, np.log10(flux_ha), label="H-alpha Flux", color="red")
    ax2.set_ylabel(f"Flux: {units}", color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    #plot irac flux
    ax3.scatter(radii_irac, np.log10(flux_irac), marker='*', label="IRAC 8 micron Flux", color="green")
    ax3.set_ylabel("Flux: MJy/sr", color='g')
    ax3.spines.right.set_position(("axes", 1.2))
    ax3.tick_params(axis='y', labelcolor='g')

    #create legend
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    handles3, labels3 = ax3.get_legend_handles_labels()
    handles = handles1 + handles2 + handles3
    labels = labels1 + labels2 + labels3
    ax1.legend(handles, labels, loc='best')
    ax1.axvline(x=hi_rim)

    #set title
    ax1.set_title(f"Flux of HI vs. H-alpha in Galaxy: {galaxy} - Region {index + 1}")

#plot one core rim
def plot_core_rim_one(radii_ha, flux_ha, axs, galaxy, index, unit_ha):
    '''
    Plots flux vs radius of three objects
    Accepts: radius a, flux a, radius b, flux b, radius c, flux c, figure axis, object name, object position, units for ha flux
    Returns: None
    '''
    
    #set first y axis
    ax1 = axs

    #plot hi flux
    ax1.scatter(radii_ha, np.log10(flux_ha), label="H-alpha Flux", color='r')
    ax1.set_xlabel("Distance from Center (kpc)")
    ax1.set_ylabel("Flux: erg/s/cm^2/A", color='r')
    ax1.tick_params(axis='y', labelcolor='r')

    #create legend
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles = handles1
    labels = labels1
    ax1.legend(handles, labels, loc='best')
    #ax1.axvline(x=hi_rim)

    #set title
    ax1.set_title(f"Flux of HI and H-alpha: {galaxy} - Region {index + 1}")

#plot one core rim
def plot_core_rim_two(radii_hi, flux_hi, axs, galaxy, index, unit_ha):
    '''
    Plots flux vs radius of three objects
    Accepts: radius a, flux a, radius b, flux b, radius c, flux c, figure axis, object name, object position, units for ha flux
    Returns: None
    '''
    
    #set first y axis
    ax1 = axs

    #plot hi flux
    ax1.scatter(radii_hi, np.log10(flux_hi), marker='s', label="HI Flux")
    ax1.set_xlabel("Distance from Center (kpc)")
    ax1.set_ylabel("Flux: JY/B*M/S", color='b')
    ax1.tick_params(axis='y', labelcolor='b')

    #create legend
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles = handles1
    labels = labels1
    ax1.legend(handles, labels, loc='best')
    #ax1.axvline(x=hi_rim)

    #set title
    #ax1.set_title(f"Flux of HI: {galaxy} - Region {index + 1}")

#convert arcsecs to radius
def arcsecs_to_radius(arcsecs, distance):
    '''
    Converts arcsecs to a length in kpc
    Accepts: arcsecs, distance to object in kpc
    Returns: Length oarcsecs in kpc
    '''

    #return length in kpc
    return arcsecs * distance * 4.85 * (10**-6)

#mask stars from cutouts
def mask_stars(image, sigma=100):
    '''
    Removes stars from FITS image
    Accepts: numpy image
    Returns: numpy image with stars masked
    '''

    #convert image to c-continous
    image = np.ascontiguousarray(image)

    #fix byte data 
    image = image.astype(image.dtype.newbyteorder('='))
    
    #calculate background noise
    bkg = sep.Background(image)
    print(bkg.globalrms)

    #subtract background from image
    if bkg.globalback < 0:
        image = image - bkg.globalback

    #grab stars
    objects, stars = sep.extract(image, sigma, err=bkg.globalrms, segmentation_map=True)

    #make mask
    mask = stars > 0

    #mask image
    masked_image = np.ma.masked_array(image, mask=mask)

    return masked_image


    