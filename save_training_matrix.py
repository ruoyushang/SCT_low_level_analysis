
import os, sys
import subprocess
import glob

import numpy as np
from astropy import units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time
from matplotlib import pyplot as plt
from matplotlib import colors
import pickle

from ctapipe.io import EventSource, SimTelEventSource, HDF5TableWriter
from ctapipe.calib import CameraCalibrator
from ctapipe.image import hillas_parameters, tailcuts_clean, ImageProcessor
from ctapipe.reco import ShowerProcessor
from ctapipe.coordinates import CameraFrame, NominalFrame, TelescopeFrame
from traitlets.config import Config

import common_functions
remove_nan_pixels = common_functions.remove_nan_pixels
reset_time = common_functions.reset_time
find_image_moments = common_functions.find_image_moments
image_translation = common_functions.image_translation
image_rotation = common_functions.image_rotation

font = {'family': 'serif', 'color':  'white', 'weight': 'normal', 'size': 10, 'rotation': 0.,}

def run_save_training_matrix(training_sample_path, min_energy=0.1, max_energy=1000., max_evt=1e10):

    fig, ax = plt.subplots()
    figsize_x = 8.6
    figsize_y = 6.4
    fig.set_figheight(figsize_y)
    fig.set_figwidth(figsize_x)

    big_movie_matrix = []
    big_moment_matrix = []
    big_truth_matrix = []

    print (f'loading file: {training_sample_path}')
    source = SimTelEventSource(training_sample_path, focal_length_choice='EQUIVALENT')
    
    # Explore the instrument description
    subarray = source.subarray
    print (subarray.to_table())
    
    ob_keys = source.observation_blocks.keys()
    run_id = list(ob_keys)[0]
    
    tel_pointing_alt = float(source.observation_blocks[run_id].subarray_pointing_lat/u.rad)
    tel_pointing_az = float(source.observation_blocks[run_id].subarray_pointing_lon/u.rad)
    print (f'tel_pointing_alt = {tel_pointing_alt}')
    print (f'tel_pointing_az = {tel_pointing_az}')
    
    image_processor_config = Config(
            {
                "ImageProcessor": {
                    "image_cleaner_type": "TailcutsImageCleaner",
                    "TailcutsImageCleaner": {
                        "picture_threshold_pe": [
                            ("type", "LST_LST_LSTCam", 7.5),
                            ("type", "MST_MST_FlashCam", 8),
                            ("type", "MST_MST_NectarCam", 8),
                            ("type", "MST_SCT_SCTCam", 8),
                            ("type", "SST_ASTRI_CHEC", 7),
                            ],
                        "boundary_threshold_pe": [
                            ("type", "LST_LST_LSTCam", 5),
                            ("type", "MST_MST_FlashCam", 4),
                            ("type", "MST_MST_NectarCam", 4),
                            ("type", "MST_SCT_SCTCam", 4),
                            ("type", "SST_ASTRI_CHEC", 4),
                            ],
                        },
                    }
                }
            )
    calib = CameraCalibrator(subarray=subarray)
    image_processor = ImageProcessor(subarray=subarray,config=image_processor_config)
    shower_processor = ShowerProcessor(subarray=subarray)

    for event in source:
    
        event_id = event.index['event_id']
        #if event_id!=89803: continue
    
        ntel = len(event.r0.tel)
        
        truth_energy = event.simulation.shower.energy
        truth_core_x = event.simulation.shower.core_x
        truth_core_y = event.simulation.shower.core_y
        truth_alt = event.simulation.shower.alt
        truth_az = event.simulation.shower.az
        truth_height = event.simulation.shower.h_first_int
        truth_x_max = event.simulation.shower.x_max
        print (f'truth_energy = {truth_energy}')

        truth_info_array = [truth_energy, truth_core_x, truth_core_y, truth_alt, truth_az, truth_height, truth_x_max]
        
        calib(event)  # fills in r1, dl0, and dl1
        #image_processor(event) # Takes DL1/Image data and cleans and parametrizes the images into DL1/parameters. Should be run after CameraCalibrator.
        #shower_processor(event) # Run the stereo event reconstruction on the input events.
    
        for tel_idx in range(0,len(list(event.dl0.tel.keys()))):

            tel_id = list(event.dl0.tel.keys())[tel_idx]
            print ('====================================================================================')
            print (f'event_id = {event_id}, tel_id = {tel_id}')

            geometry = subarray.tel[tel_id].camera.geometry

            obstime = Time("2013-11-01T03:00")
            location = EarthLocation.of_site("Roque de los Muchachos")
            altaz = AltAz(location=location, obstime=obstime)

            tel_pointing_alt = source.observation_blocks[run_id].subarray_pointing_lat
            tel_pointing_az = source.observation_blocks[run_id].subarray_pointing_lon
            tel_pointing = SkyCoord(
                alt=tel_pointing_alt,
                az=tel_pointing_az,
                frame=altaz,
            )
            alt_offset = truth_alt - tel_pointing_alt
            az_offset = truth_az - tel_pointing_az
            star_altaz = SkyCoord(
                alt=tel_pointing_alt + alt_offset,
                az=tel_pointing_az + az_offset,
                frame=altaz,
            )
           
            focal_length = source.subarray.tel[tel_id].optics.equivalent_focal_length
            camera_frame = CameraFrame(
                telescope_pointing=tel_pointing,
                focal_length=focal_length,
            )

            star_cam = star_altaz.transform_to(camera_frame)
            star_cam_x = star_cam.x.to_value(u.m)
            star_cam_y = star_cam.y.to_value(u.m)
            print (f'star_cam_x = {star_cam_x}, star_cam_y = {star_cam_y}')

            dirty_image_1d = event.dl1.tel[tel_id].image
            dirty_image_2d = geometry.image_to_cartesian_representation(dirty_image_1d)
            remove_nan_pixels(dirty_image_2d)

            clean_image_1d = event.dl1.tel[tel_id].image
            clean_time_1d = event.dl1.tel[tel_id].peak_time
            image_mask = tailcuts_clean(geometry,clean_image_1d,boundary_thresh=1,picture_thresh=3,min_number_picture_neighbors=2)
            for pix in range(0,len(image_mask)):
                if not image_mask[pix]:
                    clean_image_1d[pix] = 0.
                    clean_time_1d[pix] = 0.
            #hillas_params = hillas_parameters(geometry, clean_image_1d)
            #print (f'hillas_params = {hillas_params}')
            #hillas_psi = hillas_params['psi']
            #hillas_x = hillas_params['x']
            #hillas_y = hillas_params['y']

            reset_time(clean_image_1d, clean_time_1d)

            clean_image_2d = geometry.image_to_cartesian_representation(clean_image_1d)
            remove_nan_pixels(clean_image_2d)
            clean_time_2d = geometry.image_to_cartesian_representation(clean_time_1d)
            remove_nan_pixels(clean_time_2d)

            pixel_width = float(geometry.pixel_width[0]/u.m)
            print (f'pixel_width = {pixel_width}')

            image_moment_array = find_image_moments(geometry, clean_image_1d, clean_time_1d)
            image_size = image_moment_array[0]
            image_center_x = image_moment_array[1]
            image_center_y = image_moment_array[2]
            angle = image_moment_array[3]
            semi_major = image_moment_array[4]
            semi_minor = image_moment_array[5]
            time_direction = image_moment_array[6]
            image_direction = image_moment_array[7]
            line_a = image_moment_array[8]
            line_b = image_moment_array[9]

            if image_size==0.:
                continue
            #print (f'image_center_x = {image_center_x}')
            #print (f'image_center_y = {image_center_y}')
            #print (f'angle = {angle}')

            image_qual = abs(image_direction+time_direction)

            shift_pix_x = image_center_x/pixel_width
            shift_pix_y = image_center_y/pixel_width
            shift_image_2d = image_translation(clean_image_2d, round(float(shift_pix_y)), round(float(shift_pix_x)))
            rotate_image_2d = image_rotation(shift_image_2d, angle*u.rad)

            xmax = max(geometry.pix_x)/u.m
            xmin = min(geometry.pix_x)/u.m
            ymax = max(geometry.pix_y)/u.m
            ymin = min(geometry.pix_y)/u.m

            fig.clf()
            axbig = fig.add_subplot()
            label_x = 'X'
            label_y = 'Y'
            axbig.set_xlabel(label_x)
            axbig.set_ylabel(label_y)
            im = axbig.imshow(clean_image_2d,origin='lower',extent=(xmin,xmax,ymin,ymax))
            cbar = fig.colorbar(im)
            axbig.scatter(0., 0., s=90, facecolors='none', edgecolors='r', marker='o')
            if np.cos(angle*u.rad)>0.:
                line_x = np.linspace(image_center_x, xmax, 100)
                line_y = -(line_a*line_x + line_b)
                axbig.plot(line_x,line_y,color='w',alpha=0.3,linestyle='dashed')
            else:
                line_x = np.linspace(xmin, image_center_x, 100)
                line_y = -(line_a*line_x + line_b)
                axbig.plot(line_x,line_y,color='w',alpha=0.3,linestyle='dashed')
            axbig.set_xlim(xmin,xmax)
            axbig.set_ylim(ymin,ymax)
            txt = axbig.text(-0.35, 0.35, 'image size = %0.2e'%(image_size), fontdict=font)
            txt = axbig.text(-0.35, 0.32, 'image direction = %0.2e'%(image_direction), fontdict=font)
            txt = axbig.text(-0.35, 0.29, 'time direction = %0.2e'%(time_direction), fontdict=font)
            txt = axbig.text(-0.35, 0.26, 'image quality = %0.2e'%(image_qual), fontdict=font)
            if image_qual<1.:
                txt = axbig.text(-0.35, 0.23, 'bad image!!!', fontdict=font)
            fig.savefig(f'{ctapipe_output}/output_plots/evt{event_id}_tel{tel_id}_clean_image.png',bbox_inches='tight')
            axbig.remove()

            fig.clf()
            axbig = fig.add_subplot()
            label_x = 'X'
            label_y = 'Y'
            axbig.set_xlabel(label_x)
            axbig.set_ylabel(label_y)
            im = axbig.imshow(clean_time_2d,origin='lower',extent=(xmin,xmax,ymin,ymax))
            cbar = fig.colorbar(im)
            axbig.scatter(0., 0., s=90, facecolors='none', edgecolors='r', marker='o')
            fig.savefig(f'{ctapipe_output}/output_plots/evt{event_id}_tel{tel_id}_clean_time.png',bbox_inches='tight')
            axbig.remove()

            waveform = event.dl0.tel[tel_id].waveform
            n_pix, n_samp = waveform.shape
            print (f'tel_id = {tel_id}, n_pix = {n_pix}, n_samp = {n_samp}')

            #for pix in range(0,n_pix):
            #    #if image_mask[pix]: continue # select noise
            #    if not image_mask[pix]: continue # select signal
            #    pix_waveform = waveform[pix,:]
            #    fig.clf()
            #    axbig = fig.add_subplot()
            #    axbig.plot(pix_waveform)
            #    fig.savefig(f'{ctapipe_output}/output_plots/evt{event_id}_tel{tel_id}_pix{pix}_waveform.png',bbox_inches='tight')
            #    axbig.remove()

            max_sample = 40
            window_size = 8
            n_windows = 5
            clean_movie_1d = []
            for win in range(0,n_windows):
                clean_movie_1d += [np.zeros_like(clean_image_1d)]
            for pix in range(0,n_pix):
                if not image_mask[pix]: continue # select signal
                for win in range(0,n_windows):
                    for sample in range(0,window_size):
                        sample_idx = sample + win*window_size
                        clean_movie_1d[win][pix] +=  waveform[pix,sample_idx]

            whole_movie_1d = []
            for win in range(0,n_windows):

                clean_movie_2d = geometry.image_to_cartesian_representation(clean_movie_1d[win])
                remove_nan_pixels(clean_movie_2d)

                shift_movie_2d = image_translation(clean_movie_2d, round(float(shift_pix_y)), round(float(shift_pix_x)))
                rotate_movie_2d = image_rotation(shift_movie_2d, angle*u.rad)
                rotate_movie_1d = geometry.image_from_cartesian_representation(rotate_movie_2d)
                whole_movie_1d.extend(rotate_movie_1d)
                print (f'len(whole_movie_1d) = {len(whole_movie_1d)}')

                if image_qual>100.:
                    fig.clf()
                    axbig = fig.add_subplot()
                    label_x = 'X'
                    label_y = 'Y'
                    axbig.set_xlabel(label_x)
                    axbig.set_ylabel(label_y)
                    im = axbig.imshow(rotate_movie_2d,origin='lower',extent=(xmin,xmax,ymin,ymax))
                    cbar = fig.colorbar(im)
                    fig.savefig(f'{ctapipe_output}/output_plots/evt{event_id}_tel{tel_id}_win{win}_clean_movie.png',bbox_inches='tight')
                    axbig.remove()

            big_movie_matrix += [whole_movie_1d]
            big_moment_matrix += [image_moment_array]
            big_truth_matrix += [truth_info_array]


training_sample_path = sys.argv[1]

ctapipe_output = os.environ.get("CTAPIPE_OUTPUT_PATH")
subprocess.call(['sh', './clean_plots.sh'])

run_save_training_matrix(training_sample_path,min_energy=0.1,max_energy=100.0,max_evt=1e10)

