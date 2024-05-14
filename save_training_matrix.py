
import os, sys
import subprocess
import glob
import tracemalloc

import time
import numpy as np
from astropy import units as u
from matplotlib import pyplot as plt
from matplotlib import colors
import pickle

from ctapipe.io import EventSource, SimTelEventSource, HDF5TableWriter
from ctapipe.calib import CameraCalibrator
from ctapipe.reco import ShowerProcessor
from ctapipe.image import ImageProcessor
from traitlets.config import Config

import common_functions
training_event_select = common_functions.training_event_select
image_size_cut = common_functions.image_size_cut
remove_nan_pixels = common_functions.remove_nan_pixels
reset_time = common_functions.reset_time
find_image_moments = common_functions.find_image_moments
image_translation = common_functions.image_translation
image_rotation = common_functions.image_rotation
find_image_truth = common_functions.find_image_truth
make_a_movie = common_functions.make_a_movie
make_standard_image = common_functions.make_standard_image

fig, ax = plt.subplots()
figsize_x = 8.6
figsize_y = 6.4
fig.set_figheight(figsize_y)
fig.set_figwidth(figsize_x)

# start memory profiling
tracemalloc.start()

def run_save_training_matrix(training_sample_path,telescope_type, min_energy=0.1, max_energy=1000., max_evt=1e10):

    big_movie_matrix = []
    big_image_matrix = []
    big_time_matrix = []
    big_moment_matrix = []
    big_truth_matrix = []

    print (f'loading file: {training_sample_path}')
    source = SimTelEventSource(training_sample_path, focal_length_choice='EQUIVALENT')
    
    # Explore the instrument description
    subarray = source.subarray
    print ('Array info:')
    print (subarray.info())
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

    evt_count = 0
    for event in source:
    
        event_id = event.index['event_id']

        evt_count += 1
        if (evt_count % training_event_select)!=0: continue
    
        ntel = len(event.r0.tel)
        
        calib(event)  # fills in r1, dl0, and dl1
        #image_processor(event) # Takes DL1/Image data and cleans and parametrizes the images into DL1/parameters. Should be run after CameraCalibrator.
        #shower_processor(event) # Run the stereo event reconstruction on the input events.
    
        for tel_idx in range(0,len(list(event.dl0.tel.keys()))):

            tel_id = list(event.dl0.tel.keys())[tel_idx]
            #if event_id!=23003: continue
            #if tel_id!=20: continue

            if str(telescope_type)!=str(source.subarray.tel[tel_id]): continue

            print ('====================================================================================')
            print (f'Select telescope type: {telescope_type}')
            print (f'event_id = {event_id}, tel_id = {tel_id}')
            print("TEL{:03}: {}".format(tel_id, source.subarray.tel[tel_id]))

            tic_img = time.perf_counter()

            truth_info_array = find_image_truth(source, subarray, run_id, tel_id, event)
            star_cam_x = truth_info_array[7]
            star_cam_y = truth_info_array[8]
            star_cam_xy = [star_cam_x,star_cam_y]

            is_edge_image, lightcone, image_moment_array, eco_image_1d, eco_time_1d = make_standard_image(fig, telescope_type, subarray, run_id, tel_id, event, star_cam_xy=star_cam_xy, make_plots=False)

            image_size = image_moment_array[0]
            print (f'image_size = {image_size:0.3f}')

            if image_size<image_size_cut:
                print ('failed image_size_cut.')
                continue
            if is_edge_image:
                print ('failed: edge image.')
                continue

            is_edge_image, lightcone, image_moment_array, eco_movie_1d = make_a_movie(fig, telescope_type, subarray, run_id, tel_id, event, star_cam_xy=star_cam_xy, make_plots=False)

            if image_size>image_size_cut:
                big_movie_matrix += [eco_movie_1d]
                big_image_matrix += [eco_image_1d]
                big_time_matrix += [eco_time_1d]
                big_moment_matrix += [image_moment_array]
                big_truth_matrix += [truth_info_array]
                #exit()

            toc_img = time.perf_counter()
            print (f'Image analysis completed in {toc_img - tic_img:0.4f} sec.')
    
            print(f'memory usage (current,peak) = {tracemalloc.get_traced_memory()}')


    ana_tag = 'training_sample'
    output_filename = f'{ctapipe_output}/output_samples/{ana_tag}_run{run_id}_{telescope_type}.pkl'
    print (f'writing file to {output_filename}')
    with open(output_filename,"wb") as file:
        pickle.dump([big_truth_matrix,big_moment_matrix,big_image_matrix,big_time_matrix,big_movie_matrix], file)

    return

tic = time.perf_counter()

training_sample_path = sys.argv[1]
telescope_type = sys.argv[2]

ctapipe_output = os.environ.get("CTAPIPE_OUTPUT_PATH")
#subprocess.call(['sh', './clean_plots.sh'])

run_save_training_matrix(training_sample_path,telescope_type,min_energy=0.1,max_energy=100.0,max_evt=1e10)

toc = time.perf_counter()

print (f'Job completed in {toc - tic:0.4f} sec.')
exit()

