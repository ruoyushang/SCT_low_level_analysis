
import os, sys
import subprocess
import glob

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
remove_nan_pixels = common_functions.remove_nan_pixels
reset_time = common_functions.reset_time
find_image_moments = common_functions.find_image_moments
image_translation = common_functions.image_translation
image_rotation = common_functions.image_rotation
find_image_truth = common_functions.find_image_truth
make_a_movie = common_functions.make_a_movie
plot_monotel_reconstruction = common_functions.plot_monotel_reconstruction

fig, ax = plt.subplots()
figsize_x = 8.6
figsize_y = 6.4
fig.set_figheight(figsize_y)
fig.set_figwidth(figsize_x)


def sqaure_difference_between_1d_images(init_params,image_1d_data,lookup_table,eigen_vectors):

    fit_arrival = init_params[0]
    fit_xmax = init_params[1]
    fit_log_energy = init_params[2]
    if lookup_table[0].get_bin_content(fit_arrival,fit_xmax,fit_log_energy)==0.:
        return 1e10

    fit_latent_space = []
    for r in range(0,len(lookup_table)):
        fit_latent_space += [lookup_table[r].get_bin_content(fit_arrival,fit_xmax,fit_log_energy)]
    fit_latent_space = np.array(fit_latent_space)

    #image_1d_fit = eigen_vectors.T @ fit_latent_space
    #sum_chi2 = 0.
    #n_rows = len(image_1d_fit)
    #for row in range(0,n_rows):
    #    diff = image_1d_data[row] - image_1d_fit[row]
    #    error_sq = max(1.,image_1d_data[row])
    #    sum_chi2 += diff*diff/error_sq

    data_latent_space = eigen_vectors @ image_1d_data
    sum_chi2 = 0.
    n_rows = len(data_latent_space)
    for row in range(0,n_rows):
        diff = data_latent_space[row] - fit_latent_space[row]
        sum_chi2 += diff*diff

    return sum_chi2

def single_image_reconstruction(input_image_1d,lookup_table,eigen_vectors):

    fit_arrival = 0.1
    fit_xmax = 250.
    fit_log_energy = 0.
    init_params = [fit_arrival,fit_xmax,fit_log_energy]
    fit_chi2 = sqaure_difference_between_1d_images(init_params,input_image_1d,lookup_table,eigen_vectors)

    n_bins_arrival = len(lookup_table[0].xaxis)
    n_bins_xmax = len(lookup_table[0].yaxis)
    n_bins_energy = len(lookup_table[0].zaxis)

    for idx_x  in range(0,n_bins_arrival):
        for idx_y  in range(0,n_bins_xmax):
            for idx_z  in range(0,n_bins_energy):

                try_arrival = lookup_table[0].xaxis[idx_x]
                try_xmax = lookup_table[0].yaxis[idx_y]
                try_log_energy = lookup_table[0].zaxis[idx_z]
                init_params = [try_arrival,try_xmax,try_log_energy]

                try_chi2 = sqaure_difference_between_1d_images(init_params,input_image_1d,lookup_table,eigen_vectors)

                if try_chi2<fit_chi2:
                    fit_chi2 = try_chi2
                    fit_arrival = try_arrival
                    fit_xmax = try_xmax
                    fit_log_energy = try_log_energy

    return fit_arrival+0.005, fit_xmax+11.25, fit_log_energy+0.05, fit_chi2


def run_monotel_analysis(training_sample_path, min_energy=0.1, max_energy=1000., max_evt=1e10):

    print ('loading svd pickle data... ')
    output_filename = f'{ctapipe_output}/output_machines/lookup_table.pkl'
    lookup_table_pkl = pickle.load(open(output_filename, "rb"))
    
    output_filename = f'{ctapipe_output}/output_machines/eigen_vectors.pkl'
    eigen_vectors_pkl = pickle.load(open(output_filename, "rb"))
    
    movie_rank = len(lookup_table_pkl)

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
        
        calib(event)  # fills in r1, dl0, and dl1
        #image_processor(event) # Takes DL1/Image data and cleans and parametrizes the images into DL1/parameters. Should be run after CameraCalibrator.
        #shower_processor(event) # Run the stereo event reconstruction on the input events.
    
        for tel_idx in range(0,len(list(event.dl0.tel.keys()))):

            tel_id = list(event.dl0.tel.keys())[tel_idx]
            print ('====================================================================================')
            print (f'event_id = {event_id}, tel_id = {tel_id}')

            truth_info_array = find_image_truth(source, subarray, run_id, tel_id, event)
            truth_energy = truth_info_array[0]
            truth_core_x = truth_info_array[1]
            truth_core_y = truth_info_array[2]
            truth_alt = truth_info_array[3]
            truth_az = truth_info_array[4]
            truth_height = truth_info_array[5]
            truth_xmax = truth_info_array[6]
            star_cam_x = truth_info_array[7]
            star_cam_y = truth_info_array[8]
            impact_x = truth_info_array[9]
            impact_y = truth_info_array[10]

            image_qual, image_moment_array, whole_movie_1d = make_a_movie(fig, subarray, run_id, tel_id, event)
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

            if image_qual<1.: continue
            if image_size<100.: continue

            fit_arrival, fit_xmax, fit_log_energy, fit_chi2 = single_image_reconstruction(whole_movie_1d,lookup_table_pkl,eigen_vectors_pkl)
            fit_cam_x = image_center_x + fit_arrival*np.cos(angle*u.rad)
            fit_cam_y = image_center_y + fit_arrival*np.sin(angle*u.rad)

            print (f'image_qual = {image_qual}')
            print (f'truth_energy = {truth_energy}')
            print (f'fit_energy = {pow(10.,fit_log_energy)}')
            print (f'star_cam_x = {star_cam_x}')
            print (f'star_cam_y = {star_cam_y}')
            print (f'fit_cam_x = {fit_cam_x}')
            print (f'fit_cam_y = {fit_cam_y}')

            plot_monotel_reconstruction(fig, subarray, run_id, tel_id, event, image_moment_array, fit_cam_x, fit_cam_y)


    #ana_tag = 'training_sample'
    #output_filename = f'{ctapipe_output}/output_samples/{ana_tag}_run{run_id}.pkl'
    #print (f'writing file to {output_filename}')
    #with open(output_filename,"wb") as file:
    #    pickle.dump([big_truth_matrix,big_moment_matrix,big_movie_matrix], file)

    return

training_sample_path = sys.argv[1]

ctapipe_output = os.environ.get("CTAPIPE_OUTPUT_PATH")
#subprocess.call(['sh', './clean_plots.sh'])

run_monotel_analysis(training_sample_path,min_energy=0.1,max_energy=100.0,max_evt=1e10)
print ('Job completed.')
exit()

