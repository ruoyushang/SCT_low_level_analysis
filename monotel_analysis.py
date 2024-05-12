
import os, sys
import subprocess
import glob

from operator import itemgetter

import time
import pickle
import numpy as np
from astropy import units as u
from matplotlib import pyplot as plt
from matplotlib import colors
from scipy.optimize import least_squares, minimize

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
plot_monotel_reconstruction = common_functions.plot_monotel_reconstruction
camxy_to_altaz = common_functions.camxy_to_altaz
image_simulation = common_functions.image_simulation
movie_simulation = common_functions.movie_simulation
display_a_movie = common_functions.display_a_movie
make_a_gif = common_functions.make_a_gif
linear_model = common_functions.linear_model

n_bins_arrival = common_functions.n_bins_arrival
arrival_lower = common_functions.arrival_lower
arrival_upper = common_functions.arrival_upper
n_bins_impact = common_functions.n_bins_impact
impact_lower = common_functions.impact_lower
impact_upper = common_functions.impact_upper
n_bins_xmax = common_functions.n_bins_xmax
xmax_lower = common_functions.xmax_lower
xmax_upper = common_functions.xmax_upper
n_bins_height = common_functions.n_bins_height
height_lower = common_functions.height_lower
height_upper = common_functions.height_upper
n_bins_energy = common_functions.n_bins_energy
log_energy_lower = common_functions.log_energy_lower
log_energy_upper = common_functions.log_energy_upper

#image_size_cut = 500.

fig, ax = plt.subplots()
figsize_x = 8.6
figsize_y = 6.4
fig.set_figheight(figsize_y)
fig.set_figwidth(figsize_x)

time_weight_ratio = 1.

use_movie = False
ana_tag = 'image'
if use_movie:
    ana_tag = 'movie'

init_lookup_table_type = 'box3d'
lookup_table_type = 'box3d'
ana_tag += f'_{lookup_table_type}'

is_training = 0 # not training sample
#is_training = 1 # is training sample 
#is_training = 2 # all sample 
if is_training==1:
    ana_tag += '_train'

#do_it_fast = True
do_it_fast = False
if do_it_fast:
    ana_tag += '_fast'


select_event_id = 0
select_tel_id = 0
#select_event_id = 71904
#select_tel_id = 21

def sqaure_difference_between_1d_images(init_params,data_latent_space,lookup_table,eigen_vectors,full_table=False):

    fit_arrival = init_params[0]
    fit_impact = init_params[1]
    fit_log_energy = init_params[2]
    if not full_table:
        if lookup_table[0].get_bin_content(fit_arrival,fit_impact,fit_log_energy)==0.:
            return 1e10

    fit_latent_space = []
    for r in range(0,len(lookup_table)):
        fit_latent_space += [lookup_table[r].get_bin_content(fit_arrival,fit_impact,fit_log_energy)]
    fit_latent_space = np.array(fit_latent_space)

    sum_chi2 = 0.
    n_rows = len(data_latent_space)
    for row in range(0,n_rows):
        if data_latent_space[row]==0. and fit_latent_space[row]==0.: continue
        diff = data_latent_space[row] - fit_latent_space[row]
        sum_chi2 += diff*diff

    return sum_chi2

def sqaure_difference_between_1d_images_poisson(init_params,image_1d_data,lookup_table,eigen_vectors,full_table=False):

    fit_arrival = init_params[0]
    fit_impact = init_params[1]
    fit_log_energy = init_params[2]
    if not full_table:
        if lookup_table[0].get_bin_content(fit_arrival,fit_impact,fit_log_energy)==0.:
            return 1e10

    fit_latent_space = []
    for r in range(0,len(lookup_table)):
        fit_latent_space += [lookup_table[r].get_bin_content(fit_arrival,fit_impact,fit_log_energy)]
    fit_latent_space = np.array(fit_latent_space)

    data_latent_space = eigen_vectors @ image_1d_data

    #sum_chi2 = 0.
    #n_rows = len(data_latent_space)
    #for row in range(0,n_rows):
    #    if data_latent_space[row]==0. and fit_latent_space[row]==0.: continue
    #    diff = data_latent_space[row] - fit_latent_space[row]
    #    sum_chi2 += diff*diff

    sum_chi2 = 0.
    image_1d_fit = eigen_vectors.T @ fit_latent_space
    n_rows = len(image_1d_fit)
    for row in range(0,n_rows):
        n_expect = max(0.0001,image_1d_fit[row])
        n_data = max(0.,image_1d_data[row])
        if n_data==0.:
            sum_chi2 += n_expect
        else:
            sum_chi2 += -1.*(n_data*np.log(n_expect) - n_expect - (n_data*np.log(n_data)-n_data))

    return sum_chi2


def sortFirst(val):
    return val[0]

def box_search(init_params,image_latent_space,image_lookup_table,image_eigen_vectors,time_latent_space,time_lookup_table,time_eigen_vectors,arrival_range,impact_range,log_energy_range):

    image_norm = np.sum(np.abs(image_latent_space))
    time_norm = np.sum(np.abs(time_latent_space))

    init_arrival = init_params[0]
    init_impact = init_params[1]
    init_log_energy = init_params[2]
    short_list = []


    while len(short_list)==0:

        fit_idx_x = 0
        fit_idx_y = 0
        fit_idx_z = 0
        for idx_x  in range(0,n_bins_arrival):
            try_arrival = image_lookup_table[0].xaxis[idx_x]
            if abs(init_arrival-try_arrival)>arrival_range:
                continue
            for idx_y  in range(0,n_bins_impact):
                try_impact = image_lookup_table[0].yaxis[idx_y]
                if abs(init_impact-try_impact)>impact_range:
                    continue
                for idx_z  in range(0,n_bins_energy):
                    try_log_energy = image_lookup_table[0].zaxis[idx_z]
                    if abs(init_log_energy-try_log_energy)>log_energy_range:
                        continue

                    try_params = [try_arrival,try_impact,try_log_energy]

                    try_chi2_image = sqaure_difference_between_1d_images(try_params,image_latent_space,image_lookup_table,image_eigen_vectors)/image_norm
                    try_chi2_time = sqaure_difference_between_1d_images(try_params,time_latent_space,time_lookup_table,time_eigen_vectors)/time_norm
                    try_chi2 = try_chi2_image+try_chi2_time

                    short_list += [(try_chi2,try_arrival,try_impact,try_log_energy)]

        if len(short_list)==0:
            print ('short_list is zero. expand search range.')
            arrival_range = 1e10
            impact_range = 1e10
            log_energy_range = 1e10
        else:
            break

    short_list.sort(key=sortFirst)


    return short_list

def analyze_short_list(short_list,init_params,input_movie_1d,movie_lookup_table,movie_eigen_vectors):

    fit_chi2 = 1e10
    n_short_list = 10
    fit_arrival = init_params[0]
    fit_impact = init_params[1]
    fit_log_energy = init_params[2]
    for entry in range(0,min(n_short_list,len(short_list))):
        try_arrival = short_list[entry][1]
        try_impact = short_list[entry][2]
        try_log_energy = short_list[entry][3]
        init_params = [try_arrival,try_impact,try_log_energy]
        try_chi2 = sqaure_difference_between_1d_images_poisson(init_params,input_movie_1d,movie_lookup_table,movie_eigen_vectors)
        if try_chi2<fit_chi2:
            fit_chi2 = try_chi2
            fit_arrival = try_arrival
            fit_impact = try_impact
            fit_log_energy = try_log_energy

    return fit_arrival, fit_impact, fit_log_energy, fit_chi2

def single_movie_reconstruction(input_image_1d,image_lookup_table,image_eigen_vectors,input_time_1d,time_lookup_table,time_eigen_vectors,input_movie_1d,movie_lookup_table,movie_eigen_vectors,movie_lookup_table_poly):

    global arrival_upper
    global arrival_lower
    global n_bins_arrival
    global impact_upper
    global impact_lower
    global n_bins_impact
    global log_energy_upper
    global log_energy_lower
    global n_bins_energy
    #n_bins_arrival = len(image_lookup_table[0].xaxis)
    #n_bins_impact = len(image_lookup_table[0].yaxis)
    #n_bins_energy = len(image_lookup_table[0].zaxis)

    chi2_cut = 1.5

    arrival_step_size = (arrival_upper-arrival_lower)/float(n_bins_arrival)
    impact_step_size = (impact_upper-impact_lower)/float(n_bins_impact)
    log_energy_step_size = (log_energy_upper-log_energy_lower)/float(n_bins_energy)

    #movie_latent_space = movie_eigen_vectors @ input_movie_1d
    image_latent_space = image_eigen_vectors @ input_image_1d
    time_latent_space = time_eigen_vectors @ input_time_1d
    combine_latent_space = np.concatenate((image_latent_space, time_latent_space))

    image_size = np.sum(input_movie_1d)
    fit_arrival = linear_model(combine_latent_space, movie_lookup_table_poly[0])
    fit_impact = linear_model(combine_latent_space, movie_lookup_table_poly[1])
    fit_log_energy = linear_model(combine_latent_space, movie_lookup_table_poly[2])
    cov_arrival = 0.
    cov_impact = 0.
    cov_log_energy = 0.
    init_params = [fit_arrival,fit_impact,fit_log_energy]
    fit_chi2 = 0.
    print (f'initial fit_arrival = {fit_arrival}, fit_impact = {fit_impact}, fit_log_energy = {fit_log_energy}')

    if do_it_fast:
        return fit_arrival, fit_impact, fit_log_energy, pow(cov_arrival,0.5), pow(cov_impact,0.5), pow(cov_log_energy,0.5), fit_chi2

    arrival_range = 1.5*arrival_step_size
    impact_range = 1.5*impact_step_size
    log_energy_range = 100.5*log_energy_step_size
    init_params = [fit_arrival,fit_impact,fit_log_energy]
    short_list = box_search(init_params,image_latent_space,image_lookup_table,image_eigen_vectors,time_latent_space,time_lookup_table,time_eigen_vectors,arrival_range,impact_range,log_energy_range)

    fit_chi2 = short_list[0][0]
    fit_arrival = short_list[0][1]
    fit_impact = short_list[0][2]
    fit_log_energy = short_list[0][3]

    if not use_movie:
        return fit_arrival, fit_impact, fit_log_energy, pow(cov_arrival,0.5), pow(cov_impact,0.5), pow(cov_log_energy,0.5), fit_chi2

    fit_arrival, fit_impact, fit_log_energy, fit_chi2 = analyze_short_list(short_list,init_params,input_movie_1d,movie_lookup_table,movie_eigen_vectors)

    normalized_chi2 = fit_chi2/image_size
    print (f'normalized_chi2 = {normalized_chi2}')

    is_good_result = True
    if normalized_chi2>chi2_cut:
        print ('chi2 is bad.')
        is_good_result = False

    if is_good_result:
        return fit_arrival, fit_impact, fit_log_energy, pow(cov_arrival,0.5), pow(cov_impact,0.5), pow(cov_log_energy,0.5), fit_chi2

    arrival_range = 5.5*arrival_step_size
    impact_range = 5.5*impact_step_size
    log_energy_range = 100.5*log_energy_step_size
    init_params = [fit_arrival,fit_impact,fit_log_energy]
    short_list = box_search(init_params,image_latent_space,image_lookup_table,image_eigen_vectors,time_latent_space,time_lookup_table,time_eigen_vectors,arrival_range,impact_range,log_energy_range)

    fit_arrival, fit_impact, fit_log_energy, fit_chi2 = analyze_short_list(short_list,init_params,input_movie_1d,movie_lookup_table,movie_eigen_vectors)

    normalized_chi2 = fit_chi2/image_size
    print (f'normalized_chi2 = {normalized_chi2}')

    is_good_result = True
    if normalized_chi2>chi2_cut:
        print ('chi2 is bad.')
        is_good_result = False

    if is_good_result:
        return fit_arrival, fit_impact, fit_log_energy, pow(cov_arrival,0.5), pow(cov_impact,0.5), pow(cov_log_energy,0.5), fit_chi2

    arrival_range = 100.5*arrival_step_size
    impact_range = 100.5*impact_step_size
    log_energy_range = 100.5*log_energy_step_size
    init_params = [fit_arrival,fit_impact,fit_log_energy]
    short_list = box_search(init_params,image_latent_space,image_lookup_table,image_eigen_vectors,time_latent_space,time_lookup_table,time_eigen_vectors,arrival_range,impact_range,log_energy_range)

    fit_arrival, fit_impact, fit_log_energy, fit_chi2 = analyze_short_list(short_list,init_params,input_movie_1d,movie_lookup_table,movie_eigen_vectors)

    normalized_chi2 = fit_chi2/image_size
    print (f'normalized_chi2 = {normalized_chi2}')


    cov_arrival = 0.
    cov_impact = 0.
    cov_log_energy = 0.

    print (f'Final fit_arrival = {fit_arrival}, fit_impact = {fit_impact}, fit_log_energy = {fit_log_energy}')

    cov_arrival = 0.
    cov_impact = 0.
    cov_log_energy = 0.

    #tic_a = time.perf_counter()

    #heavy_axis = movie_lookup_table[0].get_heaviest_axis()
    #if fit_arrival>arrival_upper or fit_arrival<arrival_lower:
    #    fit_arrival = heavy_axis[0]
    #if fit_impact>impact_upper or fit_impact<impact_lower:
    #    fit_impact = heavy_axis[1]
    #if fit_log_energy>log_energy_upper or fit_log_energy<log_energy_lower:
    #    fit_log_energy = heavy_axis[2]
    #init_params = [fit_arrival,fit_impact,fit_log_energy]
    #stepsize = [1.0*arrival_step_size,1.0*impact_step_size,1.0*log_energy_step_size]
    #solution = minimize(
    #    sqaure_difference_between_1d_images_poisson,
    #    x0=init_params,
    #    args=(input_movie_1d,movie_lookup_table,movie_eigen_vectors),
    #    method='L-BFGS-B',
    #    #method='Nelder-Mead',
    #    #jac=None,
    #    #options={'eps':stepsize,'ftol':0.000001},
    #    options={'eps':stepsize},
    #)
    #fit_params = solution['x']
    #fit_arrival = fit_params[0]
    #fit_impact = fit_params[1]
    #fit_log_energy = fit_params[2]
    #cov_arrival = 0.
    #cov_impact = 0.
    #cov_log_energy = 0.
    #fit_chi2 = 0.
    #print (f'final fit_arrival = {fit_arrival}, fit_impact = {fit_impact}, fit_log_energy = {fit_log_energy}')
    #return fit_arrival+0.5*arrival_step_size, fit_impact+0.5*impact_step_size, fit_log_energy+0.5*log_energy_step_size, pow(cov_arrival,0.5), pow(cov_impact,0.5), pow(cov_log_energy,0.5), fit_chi2


    return fit_arrival+0.5*arrival_step_size, fit_impact+0.5*impact_step_size, fit_log_energy+0.5*log_energy_step_size, pow(cov_arrival,0.5), pow(cov_impact,0.5), pow(cov_log_energy,0.5), fit_chi2



def run_monotel_analysis(training_sample_path, telescope_type, min_energy=0.1, max_energy=1000., max_evt=1e10):

    analysis_result = []

    print ('loading svd pickle data... ')
    movie_lookup_table_poly_pkl = []
    output_filename = f'{ctapipe_output}/output_machines/polynomial_lookup_table_arrival_{telescope_type}.pkl'
    movie_lookup_table_poly_pkl += [pickle.load(open(output_filename, "rb"))]
    output_filename = f'{ctapipe_output}/output_machines/polynomial_lookup_table_impact_{telescope_type}.pkl'
    movie_lookup_table_poly_pkl += [pickle.load(open(output_filename, "rb"))]
    output_filename = f'{ctapipe_output}/output_machines/polynomial_lookup_table_log_energy_{telescope_type}.pkl'
    movie_lookup_table_poly_pkl += [pickle.load(open(output_filename, "rb"))]

    movie_lookup_table_pkl = None
    movie_eigen_vectors_pkl = None
    if use_movie:
        output_filename = f'{ctapipe_output}/output_machines/movie_{lookup_table_type}_lookup_table_{telescope_type}.pkl'
        movie_lookup_table_pkl = pickle.load(open(output_filename, "rb"))
        output_filename = f'{ctapipe_output}/output_machines/movie_eigen_vectors_{telescope_type}.pkl'
        movie_eigen_vectors_pkl = pickle.load(open(output_filename, "rb"))

    output_filename = f'{ctapipe_output}/output_machines/image_{init_lookup_table_type}_lookup_table_{telescope_type}.pkl'
    image_lookup_table_pkl = pickle.load(open(output_filename, "rb"))
    output_filename = f'{ctapipe_output}/output_machines/image_eigen_vectors_{telescope_type}.pkl'
    image_eigen_vectors_pkl = pickle.load(open(output_filename, "rb"))
    
    output_filename = f'{ctapipe_output}/output_machines/time_{init_lookup_table_type}_lookup_table_{telescope_type}.pkl'
    time_lookup_table_pkl = pickle.load(open(output_filename, "rb"))
    output_filename = f'{ctapipe_output}/output_machines/time_eigen_vectors_{telescope_type}.pkl'
    time_eigen_vectors_pkl = pickle.load(open(output_filename, "rb"))

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

    evt_count = 0
    good_analysis = 0
    avg_analysis = 0
    bad_analysis = 0
    for event in source:
    
        event_id = event.index['event_id']
    
        evt_count += 1
        if is_training==0:
            if (evt_count % training_event_select)==0: continue
        elif is_training==1:
            if (evt_count % training_event_select)!=0: continue

        ntel = len(event.r0.tel)
        
        calib(event)  # fills in r1, dl0, and dl1
        #image_processor(event) # Takes DL1/Image data and cleans and parametrizes the images into DL1/parameters. Should be run after CameraCalibrator.
        #shower_processor(event) # Run the stereo event reconstruction on the input events.
    
        for tel_idx in range(0,len(list(event.dl0.tel.keys()))):

            tel_id = list(event.dl0.tel.keys())[tel_idx]

            if select_event_id!=0:
                if event_id!=select_event_id: continue
            if select_tel_id!=0:
                if tel_id!=select_tel_id: continue

            print ('====================================================================================')
            print (f'event_id = {event_id}, tel_id = {tel_id}')

            truth_info_array = find_image_truth(source, subarray, run_id, tel_id, event)
            truth_energy = float(truth_info_array[0]/u.TeV)
            truth_core_x = truth_info_array[1]
            truth_core_y = truth_info_array[2]
            truth_alt = float(truth_info_array[3]/u.rad)
            truth_az = float(truth_info_array[4]/u.rad)
            truth_height = truth_info_array[5]
            truth_xmax = truth_info_array[6]
            star_cam_x = truth_info_array[7]
            star_cam_y = truth_info_array[8]
            impact_x = truth_info_array[9]
            impact_y = truth_info_array[10]
            focal_length = source.subarray.tel[tel_id].optics.equivalent_focal_length/u.m


            is_edge_image, lightcone, image_moment_array, eco_image_1d, eco_time_1d = make_standard_image(fig, subarray, run_id, tel_id, event)
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
            print (f'image_size = {image_size}')

            if image_size<image_size_cut: 
                print ('failed image_size_cut')
                continue
            #if is_edge_image:
            #    print ('failed: edge image.')
            #    continue

            truth_projection = image_moment_array[10]
            print (f'truth_projection = {truth_projection:0.3f}')

            image_fit_arrival = 0.
            image_fit_impact = 0.
            image_fit_log_energy = 0.
            image_fit_arrival_err = 0.
            image_fit_impact_err = 0.
            image_fit_log_energy_err = 0.
            image_fit_chi2 = 1e10

            print ('==================================================================')
            tic_task = time.perf_counter()

            is_edge_image, lightcone, image_moment_array, eco_movie_1d = make_a_movie(fig, subarray, run_id, tel_id, event, make_plots=False)

            image_fit_arrival, image_fit_impact, image_fit_log_energy, image_fit_arrival_err, image_fit_impact_err, image_fit_log_energy_err, image_fit_chi2 = single_movie_reconstruction(eco_image_1d,image_lookup_table_pkl,image_eigen_vectors_pkl,eco_time_1d,time_lookup_table_pkl,time_eigen_vectors_pkl,eco_movie_1d,movie_lookup_table_pkl,movie_eigen_vectors_pkl,movie_lookup_table_poly_pkl)

            toc_task = time.perf_counter()
            print (f'Image reconstruction completed in {toc_task - tic_task:0.4f} sec.')
            print ('==================================================================')

            image_fit_cam_x = image_center_x + image_fit_arrival*np.cos(angle*u.rad)
            image_fit_cam_y = image_center_y + image_fit_arrival*np.sin(angle*u.rad)
            image_fit_alt, image_fit_az = camxy_to_altaz(source, subarray, run_id, tel_id, image_fit_cam_x, image_fit_cam_y)

            image_method_unc = image_fit_arrival_err/focal_length*180./np.pi
            image_method_error = pow(pow(image_fit_cam_x-star_cam_x,2)+pow(image_fit_cam_y-star_cam_y,2),0.5)/focal_length*180./np.pi

            print (f'focal_length = {focal_length}')
            print (f'lightcone = {lightcone}')
            print (f'star_cam_x = {star_cam_x}')
            print (f'star_cam_y = {star_cam_y}')
            print (f'image_fit_cam_x = {image_fit_cam_x}')
            print (f'image_fit_cam_y = {image_fit_cam_y}')
            print (f'truth_alt = {truth_alt}')
            print (f'truth_az = {truth_az}')
            print (f'image_fit_alt = {image_fit_alt}')
            print (f'image_fit_az = {image_fit_az}')

            print (f'truth_energy     = {truth_energy}')
            print (f'image_fit_energy = {pow(10.,image_fit_log_energy)}')
            print (f'truth_fit_impact = {pow(impact_x*impact_x+impact_y*impact_y,0.5)}')
            print (f'image_fit_impact = {image_fit_impact}')
            print (f'image_method_error = {image_method_error:0.3f} deg')
            print (f'image_method_unc = {image_method_unc:0.3f} deg')

            fit_arrival = image_fit_arrival
            fit_impact = image_fit_impact
            fit_log_energy = image_fit_log_energy
            fit_cam_x = image_fit_cam_x
            fit_cam_y = image_fit_cam_y
            fit_alt = image_fit_alt
            fit_az = image_fit_az

            #if image_size>image_size_cut:
            if image_size>5000. or select_event_id!=0:

                if 'image' in ana_tag:

                    fit_params = [fit_arrival,fit_impact,fit_log_energy]
                    plot_monotel_reconstruction(fig, subarray, run_id, tel_id, event, image_moment_array, star_cam_x, star_cam_y, fit_cam_x, fit_cam_y, 'image')
                    if not do_it_fast:
                        image_simulation(fig, subarray, run_id, tel_id, event, fit_params, image_lookup_table_pkl, image_eigen_vectors_pkl, time_lookup_table_pkl, time_eigen_vectors_pkl)

                if 'movie' in ana_tag:

                    fit_params = [fit_arrival,fit_impact,fit_log_energy]
                    plot_monotel_reconstruction(fig, subarray, run_id, tel_id, event, image_moment_array, star_cam_x, star_cam_y, fit_cam_x, fit_cam_y, 'movie')
                    sim_image, sim_movie = movie_simulation(fig, subarray, run_id, tel_id, event, fit_params, movie_lookup_table_pkl, movie_eigen_vectors_pkl)
                    data_image, data_movie = display_a_movie(fig, subarray, run_id, tel_id, event, len(eco_image_1d), eco_movie_1d)
                    make_a_gif(fig, subarray, run_id, tel_id, event, data_image, data_movie, sim_movie)

            evt_header = [training_sample_path,event_id,tel_id]
            evt_geometry = [image_size,lightcone,focal_length,image_direction,time_direction]
            evt_truth = [truth_energy,truth_alt,truth_az,star_cam_x,star_cam_y,truth_projection]
            evt_model = [pow(10.,fit_log_energy),fit_alt,fit_az,fit_cam_x,fit_cam_y,image_fit_chi2]

            single_analysis_result = [evt_header,evt_geometry,evt_truth,evt_model]
            analysis_result += [single_analysis_result]


    if select_event_id!=0: 
        print ('No file saved.')
        return

    output_filename = f'{ctapipe_output}/output_analysis/{ana_tag}_run{run_id}_{telescope_type}.pkl'
    print (f'writing file to {output_filename}')
    with open(output_filename,"wb") as file:
        pickle.dump(analysis_result, file)

    return

training_sample_path = sys.argv[1]
telescope_type = sys.argv[2]

ctapipe_output = os.environ.get("CTAPIPE_OUTPUT_PATH")
#subprocess.call(['sh', './clean_plots.sh'])

run_monotel_analysis(training_sample_path,telescope_type,min_energy=0.1,max_energy=100.0,max_evt=1e10)
print ('Job completed.')
exit()

