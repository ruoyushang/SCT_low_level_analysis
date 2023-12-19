import os, sys
import subprocess
import glob

import time
import math
import numpy as np
from astropy import units as u
from scipy.optimize import least_squares, minimize, brute, dual_annealing
from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib import patheffects
import pickle

from ctapipe import utils
from ctapipe.utils.datasets import get_dataset_path
from ctapipe.calib import CameraCalibrator
from ctapipe.image import hillas_parameters, tailcuts_clean, ImageProcessor
from ctapipe.reco import ShowerProcessor
from ctapipe.io import EventSource, SimTelEventSource, HDF5TableWriter
from ctapipe.visualization import CameraDisplay

from load_cta_data import load_training_samples
from load_cta_data import rank_brightest_telescope
from load_cta_data import image_translation
from load_cta_data import image_rotation
from load_cta_data import NeuralNetwork
from load_cta_data import sqaure_difference_between_1d_images
from load_cta_data import find_image_moments_guided
from load_cta_data import get_average
from load_cta_data import MyArray2D
from load_cta_data import signle_image_reconstruction

ctapipe_output = os.environ.get("CTAPIPE_OUTPUT_PATH")
#subprocess.call(['sh', './clean_plots.sh'])

testing_sample_path = sys.argv[1]

fig, ax = plt.subplots()
figsize_x = 8.6
figsize_y = 6.4
fig.set_figheight(figsize_y)
fig.set_figwidth(figsize_x)

font = {'family': 'serif', 'color':  'black', 'weight': 'normal', 'size': 10, 'rotation': 0.,}

image_size_cut = 100.
make_plots = False


print ('loading svd pickle data... ')
output_filename = f'{ctapipe_output}/output_machines/lookup_table.pkl'
lookup_table_pkl = pickle.load(open(output_filename, "rb"))

output_filename = f'{ctapipe_output}/output_machines/lookup_table_time.pkl'
lookup_table_time_pkl = pickle.load(open(output_filename, "rb"))

output_filename = f'{ctapipe_output}/output_machines/eigen_vectors.pkl'
eigen_vectors_pkl = pickle.load(open(output_filename, "rb"))

output_filename = f'{ctapipe_output}/output_machines/eigen_vectors_time.pkl'
eigen_vectors_time_pkl = pickle.load(open(output_filename, "rb"))

output_filename = f'{ctapipe_output}/output_machines/lookup_table_impact.pkl'
lookup_table_impact_pkl = pickle.load(open(output_filename, "rb"))

output_filename = f'{ctapipe_output}/output_machines/lookup_table_impact_rms.pkl'
lookup_table_impact_rms_pkl = pickle.load(open(output_filename, "rb"))

rank = len(lookup_table_pkl)

for r in range(0,rank):
    fig.clf()
    axbig = fig.add_subplot()
    label_x = 'Arrival'
    label_y = 'log Energy [TeV]'
    axbig.set_xlabel(label_x)
    axbig.set_ylabel(label_y)
    xmin = lookup_table_pkl[r].xaxis.min()
    xmax = lookup_table_pkl[r].xaxis.max()
    ymin = lookup_table_pkl[r].yaxis.min()
    ymax = lookup_table_pkl[r].yaxis.max()
    im = axbig.imshow(lookup_table_pkl[r].zaxis[:,:].T,origin='lower',extent=(xmin,xmax,ymin,ymax))
    cbar = fig.colorbar(im)
    fig.savefig(f'{ctapipe_output}/output_plots/lookup_table_rank{r}_t0.png',bbox_inches='tight')
    axbig.remove()

fig.clf()
axbig = fig.add_subplot()
label_x = 'Arrival'
label_y = 'log Energy [TeV]'
axbig.set_xlabel(label_x)
axbig.set_ylabel(label_y)
xmin = lookup_table_impact_pkl.xaxis.min()
xmax = lookup_table_impact_pkl.xaxis.max()
ymin = lookup_table_impact_pkl.yaxis.min()
ymax = lookup_table_impact_pkl.yaxis.max()
im = axbig.imshow(lookup_table_impact_pkl.zaxis[:,:].T,origin='lower',extent=(xmin,xmax,ymin,ymax),aspect='auto')
cbar = fig.colorbar(im)
fig.savefig(f'{ctapipe_output}/output_plots/lookup_table_impact_t0.png',bbox_inches='tight')
axbig.remove()

fig.clf()
axbig = fig.add_subplot()
label_x = 'Arrival'
label_y = 'log Energy [TeV]'
axbig.set_xlabel(label_x)
axbig.set_ylabel(label_y)
xmin = lookup_table_impact_rms_pkl.xaxis.min()
xmax = lookup_table_impact_rms_pkl.xaxis.max()
ymin = lookup_table_impact_rms_pkl.yaxis.min()
ymax = lookup_table_impact_rms_pkl.yaxis.max()
im = axbig.imshow(lookup_table_impact_rms_pkl.zaxis[:,:].T,origin='lower',extent=(xmin,xmax,ymin,ymax),aspect='auto')
cbar = fig.colorbar(im)
fig.savefig(f'{ctapipe_output}/output_plots/lookup_table_impact_rms_t0.png',bbox_inches='tight')
axbig.remove()

## neural network for lookup table
#
#nn_input = []
#nn_output = []
#for img in range(0,len(big_training_image_matrix)):
#
#    img_moment = big_training_moment_matrix[img]
#    delta_foci_time = img_moment[7]
#
#    arrival = big_training_param_matrix[img][0]
#    log_energy = big_training_param_matrix[img][1]
#    impact = big_training_param_matrix[img][2]
#    image = np.array(big_training_image_matrix[img])
#    latent_space = eigen_vectors_pkl @ image
#
#    new_entry_input = latent_space+[delta_foci_time]
#    new_entry_output = [arrival,log_energy,impact]
#    nn_input += [new_entry_input]
#    nn_output += [new_entry_output]
#
#nn_input = np.array(nn_input)
#nn_output = np.array(nn_output)
#
#learning_rate = 0.1
#n_node = 20
#nn_lookup_table = NeuralNetwork(nn_input[0],nn_output[0],learning_rate,n_node)
#training_error = nn_lookup_table.train(nn_input, nn_output, 10000)

#fig.clf()
#axbig = fig.add_subplot()
#label_x = 'Iterations'
#label_y = 'Error'
#axbig.set_xlabel(label_x)
#axbig.set_ylabel(label_y)
#axbig.plot(training_error)
#fig.savefig(f'{ctapipe_output}/output_plots/training_error.png',bbox_inches='tight')
#axbig.remove()


log_energy_truth_filt = []
arrival_hist_error_filt = []
log_energy_truth = []
impact_truth = []
arrival_truth = []
log_energy_hist_guess = []
impact_hist_guess = []
arrival_hist_guess = []
arrival_hist_error = []
delta_time = []
image_size = []
semi_major = []
delta_time_good = []
image_size_good = []
delta_time_bad = []
image_size_bad = []

source = SimTelEventSource(testing_sample_path, focal_length_choice='EQUIVALENT')
subarray = source.subarray
ob_keys = source.observation_blocks.keys()
run_id = list(ob_keys)[0]
output_filename = f'{ctapipe_output}/output_samples/testing_sample_noisy_clean_origin_run{run_id}.pkl'
#output_filename = f'{ctapipe_output}/output_samples/training_sample_noisy_clean_repose_run{run_id}.pkl'
print (f'loading pickle trainging sample data: {output_filename}')
if not os.path.exists(output_filename):
    print (f'file does not exist.')
    exit()
training_sample = pickle.load(open(output_filename, "rb"))

training_id_list = training_sample[0]
training_telesc_position_matrix = training_sample[1]
training_truth_shower_position_matrix = training_sample[2]
train_cam_axes = training_sample[3]
big_training_image_matrix = training_sample[4]
big_training_time_matrix = training_sample[5]
big_training_param_matrix = training_sample[6]
big_training_moment_matrix = training_sample[7]

big_training_image_matrix = np.array(big_training_image_matrix)
big_training_time_matrix = np.array(big_training_time_matrix)
big_training_param_matrix = np.array(big_training_param_matrix)
big_training_moment_matrix = np.array(big_training_moment_matrix)

for img in range(0,len(training_id_list)):

    focal_length = training_telesc_position_matrix[img][4]

    current_run = training_id_list[img][0]
    current_event = training_id_list[img][1]
    current_tel_id = training_id_list[img][2]
    subarray = training_id_list[img][3]
    geom = subarray.tel[current_tel_id].camera.geometry
    cam_axes = train_cam_axes[img]

    image_center_x = big_training_moment_matrix[img][0]
    image_center_y = big_training_moment_matrix[img][1]
    image_foci_x1 = big_training_moment_matrix[img][2]
    image_foci_y1 = big_training_moment_matrix[img][3]
    image_foci_x2 = big_training_moment_matrix[img][4]
    image_foci_y2 = big_training_moment_matrix[img][5]
    center_time = big_training_moment_matrix[img][6]
    delta_foci_time = big_training_moment_matrix[img][7]
    semi_major_sq = big_training_moment_matrix[img][8]
    semi_minor_sq = big_training_moment_matrix[img][9]
    foci_r1 = pow(image_foci_x1*image_foci_x1+image_foci_y1*image_foci_y1,0.5)
    foci_r2 = pow(image_foci_x2*image_foci_x2+image_foci_y2*image_foci_y2,0.5)

    shower_param = big_training_param_matrix[img]
    shower_arrival = shower_param[0]
    shower_log_energy = shower_param[1]
    shower_impact = shower_param[2]
    
    #if delta_foci_time>30.: continue
    #if delta_foci_time<0.: continue
    #if foci_r2-foci_r1<0.: continue

    truth_image = big_training_image_matrix[img]
    truth_time = big_training_time_matrix[img]
    size = np.sum(truth_image)

    sim_arrival = big_training_param_matrix[img][0]
    sim_log_energy = big_training_param_matrix[img][1]
    sim_impact = big_training_param_matrix[img][2]
    print ('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print (f'img = {img}')
    print (f'image_size = {size}')

    #if img!=2: continue
    if sim_log_energy<0.: continue
    if size<image_size_cut: continue

    truth_image_2d = geom.image_to_cartesian_representation(truth_image)
    truth_time_2d = geom.image_to_cartesian_representation(truth_time)
    num_rows, num_cols = truth_image_2d.shape
    for row in range(0,num_rows):
        for col in range(0,num_cols):
            if math.isnan(truth_image_2d[row,col]): 
                truth_image_2d[row,col] = 0.
                truth_time_2d[row,col] = 0.

    image_center_x, image_center_y, image_foci_x1, image_foci_y1, image_foci_x2, image_foci_y2, center_time, delta_foci_time, semi_major_sq, semi_minor_sq = find_image_moments_guided(truth_image_2d, truth_time_2d, cam_axes[0], cam_axes[1])
    image_foci_x = image_foci_x1
    image_foci_y = image_foci_y1

    latent_space = []
    for r in range(0,rank):
        latent_space += [lookup_table_pkl[r].get_bin_content(sim_arrival,sim_log_energy)]
    latent_space = np.array(latent_space)

    sim_image = eigen_vectors_pkl.T @ latent_space
    sim_image_2d = geom.image_to_cartesian_representation(sim_image)

    latent_space_time = []
    for r in range(0,rank):
        latent_space_time += [lookup_table_time_pkl[r].get_bin_content(sim_arrival,sim_log_energy)]
    latent_space_time = np.array(latent_space_time)

    sim_time = eigen_vectors_time_pkl.T @ latent_space_time
    sim_time_2d = geom.image_to_cartesian_representation(sim_time)

    for row in range(0,num_rows):
        for col in range(0,num_cols):
            if math.isnan(sim_image_2d[row,col]): 
                sim_image_2d[row,col] = 0.
                sim_time_2d[row,col] = 0.

    fit_arrival, fit_impact, fit_log_energy = signle_image_reconstruction(truth_image,truth_time,geom,cam_axes,lookup_table_pkl,eigen_vectors_pkl,lookup_table_time_pkl,eigen_vectors_time_pkl,lookup_table_impact_pkl,lookup_table_impact_rms_pkl)

    print (f'sim_log_energy = {sim_log_energy}')
    print (f'fit_log_energy = {fit_log_energy}')
    print (f'sim_impact = {sim_impact}')
    print (f'fit_impact = {fit_impact}')
    print (f'sim_arrival = {sim_arrival}')
    print (f'fit_arrival = {fit_arrival}')

    angle_rad = np.arctan2(image_foci_y-image_center_y,image_foci_x-image_center_x)
    delta_x = fit_arrival*np.cos(-angle_rad)
    delta_y = fit_arrival*np.sin(-angle_rad)
    guess_cam_x = image_center_x + delta_x
    guess_cam_y = image_center_y - delta_y
    print (f'guess_cam_x = {guess_cam_x}')
    print (f'guess_cam_y = {guess_cam_y}')

    image_size += [np.log10(size)]
    log_energy_truth += [sim_log_energy]
    impact_truth += [sim_impact]
    arrival_truth += [sim_arrival]
    log_energy_hist_guess += [fit_log_energy]
    impact_hist_guess += [fit_impact]
    arrival_hist_guess += [fit_arrival]
    arrival_hist_error += [(fit_arrival-sim_arrival)/focal_length*180./np.pi]
    delta_time += [delta_foci_time]
    semi_major += [pow(semi_major_sq,0.5)]

    if abs(fit_arrival-sim_arrival)<0.05:
        delta_time_good += [delta_foci_time]
        image_size_good += [size]
    else:
        delta_time_bad += [delta_foci_time]
        image_size_bad += [size]

    if size>image_size_cut:
        log_energy_truth_filt += [sim_log_energy]
        arrival_hist_error_filt += [(fit_arrival-sim_arrival)/focal_length*180./np.pi]

    log_energy_axis = []
    for x in range(0,6):
        log_energy_axis += [(-1+x*0.5)]
    hist_sky_err_vs_energy = get_average(log_energy_truth_filt,pow(np.array(arrival_hist_error_filt),2),log_energy_axis)
    hist_sky_err_vs_energy.yaxis = pow(np.array(hist_sky_err_vs_energy.yaxis),0.5)
    print (f'hist_sky_err_vs_energy.yaxis = {hist_sky_err_vs_energy.yaxis}')

    log_size_axis = []
    for x in range(0,8):
        log_size_axis += [(1.5+x*0.5)]
    hist_sky_err_vs_size = get_average(image_size,pow(np.array(arrival_hist_error),2),log_size_axis)
    hist_sky_err_vs_size.yaxis = pow(np.array(hist_sky_err_vs_size.yaxis),0.5)
    print (f'hist_sky_err_vs_size.yaxis = {hist_sky_err_vs_size.yaxis}')

    fit_latent_space = []
    for r in range(0,rank):
        fit_latent_space += [lookup_table_pkl[r].get_bin_content(fit_arrival,fit_log_energy)]
    fit_latent_space = np.array(fit_latent_space)

    fit_latent_space_time = []
    for r in range(0,rank):
        fit_latent_space_time += [lookup_table_time_pkl[r].get_bin_content(fit_arrival,fit_log_energy)]
    fit_latent_space_time = np.array(fit_latent_space_time)

    fit_image = eigen_vectors_pkl.T @ fit_latent_space
    fit_image_2d = geom.image_to_cartesian_representation(fit_image)

    fit_time = eigen_vectors_time_pkl.T @ fit_latent_space_time
    fit_time_2d = geom.image_to_cartesian_representation(fit_time)

    for row in range(0,num_rows):
        for col in range(0,num_cols):
            if math.isnan(fit_image_2d[row,col]): 
                fit_image_2d[row,col] = 0.
                fit_time_2d[row,col] = 0.

    fit_image_derotate = np.zeros_like(fit_image_2d)
    fit_time_derotate = np.zeros_like(fit_time_2d)
    angle_rad = np.arctan2(image_foci_y-image_center_y,image_foci_x-image_center_x)
    fit_image_derotate = image_rotation(fit_image_2d, cam_axes[0], cam_axes[1], angle_rad)
    fit_time_derotate = image_rotation(fit_time_2d, cam_axes[0], cam_axes[1], angle_rad)
    
    fit_image_decenter = np.zeros_like(fit_image_2d)
    fit_time_decenter = np.zeros_like(fit_time_2d)
    shift_x = image_center_x
    shift_y = image_center_y
    fit_image_decenter = image_translation(fit_image_derotate, cam_axes[0], cam_axes[1], shift_x, shift_y)
    fit_time_decenter = image_translation(fit_time_derotate, cam_axes[0], cam_axes[1], shift_x, shift_y)


    if make_plots:
        #if abs(fit_arrival-sim_arrival)>0.04:
        if (img % 10 == 0):

            fig.clf()
            axbig = fig.add_subplot()
            label_x = 'X'
            label_y = 'Y'
            axbig.set_xlabel(label_x)
            axbig.set_ylabel(label_y)
            im = axbig.imshow(fit_image_decenter,origin='lower')
            #im = axbig.imshow(fit_image_2d,origin='lower')
            cbar = fig.colorbar(im)
            txt = axbig.text(10., 105., 'fit log energy = %0.2f'%(fit_log_energy), fontdict=font)
            txt.set_path_effects([patheffects.withStroke(linewidth=2, foreground='w')])
            txt = axbig.text(10., 100., 'fit impact = %0.2f'%(fit_impact), fontdict=font)
            txt.set_path_effects([patheffects.withStroke(linewidth=2, foreground='w')])
            txt = axbig.text(10., 95., 'fit arrival = %0.2f'%(fit_arrival), fontdict=font)
            txt.set_path_effects([patheffects.withStroke(linewidth=2, foreground='w')])
            fig.savefig(f'{ctapipe_output}/output_plots/evt_{img}_image_fit.png',bbox_inches='tight')
            axbig.remove()

            fig.clf()
            axbig = fig.add_subplot()
            label_x = 'X'
            label_y = 'Y'
            axbig.set_xlabel(label_x)
            axbig.set_ylabel(label_y)
            im = axbig.imshow(fit_time_decenter,origin='lower')
            #im = axbig.imshow(fit_time_2d,origin='lower')
            cbar = fig.colorbar(im)
            txt = axbig.text(10., 105., 'fit log energy = %0.2f'%(fit_log_energy), fontdict=font)
            txt.set_path_effects([patheffects.withStroke(linewidth=2, foreground='w')])
            txt = axbig.text(10., 100., 'fit impact = %0.2f'%(fit_impact), fontdict=font)
            txt.set_path_effects([patheffects.withStroke(linewidth=2, foreground='w')])
            txt = axbig.text(10., 95., 'fit arrival = %0.2f'%(fit_arrival), fontdict=font)
            txt.set_path_effects([patheffects.withStroke(linewidth=2, foreground='w')])
            fig.savefig(f'{ctapipe_output}/output_plots/evt_{img}_time_fit.png',bbox_inches='tight')
            axbig.remove()

            #fig.clf()
            #axbig = fig.add_subplot()
            #label_x = 'X'
            #label_y = 'Y'
            #axbig.set_xlabel(label_x)
            #axbig.set_ylabel(label_y)
            #im = axbig.imshow(sim_image_2d,origin='lower')
            #cbar = fig.colorbar(im)
            #txt = axbig.text(10., 105., 'truth log energy = %0.2f'%(sim_log_energy), fontdict=font)
            #txt.set_path_effects([patheffects.withStroke(linewidth=2, foreground='w')])
            #txt = axbig.text(10., 100., 'truth impact = %0.2f'%(sim_impact), fontdict=font)
            #txt.set_path_effects([patheffects.withStroke(linewidth=2, foreground='w')])
            #txt = axbig.text(10., 95., 'truth arrival = %0.2f'%(sim_arrival), fontdict=font)
            #txt.set_path_effects([patheffects.withStroke(linewidth=2, foreground='w')])
            #fig.savefig(f'{ctapipe_output}/output_plots/evt_{img}_image_sim.png',bbox_inches='tight')
            #axbig.remove()

            #fig.clf()
            #axbig = fig.add_subplot()
            #label_x = 'X'
            #label_y = 'Y'
            #axbig.set_xlabel(label_x)
            #axbig.set_ylabel(label_y)
            #im = axbig.imshow(sim_time_2d,origin='lower')
            #cbar = fig.colorbar(im)
            #txt = axbig.text(10., 105., 'truth log energy = %0.2f'%(sim_log_energy), fontdict=font)
            #txt.set_path_effects([patheffects.withStroke(linewidth=2, foreground='w')])
            #txt = axbig.text(10., 100., 'truth impact = %0.2f'%(sim_impact), fontdict=font)
            #txt.set_path_effects([patheffects.withStroke(linewidth=2, foreground='w')])
            #txt = axbig.text(10., 95., 'truth arrival = %0.2f'%(sim_arrival), fontdict=font)
            #txt.set_path_effects([patheffects.withStroke(linewidth=2, foreground='w')])
            #fig.savefig(f'{ctapipe_output}/output_plots/evt_{img}_time_sim.png',bbox_inches='tight')
            #axbig.remove()

            fig.clf()
            axbig = fig.add_subplot()
            label_x = 'X'
            label_y = 'Y'
            axbig.set_xlabel(label_x)
            axbig.set_ylabel(label_y)
            im = axbig.imshow(truth_image_2d,origin='lower')
            #im = axbig.imshow(image_data_rotate,origin='lower')
            cbar = fig.colorbar(im)
            txt = axbig.text(10., 105., 'truth log energy = %0.2f'%(sim_log_energy), fontdict=font)
            txt.set_path_effects([patheffects.withStroke(linewidth=2, foreground='w')])
            txt = axbig.text(10., 100., 'truth impact = %0.2f'%(sim_impact), fontdict=font)
            txt.set_path_effects([patheffects.withStroke(linewidth=2, foreground='w')])
            txt = axbig.text(10., 95., 'truth arrival = %0.2f'%(sim_arrival), fontdict=font)
            txt.set_path_effects([patheffects.withStroke(linewidth=2, foreground='w')])
            fig.savefig(f'{ctapipe_output}/output_plots/evt_{img}_image_truth.png',bbox_inches='tight')
            axbig.remove()

            fig.clf()
            axbig = fig.add_subplot()
            label_x = 'X'
            label_y = 'Y'
            axbig.set_xlabel(label_x)
            axbig.set_ylabel(label_y)
            im = axbig.imshow(truth_time_2d,origin='lower')
            #im = axbig.imshow(time_data_rotate,origin='lower')
            cbar = fig.colorbar(im)
            txt = axbig.text(10., 105., 'truth log energy = %0.2f'%(sim_log_energy), fontdict=font)
            txt.set_path_effects([patheffects.withStroke(linewidth=2, foreground='w')])
            txt = axbig.text(10., 100., 'truth impact = %0.2f'%(sim_impact), fontdict=font)
            txt.set_path_effects([patheffects.withStroke(linewidth=2, foreground='w')])
            txt = axbig.text(10., 95., 'truth arrival = %0.2f'%(sim_arrival), fontdict=font)
            txt.set_path_effects([patheffects.withStroke(linewidth=2, foreground='w')])
            fig.savefig(f'{ctapipe_output}/output_plots/evt_{img}_time_truth.png',bbox_inches='tight')
            axbig.remove()

            #exit()


single_tel_ana_output = []
single_tel_ana_output += [log_energy_truth_filt]
single_tel_ana_output += [arrival_hist_error_filt]
single_tel_ana_output += [log_energy_truth]
single_tel_ana_output += [impact_truth]
single_tel_ana_output += [arrival_truth]
single_tel_ana_output += [log_energy_hist_guess]
single_tel_ana_output += [impact_hist_guess]
single_tel_ana_output += [arrival_hist_guess]
single_tel_ana_output += [arrival_hist_error]
single_tel_ana_output += [image_size]

output_filename = f'{ctapipe_output}/output_analysis/single_tel_ana_output_run{run_id}.pkl'
with open(output_filename,"wb") as file:
    pickle.dump(single_tel_ana_output, file)

