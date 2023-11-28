import os
import subprocess
import glob

import time
import math
import numpy as np
from astropy import units as u
from scipy.optimize import least_squares, minimize, brute, dual_annealing
from matplotlib import pyplot as plt
from matplotlib import colors
import pickle

from ctapipe import utils
from ctapipe.utils.datasets import get_dataset_path
from ctapipe.calib import CameraCalibrator
from ctapipe.image import hillas_parameters, tailcuts_clean, ImageProcessor
from ctapipe.reco import ShowerProcessor
from ctapipe.io import EventSource, SimTelEventSource, HDF5TableWriter
from ctapipe.visualization import CameraDisplay

from load_cta_data import load_training_samples
from load_cta_data import NeuralNetwork
from load_cta_data import MyArray2D
from load_cta_data import sqaure_difference_between_1d_images

ctapipe_output = os.environ.get("CTAPIPE_OUTPUT_PATH")
subprocess.call(['sh', './clean_plots.sh'])

fig, ax = plt.subplots()
figsize_x = 8.6
figsize_y = 6.4
fig.set_figheight(figsize_y)
fig.set_figwidth(figsize_x)


training_sample_path = []
#training_sample_path += [get_dataset_path("gamma_20deg_0deg_run853___cta-prod3-sct_desert-2150m-Paranal-SCT.simtel.gz")]
with open('train_sim_files.txt', 'r') as file:
    for line in file:
        training_sample_path += [get_dataset_path(line.strip('\n'))]

old_training_id_list = []
old_training_telesc_position_matrix = []
old_training_truth_shower_position_matrix = []
old_train_cam_axes = []
old_big_training_image_matrix = []
old_big_training_time_matrix = []
old_big_training_param_matrix = []
old_big_training_moment_matrix = []
for path in range(0,len(training_sample_path)):
    source = SimTelEventSource(training_sample_path[path], focal_length_choice='EQUIVALENT')
    subarray = source.subarray
    ob_keys = source.observation_blocks.keys()
    run_id = list(ob_keys)[0]
    output_filename = f'{ctapipe_output}/output_samples/training_sample_noisy_clean_repose_run{run_id}.pkl'
    #output_filename = f'{ctapipe_output}/output_samples/training_sample_truth_dirty_repose_run{run_id}.pkl'
    print (f'loading pickle trainging sample data: {output_filename}')
    if not os.path.exists(output_filename):
        print (f'file does not exist.')
        continue
    training_sample = pickle.load(open(output_filename, "rb"))

    old_training_id_list += training_sample[0]
    old_training_telesc_position_matrix += training_sample[1]
    old_training_truth_shower_position_matrix += training_sample[2]
    old_train_cam_axes += training_sample[3]
    old_big_training_image_matrix += training_sample[4]
    old_big_training_time_matrix += training_sample[5]
    old_big_training_param_matrix += training_sample[6]
    old_big_training_moment_matrix += training_sample[7]

all_delta_foci_r = []
all_delta_time = []
all_log_energy = []
all_impact = []
training_id_list = []
training_telesc_position_matrix = []
training_truth_shower_position_matrix = []
train_cam_axes = []
big_training_image_matrix = []
big_training_time_matrix = []
big_training_param_matrix = []
big_training_moment_matrix = []
for img in range(0,len(old_big_training_image_matrix)):

    img_moment = old_big_training_moment_matrix[img]
    image_center_x = img_moment[0]
    image_center_y = img_moment[1]
    image_foci_x1 = img_moment[2]
    image_foci_y1 = img_moment[3]
    image_foci_x2 = img_moment[4]
    image_foci_y2 = img_moment[5]
    center_time = img_moment[6]
    delta_foci_time = img_moment[7]
    semi_major_sq = img_moment[8]
    semi_minor_sq = img_moment[9]
    foci_r1 = pow(image_foci_x1*image_foci_x1+image_foci_y1*image_foci_y1,0.5)
    foci_r2 = pow(image_foci_x2*image_foci_x2+image_foci_y2*image_foci_y2,0.5)

    shower_param = old_big_training_param_matrix[img]
    shower_arrival = shower_param[0]
    shower_log_energy = shower_param[1]
    shower_impact = shower_param[2]
    
    if delta_foci_time>30.: continue
    if delta_foci_time<0.: continue
    if foci_r2-foci_r1<0.: continue
    training_id_list += [old_training_id_list[img]]
    training_telesc_position_matrix += [old_training_telesc_position_matrix[img]]
    training_truth_shower_position_matrix += [old_training_truth_shower_position_matrix[img]]
    train_cam_axes += [old_train_cam_axes[img]]
    big_training_image_matrix += [old_big_training_image_matrix[img]]
    big_training_time_matrix += [old_big_training_time_matrix[img]]
    big_training_param_matrix += [old_big_training_param_matrix[img]]
    big_training_moment_matrix += [old_big_training_moment_matrix[img]]

    all_delta_time += [delta_foci_time]
    all_delta_foci_r += [foci_r2-foci_r1]
    all_log_energy += [shower_log_energy]
    all_impact += [shower_impact]

all_delta_foci_r = np.array(all_delta_foci_r)
all_delta_time = np.array(all_delta_time)
all_log_energy = np.array(all_log_energy)
all_impact = np.array(all_impact)

fig.clf()
axbig = fig.add_subplot()
label_x = 'delta time'
label_y = 'delta foci r'
axbig.set_xlabel(label_x)
axbig.set_ylabel(label_y)
axbig.scatter(all_delta_time, all_delta_foci_r, s=90, c='r', marker='+')
axbig.set_xlim(1e-3,50.)
axbig.set_xscale('log')
fig.savefig(f'{ctapipe_output}/output_plots/delta_time_vs_delta_foci_r.png',bbox_inches='tight')
axbig.remove()

fig.clf()
axbig = fig.add_subplot()
label_x = 'impact'
label_y = 'delta foci t'
axbig.set_xlabel(label_x)
axbig.set_ylabel(label_y)
axbig.scatter(all_log_energy, all_delta_time, s=90, c='r', marker='+')
fig.savefig(f'{ctapipe_output}/output_plots/log_energy_vs_delta_foci_t.png',bbox_inches='tight')
axbig.remove()

fig.clf()
axbig = fig.add_subplot()
label_x = 'impact'
label_y = 'delta foci t'
axbig.set_xlabel(label_x)
axbig.set_ylabel(label_y)
axbig.scatter(all_impact, all_delta_time, s=90, c='r', marker='+')
fig.savefig(f'{ctapipe_output}/output_plots/impact_vs_delta_foci_t.png',bbox_inches='tight')
axbig.remove()

fig.clf()
axbig = fig.add_subplot()
label_x = 'impact'
label_y = 'delta foci r'
axbig.set_xlabel(label_x)
axbig.set_ylabel(label_y)
axbig.scatter(all_impact, all_delta_foci_r, s=90, c='r', marker='+')
fig.savefig(f'{ctapipe_output}/output_plots/impact_vs_delta_foci_r.png',bbox_inches='tight')
axbig.remove()

big_training_image_matrix = np.array(big_training_image_matrix)
big_training_time_matrix = np.array(big_training_time_matrix)
big_training_param_matrix = np.array(big_training_param_matrix)
big_training_moment_matrix = np.array(big_training_moment_matrix)
big_training_time_matrix = big_training_time_matrix * big_training_image_matrix



print ('loading svd pickle data... ')
output_filename = f'{ctapipe_output}/output_machines/lookup_table.pkl'
lookup_table_pkl = pickle.load(open(output_filename, "rb"))

output_filename = f'{ctapipe_output}/output_machines/eigen_vectors.pkl'
eigen_vectors_pkl = pickle.load(open(output_filename, "rb"))

output_filename = f'{ctapipe_output}/output_machines/lookup_table_arrival.pkl'
lookup_table_arrival_pkl = pickle.load(open(output_filename, "rb"))

output_filename = f'{ctapipe_output}/output_machines/lookup_table_arrival_rms.pkl'
lookup_table_arrival_rms_pkl = pickle.load(open(output_filename, "rb"))

rank = len(lookup_table_pkl)

for r in range(0,rank):
    fig.clf()
    axbig = fig.add_subplot()
    label_x = 'Impact [km]'
    label_y = 'log Energy [TeV]'
    axbig.set_xlabel(label_x)
    axbig.set_ylabel(label_y)
    xmin = lookup_table_pkl[r].xaxis.min()
    xmax = lookup_table_pkl[r].xaxis.max()
    ymin = lookup_table_pkl[r].yaxis.min()
    ymax = lookup_table_pkl[r].yaxis.max()
    im = axbig.imshow(lookup_table_pkl[r].waxis[:,:,0].T,origin='lower',extent=(xmin,xmax,ymin,ymax))
    cbar = fig.colorbar(im)
    fig.savefig(f'{ctapipe_output}/output_plots/lookup_table_rank{r}_t0.png',bbox_inches='tight')
    axbig.remove()


fig.clf()
axbig = fig.add_subplot()
label_x = 'Impact [km]'
label_y = 'log Energy [TeV]'
axbig.set_xlabel(label_x)
axbig.set_ylabel(label_y)
xmin = lookup_table_arrival_pkl.xaxis.min()
xmax = lookup_table_arrival_pkl.xaxis.max()
ymin = lookup_table_arrival_pkl.yaxis.min()
ymax = lookup_table_arrival_pkl.yaxis.max()
im = axbig.imshow(lookup_table_arrival_pkl.waxis[:,:,0].T,origin='lower',extent=(xmin,xmax,ymin,ymax),aspect='auto')
cbar = fig.colorbar(im)
fig.savefig(f'{ctapipe_output}/output_plots/lookup_table_arrival_t0.png',bbox_inches='tight')
axbig.remove()



fig.clf()
axbig = fig.add_subplot()
label_x = 'Impact [km]'
label_y = 'log Energy [TeV]'
axbig.set_xlabel(label_x)
axbig.set_ylabel(label_y)
xmin = lookup_table_arrival_rms_pkl.xaxis.min()
xmax = lookup_table_arrival_rms_pkl.xaxis.max()
ymin = lookup_table_arrival_rms_pkl.yaxis.min()
ymax = lookup_table_arrival_rms_pkl.yaxis.max()
im = axbig.imshow(lookup_table_arrival_rms_pkl.waxis[:,:,0].T,origin='lower',extent=(xmin,xmax,ymin,ymax),aspect='auto')
cbar = fig.colorbar(im)
fig.savefig(f'{ctapipe_output}/output_plots/lookup_table_arrival_rms_t0.png',bbox_inches='tight')
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


log_energy_truth = []
impact_truth = []
arrival_truth = []
log_energy_hist_guess = []
impact_hist_guess = []
arrival_hist_guess = []
arrival_hist_error = []
delta_time = []
image_size = []
delta_time_good = []
image_size_good = []
delta_time_bad = []
image_size_bad = []
for img in range(0,len(training_id_list)):

    current_run = training_id_list[img][0]
    current_event = training_id_list[img][1]
    current_tel_id = training_id_list[img][2]
    subarray = training_id_list[img][3]
    geom = subarray.tel[current_tel_id].camera.geometry

    truth_image = big_training_image_matrix[img]
    truth_image_2d = geom.image_to_cartesian_representation(truth_image)

    sim_arrival = big_training_param_matrix[img][0]
    sim_log_energy = big_training_param_matrix[img][1]
    sim_impact = big_training_param_matrix[img][2]
    print ('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print (f'img = {img}')
    print (f'sim_log_energy = {sim_log_energy}')
    print (f'sim_impact = {sim_impact}')
    #if img!=2: continue
    if sim_log_energy<0.: continue

    image_foci_x1 = big_training_moment_matrix[img][2]
    image_foci_y1 = big_training_moment_matrix[img][3]
    image_foci_x2 = big_training_moment_matrix[img][4]
    image_foci_y2 = big_training_moment_matrix[img][5]
    delta_foci_time = np.log10(max(1e-3,big_training_moment_matrix[img][7]))
    delta_foci_r = pow(pow(image_foci_x1-image_foci_x2,2)+pow(image_foci_y1-image_foci_y2,2),0.5)

    latent_space = []
    for r in range(0,rank):
        latent_space += [lookup_table_pkl[r].get_bin_content(sim_impact,sim_log_energy,delta_foci_time)]
    latent_space = np.array(latent_space)

    sim_image = eigen_vectors_pkl.T @ latent_space
    sim_image_2d = geom.image_to_cartesian_representation(sim_image)

    fit_log_energy = 0.
    fit_impact = 0.1
    init_params = [fit_log_energy,fit_impact,delta_foci_time]
    fit_chi2 = sqaure_difference_between_1d_images(init_params,train_cam_axes[img],geom,truth_image,lookup_table_pkl,eigen_vectors_pkl)
    print (f'init_log_energy = {fit_log_energy}')
    print (f'init_impact = {fit_impact}')
    print (f'init_chi2 = {fit_chi2}')
    n_bins_energy = len(lookup_table_pkl[0].yaxis)
    n_bins_impact = len(lookup_table_pkl[0].xaxis)
    for idx_x  in range(0,n_bins_impact):
        for idx_y  in range(0,n_bins_energy):
            try_log_energy = lookup_table_pkl[0].yaxis[idx_y]
            try_impact = lookup_table_pkl[0].xaxis[idx_x]
            init_params = [try_log_energy,try_impact,delta_foci_time]
            try_chi2 = sqaure_difference_between_1d_images(init_params,train_cam_axes[img],geom,truth_image,lookup_table_pkl,eigen_vectors_pkl)
            if try_chi2<fit_chi2:
                fit_chi2 = try_chi2
                fit_log_energy = try_log_energy
                fit_impact = try_impact
    print (f'fit_log_energy = {fit_log_energy}')
    print (f'fit_impact = {fit_impact}')
    print (f'fit_chi2 = {fit_chi2}')

    fit_arrival = lookup_table_arrival_pkl.get_bin_content(fit_impact,fit_log_energy,delta_foci_time)

    log_energy_truth += [sim_log_energy]
    impact_truth += [sim_impact]
    arrival_truth += [sim_arrival]
    log_energy_hist_guess += [fit_log_energy]
    impact_hist_guess += [fit_impact]
    arrival_hist_guess += [fit_arrival]
    arrival_hist_error += [fit_arrival-sim_arrival]
    delta_time += [delta_foci_time]


    size = 0.
    for pix in range(0,len(sim_image)):
        size += sim_image[pix]
    image_size += [size]

    if abs(fit_arrival-sim_arrival)<0.05:
        delta_time_good += [delta_foci_time]
        image_size_good += [size]
    else:
        delta_time_bad += [delta_foci_time]
        image_size_bad += [size]

    fit_latent_space = []
    for r in range(0,rank):
        fit_latent_space += [lookup_table_pkl[r].get_bin_content(fit_impact,fit_log_energy,delta_foci_time)]
    fit_latent_space = np.array(fit_latent_space)

    fit_image = eigen_vectors_pkl.T @ fit_latent_space
    fit_image_2d = geom.image_to_cartesian_representation(fit_image)

    if abs(fit_arrival-sim_arrival)>0.05:
        #fig.clf()
        #axbig = fig.add_subplot()
        #label_x = 'X'
        #label_y = 'Y'
        #axbig.set_xlabel(label_x)
        #axbig.set_ylabel(label_y)
        #im = axbig.imshow(fit_image_2d,origin='lower')
        #cbar = fig.colorbar(im)
        #fig.savefig(f'{ctapipe_output}/output_plots/image_{img}_fit.png',bbox_inches='tight')
        #axbig.remove()

        fig.clf()
        axbig = fig.add_subplot()
        label_x = 'X'
        label_y = 'Y'
        axbig.set_xlabel(label_x)
        axbig.set_ylabel(label_y)
        im = axbig.imshow(sim_image_2d,origin='lower')
        cbar = fig.colorbar(im)
        fig.savefig(f'{ctapipe_output}/output_plots/evt_{img}_image_sim.png',bbox_inches='tight')
        axbig.remove()

        fig.clf()
        axbig = fig.add_subplot()
        label_x = 'X'
        label_y = 'Y'
        axbig.set_xlabel(label_x)
        axbig.set_ylabel(label_y)
        im = axbig.imshow(truth_image_2d,origin='lower')
        cbar = fig.colorbar(im)
        fig.savefig(f'{ctapipe_output}/output_plots/evt_{img}_image_truth.png',bbox_inches='tight')
        axbig.remove()

        #exit()

    fig.clf()
    axbig = fig.add_subplot()
    label_x = 'log energy truth'
    label_y = 'log energy predict'
    axbig.set_xlabel(label_x)
    axbig.set_ylabel(label_y)
    axbig.scatter(log_energy_truth, log_energy_hist_guess, s=90, c='r', marker='+')
    fig.savefig(f'{ctapipe_output}/output_plots/reconstruction_energy_guess.png',bbox_inches='tight')
    axbig.remove()

    fig.clf()
    axbig = fig.add_subplot()
    label_x = 'impact distance truth'
    label_y = 'impact distance predict'
    axbig.set_xlabel(label_x)
    axbig.set_ylabel(label_y)
    axbig.scatter(impact_truth, impact_hist_guess, s=90, c='r', marker='+')
    fig.savefig(f'{ctapipe_output}/output_plots/reconstruction_impact_hist_guess.png',bbox_inches='tight')
    axbig.remove()

    fig.clf()
    axbig = fig.add_subplot()
    label_x = 'arrival truth'
    label_y = 'arrival predict'
    axbig.set_xlabel(label_x)
    axbig.set_ylabel(label_y)
    axbig.scatter(arrival_truth, arrival_hist_guess, s=90, c='r', marker='+')
    fig.savefig(f'{ctapipe_output}/output_plots/reconstruction_arrival_hist_guess.png',bbox_inches='tight')
    axbig.remove()

    fig.clf()
    axbig = fig.add_subplot()
    label_x = 'impact distance truth'
    label_y = 'arrival error'
    axbig.set_xlabel(label_x)
    axbig.set_ylabel(label_y)
    axbig.scatter(impact_hist_guess, arrival_hist_error, s=90, c='r', marker='+')
    fig.savefig(f'{ctapipe_output}/output_plots/reconstruction_arrival_hist_error_vs_impact.png',bbox_inches='tight')
    axbig.remove()

    fig.clf()
    axbig = fig.add_subplot()
    label_x = 'log energy truth'
    label_y = 'arrival error'
    axbig.set_xlabel(label_x)
    axbig.set_ylabel(label_y)
    axbig.scatter(log_energy_hist_guess, arrival_hist_error, s=90, c='r', marker='+')
    fig.savefig(f'{ctapipe_output}/output_plots/reconstruction_arrival_hist_error_vs_energy.png',bbox_inches='tight')
    axbig.remove()

    fig.clf()
    axbig = fig.add_subplot()
    label_x = 'delta time'
    label_y = 'arrival error'
    axbig.set_xlabel(label_x)
    axbig.set_ylabel(label_y)
    axbig.scatter(delta_time, arrival_hist_error, s=90, c='r', marker='+')
    fig.savefig(f'{ctapipe_output}/output_plots/reconstruction_arrival_hist_error_vs_delta_time.png',bbox_inches='tight')
    axbig.remove()

    fig.clf()
    axbig = fig.add_subplot()
    label_x = 'image size'
    label_y = 'arrival error'
    axbig.set_xlabel(label_x)
    axbig.set_ylabel(label_y)
    axbig.scatter(image_size, arrival_hist_error, s=90, c='r', marker='+')
    axbig.set_xscale('log')
    fig.savefig(f'{ctapipe_output}/output_plots/reconstruction_arrival_hist_error_vs_size.png',bbox_inches='tight')
    axbig.remove()

    fig.clf()
    axbig = fig.add_subplot()
    label_x = 'image size'
    label_y = 'delta time'
    axbig.set_xlabel(label_x)
    axbig.set_ylabel(label_y)
    axbig.scatter(image_size_good, delta_time_good, s=90, c='g', marker='+')
    axbig.scatter(image_size_bad, delta_time_bad, s=90, c='r', marker='+')
    axbig.set_xscale('log')
    fig.savefig(f'{ctapipe_output}/output_plots/reconstruction_size_vs_time.png',bbox_inches='tight')
    axbig.remove()

    #exit()


