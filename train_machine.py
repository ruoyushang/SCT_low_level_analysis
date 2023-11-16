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

training_sample_path = []
#training_sample_path += [get_dataset_path("gamma_20deg_0deg_run853___cta-prod3-sct_desert-2150m-Paranal-SCT.simtel.gz")]
with open('train_sim_files.txt', 'r') as file:
    for line in file:
        training_sample_path += [get_dataset_path(line.strip('\n'))]

training_id_list = []
training_telesc_position_matrix = []
training_truth_shower_position_matrix = []
train_cam_axes = []
big_training_image_matrix = []
big_training_param_matrix = []
big_training_hillas_matrix = []
for path in range(0,len(training_sample_path)):
    source = SimTelEventSource(training_sample_path[path], focal_length_choice='EQUIVALENT')
    subarray = source.subarray
    ob_keys = source.observation_blocks.keys()
    run_id = list(ob_keys)[0]
    output_filename = f'{ctapipe_output}/output_samples/training_sample_run{run_id}.pkl'
    print (f'loading pickle trainging sample data: {output_filename}')
    if not os.path.exists(output_filename):
        print (f'file does not exist.')
        continue
    training_sample = pickle.load(open(output_filename, "rb"))

    training_id_list += training_sample[0]
    training_telesc_position_matrix += training_sample[1]
    training_truth_shower_position_matrix = training_sample[2]
    train_cam_axes += training_sample[3]
    big_training_image_matrix += training_sample[4]
    big_training_param_matrix += training_sample[5]
    big_training_hillas_matrix += training_sample[6]

print ('Compute big matrix SVD...')
big_training_image_matrix = np.array(big_training_image_matrix)
big_training_param_matrix = np.array(big_training_param_matrix)
big_training_hillas_matrix = np.array(big_training_hillas_matrix)

rank = 20

# Calculate the unweighted pseudo-inverse
U_full, S_full, VT_full = np.linalg.svd(big_training_image_matrix,full_matrices=False)
U_eco = U_full[:, :rank]
VT_eco = VT_full[:rank, :]
S_pseudo = np.diag(1 / S_full[:rank])
inv_M_pseudo = VT_eco.T @ S_pseudo @ U_eco.T
# Compute the weighted least-squares solution
svd_image2param = inv_M_pseudo @ big_training_param_matrix

output_filename = f'{ctapipe_output}/output_machines/eigen_vectors.pkl'
with open(output_filename,"wb") as file:
    pickle.dump(VT_eco, file)

n_bins = 10
lookup_table = []
lookup_table_arrival = MyArray2D(x_bins=n_bins,start_x=0.,end_x=0.5,y_bins=n_bins,start_y=-1.,end_y=1.)
lookup_table_norm = MyArray2D(x_bins=n_bins,start_x=0.,end_x=0.5,y_bins=n_bins,start_y=-1.,end_y=1.)
for r in range(0,rank):
    lookup_table += [MyArray2D(x_bins=n_bins,start_x=0.,end_x=0.5,y_bins=n_bins,start_y=-1.,end_y=1.)]
for img in range(0,len(big_training_image_matrix)):
    arrival = big_training_param_matrix[img][0]
    log_energy = big_training_param_matrix[img][1]
    impact = big_training_param_matrix[img][2]
    image = np.array(big_training_image_matrix[img])
    latent_space = VT_eco @ image
    for r in range(0,rank):
        lookup_table[r].fill(impact,log_energy,weight=latent_space[r])
    lookup_table_arrival.fill(impact,log_energy,weight=arrival)
    lookup_table_norm.fill(impact,log_energy,weight=1.)
for r in range(0,rank):
    lookup_table[r].divide(lookup_table_norm)
lookup_table_arrival.divide(lookup_table_norm)

output_filename = f'{ctapipe_output}/output_machines/lookup_table.pkl'
with open(output_filename,"wb") as file:
    pickle.dump(lookup_table, file)
output_filename = f'{ctapipe_output}/output_machines/lookup_table_arrival.pkl'
with open(output_filename,"wb") as file:
    pickle.dump(lookup_table_arrival, file)

fig, ax = plt.subplots()
figsize_x = 8.6
figsize_y = 6.4
fig.set_figheight(figsize_y)
fig.set_figwidth(figsize_x)

fig.clf()
axbig = fig.add_subplot()
label_x = 'Rank'
label_y = 'Signular value'
axbig.set_xlabel(label_x)
axbig.set_ylabel(label_y)
axbig.set_yscale('log')
axbig.plot(S_full)
fig.savefig(f'{ctapipe_output}/output_plots/training_sample_signularvalue.png',bbox_inches='tight')
axbig.remove()

for r in range(0,rank):
    fig.clf()
    axbig = fig.add_subplot()
    label_x = 'Impact [km]'
    label_y = 'log Energy [TeV]'
    axbig.set_xlabel(label_x)
    axbig.set_ylabel(label_y)
    xmin = lookup_table[r].xaxis.min()
    xmax = lookup_table[r].xaxis.max()
    ymin = lookup_table[r].yaxis.min()
    ymax = lookup_table[r].yaxis.max()
    im = axbig.imshow(lookup_table[r].zaxis[:,:].T,origin='lower',extent=(xmin,xmax,ymin,ymax))
    cbar = fig.colorbar(im)
fig.savefig(f'{ctapipe_output}/output_plots/lookup_table_rank{r}.png',bbox_inches='tight')
axbig.remove()
fig.clf()
axbig = fig.add_subplot()
label_x = 'Impact [km]'
label_y = 'log Energy [TeV]'
axbig.set_xlabel(label_x)
axbig.set_ylabel(label_y)
xmin = lookup_table_arrival.xaxis.min()
xmax = lookup_table_arrival.xaxis.max()
ymin = lookup_table_arrival.yaxis.min()
ymax = lookup_table_arrival.yaxis.max()
im = axbig.imshow(lookup_table_arrival.zaxis[:,:].T,origin='lower',extent=(xmin,xmax,ymin,ymax))
cbar = fig.colorbar(im)
fig.savefig(f'{ctapipe_output}/output_plots/lookup_table_arrival.png',bbox_inches='tight')
axbig.remove()

log_energy_truth = []
impact_truth = []
arrival_truth = []
log_energy_guess = []
impact_guess = []
arrival_guess = []
for img in range(0,len(training_id_list)):

    current_run = training_id_list[img][0]
    current_event = training_id_list[img][1]
    current_tel_id = training_id_list[img][2]
    subarray = training_id_list[img][3]
    geom = subarray.tel[current_tel_id].camera.geometry

    truth_image = big_training_image_matrix[img]
    truth_image_2d = geom.image_to_cartesian_representation(truth_image)

    #truth_latent_space = VT_eco @ truth_image
    #reco_param = latent2param.predict(truth_latent_space)[:,0]

    sim_arrival = big_training_param_matrix[img][0]
    sim_log_energy = big_training_param_matrix[img][1]
    sim_impact = big_training_param_matrix[img][2]
    print ('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print (f'img = {img}')
    print (f'sim_log_energy = {sim_log_energy}')
    print (f'sim_impact = {sim_impact}')
    #if img!=2: continue

    latent_space = []
    for r in range(0,rank):
        latent_space += [lookup_table[r].get_bin_content(sim_impact,sim_log_energy)]
    latent_space = np.array(latent_space)

    sim_image = VT_eco.T @ latent_space
    sim_image_2d = geom.image_to_cartesian_representation(sim_image)

    fit_log_energy = 0.
    fit_impact = 0.1
    init_params = [fit_log_energy,fit_impact]
    fit_chi2 = sqaure_difference_between_1d_images(init_params,train_cam_axes[img],geom,truth_image,lookup_table,VT_eco)
    print (f'init_log_energy = {fit_log_energy}')
    print (f'init_impact = {fit_impact}')
    print (f'init_chi2 = {fit_chi2}')
    n_bins_energy = len(lookup_table[0].yaxis)
    n_bins_impact = len(lookup_table[0].xaxis)
    for idx_x  in range(0,n_bins_impact):
        for idx_y  in range(0,n_bins_energy):
            try_log_energy = lookup_table[0].yaxis[idx_y]
            try_impact = lookup_table[0].xaxis[idx_x]
            init_params = [try_log_energy,try_impact]
            try_chi2 = sqaure_difference_between_1d_images(init_params,train_cam_axes[img],geom,truth_image,lookup_table,VT_eco)
            if try_chi2<fit_chi2:
                fit_chi2 = try_chi2
                fit_log_energy = try_log_energy
                fit_impact = try_impact
    print (f'fit_log_energy = {fit_log_energy}')
    print (f'fit_impact = {fit_impact}')
    print (f'fit_chi2 = {fit_chi2}')

    fit_arrival = lookup_table_arrival.get_bin_content(fit_impact,fit_log_energy)

    log_energy_truth += [sim_log_energy]
    impact_truth += [sim_impact]
    arrival_truth += [sim_arrival]
    log_energy_guess += [fit_log_energy]
    impact_guess += [fit_impact]
    arrival_guess += [fit_arrival]

    fit_latent_space = []
    for r in range(0,rank):
        fit_latent_space += [lookup_table[r].get_bin_content(fit_impact,fit_log_energy)]
    fit_latent_space = np.array(fit_latent_space)

    fit_image = VT_eco.T @ fit_latent_space
    fit_image_2d = geom.image_to_cartesian_representation(fit_image)

    fig.clf()
    axbig = fig.add_subplot()
    label_x = 'X'
    label_y = 'Y'
    axbig.set_xlabel(label_x)
    axbig.set_ylabel(label_y)
    im = axbig.imshow(sim_image_2d,origin='lower')
    cbar = fig.colorbar(im)
    fig.savefig(f'{ctapipe_output}/output_plots/image_{img}_sim.png',bbox_inches='tight')
    axbig.remove()

    fig.clf()
    axbig = fig.add_subplot()
    label_x = 'X'
    label_y = 'Y'
    axbig.set_xlabel(label_x)
    axbig.set_ylabel(label_y)
    im = axbig.imshow(fit_image_2d,origin='lower')
    cbar = fig.colorbar(im)
    fig.savefig(f'{ctapipe_output}/output_plots/image_{img}_fit.png',bbox_inches='tight')
    axbig.remove()

    fig.clf()
    axbig = fig.add_subplot()
    label_x = 'X'
    label_y = 'Y'
    axbig.set_xlabel(label_x)
    axbig.set_ylabel(label_y)
    im = axbig.imshow(truth_image_2d,origin='lower')
    cbar = fig.colorbar(im)
    fig.savefig(f'{ctapipe_output}/output_plots/image_{img}_truth.png',bbox_inches='tight')
    axbig.remove()

    fig.clf()
    axbig = fig.add_subplot()
    label_x = 'log energy truth'
    label_y = 'log energy predict'
    axbig.set_xlabel(label_x)
    axbig.set_ylabel(label_y)
    axbig.scatter(log_energy_truth, log_energy_guess, s=90, c='r', marker='+')
    fig.savefig(f'{ctapipe_output}/output_plots/reconstruction_energy_error.png',bbox_inches='tight')
    axbig.remove()

    fig.clf()
    axbig = fig.add_subplot()
    label_x = 'impact distance truth'
    label_y = 'impact distance predict'
    axbig.set_xlabel(label_x)
    axbig.set_ylabel(label_y)
    axbig.scatter(impact_truth, impact_guess, s=90, c='r', marker='+')
    fig.savefig(f'{ctapipe_output}/output_plots/reconstruction_impact_guess.png',bbox_inches='tight')
    axbig.remove()

    fig.clf()
    axbig = fig.add_subplot()
    label_x = 'arrival truth'
    label_y = 'arrival predict'
    axbig.set_xlabel(label_x)
    axbig.set_ylabel(label_y)
    axbig.scatter(arrival_truth, arrival_guess, s=90, c='r', marker='+')
    fig.savefig(f'{ctapipe_output}/output_plots/reconstruction_arrival_guess.png',bbox_inches='tight')
    axbig.remove()

    #exit()

#current_run = training_id_list[0][0]
#current_event = training_id_list[0][1]
#current_tel_id = training_id_list[0][2]
#subarray = training_id_list[0][3]
#geom = subarray.tel[current_tel_id].camera.geometry
#
#sim_energy = 0.4
#sim_impact = 0.1
#sim_log_energy = math.log10(sim_energy)
#
#latent_space = []
#for r in range(0,rank):
#    latent_space += [lookup_table[r].get_bin_content(sim_impact,sim_log_energy)]
#latent_space = np.array(latent_space)
#
#sim_image = VT_eco.T @ latent_space
#sim_image_2d = geom.image_to_cartesian_representation(sim_image)
#
#fig.clf()
#axbig = fig.add_subplot()
#label_x = 'X'
#label_y = 'Y'
#axbig.set_xlabel(label_x)
#axbig.set_ylabel(label_y)
#im = axbig.imshow(sim_image_2d,origin='lower')
#cbar = fig.colorbar(im)
#fig.savefig(f'{ctapipe_output}/output_plots/image_{img}_energy{sim_energy}_impact{sim_impact}.png',bbox_inches='tight')
#axbig.remove()
#
#sim_energy = 0.8
#sim_impact = 0.2
#sim_log_energy = math.log10(sim_energy)
#
#latent_space = []
#for r in range(0,rank):
#    latent_space += [lookup_table[r].get_bin_content(sim_impact,sim_log_energy)]
#latent_space = np.array(latent_space)
#
#sim_image = VT_eco.T @ latent_space
#sim_image_2d = geom.image_to_cartesian_representation(sim_image)
#
#fig.clf()
#axbig = fig.add_subplot()
#label_x = 'X'
#label_y = 'Y'
#axbig.set_xlabel(label_x)
#axbig.set_ylabel(label_y)
#im = axbig.imshow(sim_image_2d,origin='lower')
#cbar = fig.colorbar(im)
#fig.savefig(f'{ctapipe_output}/output_plots/image_{img}_energy{sim_energy}_impact{sim_impact}.png',bbox_inches='tight')
#axbig.remove()
#
