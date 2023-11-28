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
from load_cta_data import MyArray3D
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
    all_delta_time += [delta_foci_time]
    all_delta_foci_r += [foci_r2-foci_r1]
    
    if delta_foci_time>30.: continue
    if delta_foci_time<0.: continue
    training_id_list += [old_training_id_list[img]]
    training_telesc_position_matrix += [old_training_telesc_position_matrix[img]]
    training_truth_shower_position_matrix += [old_training_truth_shower_position_matrix[img]]
    train_cam_axes += [old_train_cam_axes[img]]
    big_training_image_matrix += [old_big_training_image_matrix[img]]
    big_training_time_matrix += [old_big_training_time_matrix[img]]
    big_training_param_matrix += [old_big_training_param_matrix[img]]
    big_training_moment_matrix += [old_big_training_moment_matrix[img]]

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

big_training_image_matrix = np.array(big_training_image_matrix)
big_training_time_matrix = np.array(big_training_time_matrix)
big_training_param_matrix = np.array(big_training_param_matrix)
big_training_moment_matrix = np.array(big_training_moment_matrix)


print ('Compute big matrix SVD...')
rank = 100
U_full, S_full, VT_full = np.linalg.svd(big_training_image_matrix,full_matrices=False)
U_eco = U_full[:, :rank]
VT_eco = VT_full[:rank, :]

output_filename = f'{ctapipe_output}/output_machines/eigen_vectors.pkl'
with open(output_filename,"wb") as file:
    pickle.dump(VT_eco, file)

n_bins = 20
n_zbins = 1
time_lower = -3.
time_upper = 1.5
impact_lower = 0.
impact_upper = 1.
log_energy_lower = -1.
log_energy_upper = 2.
lookup_table = []
lookup_table_arrival = MyArray3D(x_bins=n_bins,start_x=impact_lower,end_x=impact_upper,y_bins=n_bins,start_y=log_energy_lower,end_y=log_energy_upper,z_bins=n_zbins,start_z=time_lower,end_z=time_upper)
lookup_table_norm = MyArray3D(x_bins=n_bins,start_x=impact_lower,end_x=impact_upper,y_bins=n_bins,start_y=log_energy_lower,end_y=log_energy_upper,z_bins=n_zbins,start_z=time_lower,end_z=time_upper)
for r in range(0,rank):
    lookup_table += [MyArray3D(x_bins=n_bins,start_x=impact_lower,end_x=impact_upper,y_bins=n_bins,start_y=log_energy_lower,end_y=log_energy_upper,z_bins=n_zbins,start_z=time_lower,end_z=time_upper)]
for img in range(0,len(big_training_image_matrix)):
    delta_foci_time = np.log10(max(1e-3,big_training_moment_matrix[img][7]))
    semi_major_sq = big_training_moment_matrix[img][8]
    arrival = big_training_param_matrix[img][0]
    log_energy = big_training_param_matrix[img][1]
    impact = big_training_param_matrix[img][2]
    image = np.array(big_training_image_matrix[img])
    latent_space = VT_eco @ image
    for r in range(0,rank):
        lookup_table[r].fill(impact,log_energy,delta_foci_time,weight=latent_space[r])
    lookup_table_arrival.fill(impact,log_energy,delta_foci_time,weight=arrival)
    lookup_table_norm.fill(impact,log_energy,delta_foci_time,weight=1.)
for r in range(0,rank):
    lookup_table[r].divide(lookup_table_norm)
lookup_table_arrival.divide(lookup_table_norm)

lookup_table_arrival_rms = MyArray3D(x_bins=n_bins,start_x=impact_lower,end_x=impact_upper,y_bins=n_bins,start_y=log_energy_lower,end_y=log_energy_upper,z_bins=n_zbins,start_z=time_lower,end_z=time_upper)
for img in range(0,len(big_training_image_matrix)):
    delta_foci_time = np.log10(max(1e-3,big_training_moment_matrix[img][7]))
    semi_major_sq = big_training_moment_matrix[img][8]
    arrival = big_training_param_matrix[img][0]
    log_energy = big_training_param_matrix[img][1]
    impact = big_training_param_matrix[img][2]
    avg_arrival = lookup_table_arrival.get_bin_content(impact,log_energy,delta_foci_time)
    lookup_table_arrival_rms.fill(impact,log_energy,delta_foci_time,weight=pow(arrival-avg_arrival,2))
lookup_table_arrival_rms.divide(lookup_table_norm)
lookup_table_arrival_rms.waxis = np.sqrt(lookup_table_arrival_rms.waxis)

output_filename = f'{ctapipe_output}/output_machines/lookup_table.pkl'
with open(output_filename,"wb") as file:
    pickle.dump(lookup_table, file)
output_filename = f'{ctapipe_output}/output_machines/lookup_table_arrival.pkl'
with open(output_filename,"wb") as file:
    pickle.dump(lookup_table_arrival, file)
output_filename = f'{ctapipe_output}/output_machines/lookup_table_arrival_rms.pkl'
with open(output_filename,"wb") as file:
    pickle.dump(lookup_table_arrival_rms, file)

fig.clf()
axbig = fig.add_subplot()
label_x = 'Rank'
label_y = 'Signular value'
axbig.set_xlabel(label_x)
axbig.set_ylabel(label_y)
axbig.set_xlim(0,100)
axbig.plot(S_full)
fig.savefig(f'{ctapipe_output}/output_plots/training_sample_image_signularvalue.png',bbox_inches='tight')
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
    im = axbig.imshow(lookup_table[r].waxis[:,:,0].T,origin='lower',extent=(xmin,xmax,ymin,ymax))
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
im = axbig.imshow(lookup_table_arrival.waxis[:,:,0].T,origin='lower',extent=(xmin,xmax,ymin,ymax),aspect='auto')
cbar = fig.colorbar(im)
fig.savefig(f'{ctapipe_output}/output_plots/lookup_table_arrival.png',bbox_inches='tight')
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
im = axbig.imshow(lookup_table_arrival_rms.waxis[:,:,0].T,origin='lower',extent=(xmin,xmax,ymin,ymax),aspect='auto')
cbar = fig.colorbar(im)
fig.savefig(f'{ctapipe_output}/output_plots/lookup_table_arrival_rms.png',bbox_inches='tight')
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
#    latent_space = VT_eco @ image
#
#    nn_input += [latent_space+[delta_foci_time]]
#    nn_output += [[arrival,log_energy,impact]]
#
#learning_rate = 0.1
#n_node = 20
#nn_lookup_table = NeuralNetwork(nn_input[0],nn_output[0],learning_rate,n_node)
#training_error = nn_cleaner.train(nn_input, nn_output, 10000)
#
#fig.clf()
#axbig = fig.add_subplot()
#label_x = 'Iterations'
#label_y = 'Error'
#axbig.set_xlabel(label_x)
#axbig.set_ylabel(label_y)
#axbig.plot(training_error)
#fig.savefig(f'{ctapipe_output}/output_plots/training_error.png',bbox_inches='tight')
#axbig.remove()
#
#exit()
#
