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

ctapipe_output = os.environ.get("CTAPIPE_OUTPUT_PATH")

training_sample_path = []
#training_sample_path = [get_dataset_path('gamma_20deg_0deg_run802___cta-prod3-sct_desert-2150m-Paranal-SCT.simtel.gz')]
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
    print ('loading pickle trainging sample data... ')
    output_filename = f'{ctapipe_output}/output_samples/training_sample_run{run_id}.pkl'
    if not os.path.exists(output_filename):
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

n_bins = 20
lookup_table = []
lookup_table_norm = []
for r in range(0,rank):
    lookup_table += [MyArray2D(x_bins=n_bins,start_x=0.,end_x=0.5,y_bins=n_bins,start_y=-1.,end_y=1.)]
    lookup_table_norm += [MyArray2D(x_bins=n_bins,start_x=0.,end_x=0.5,y_bins=n_bins,start_y=-1.,end_y=1.)]

for img in range(0,len(big_training_image_matrix)):
    energy = big_training_param_matrix[img][1]
    impact = big_training_param_matrix[img][2]
    log_energy = math.log10(energy)
    image = np.array(big_training_image_matrix[img])
    #print (f'image.shape = {image.shape}')
    #print (f'VT_eco.shape = {VT_eco.shape}')
    latent_space = VT_eco @ image
    #print (f'latent_space = {latent_space}')
    for r in range(0,rank):
        lookup_table[r].fill(impact,log_energy,weight=latent_space[r])
        lookup_table_norm[r].fill(impact,log_energy,weight=1.)

for r in range(0,rank):
    lookup_table[r].divide(lookup_table_norm[r])

output_filename = f'{ctapipe_output}/output_machines/lookup_table.pkl'
with open(output_filename,"wb") as file:
    pickle.dump(lookup_table, file)

output_filename = f'{ctapipe_output}/output_machines/eigen_vectors.pkl'
with open(output_filename,"wb") as file:
    pickle.dump(VT_eco, file)


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

#for img in range(0,len(training_id_list)):
#
#    current_run = training_id_list[img][0]
#    current_event = training_id_list[img][1]
#    current_tel_id = training_id_list[img][2]
#    subarray = training_id_list[img][3]
#    geom = subarray.tel[current_tel_id].camera.geometry
#
#    sim_energy = big_training_param_matrix[img][1]
#    sim_impact = big_training_param_matrix[img][2]
#    sim_log_energy = math.log10(sim_energy)
#
#    latent_space = []
#    for r in range(0,rank):
#        latent_space += [lookup_table[r].get_bin_content(sim_impact,sim_log_energy)]
#    latent_space = np.array(latent_space)
#    
#    sim_image = VT_eco.T @ latent_space
#    sim_image_2d = geom.image_to_cartesian_representation(sim_image)
#
#    truth_image = big_training_image_matrix[img]
#    truth_image_2d = geom.image_to_cartesian_representation(truth_image)
#
#    fig.clf()
#    axbig = fig.add_subplot()
#    label_x = 'X'
#    label_y = 'Y'
#    axbig.set_xlabel(label_x)
#    axbig.set_ylabel(label_y)
#    im = axbig.imshow(sim_image_2d,origin='lower')
#    cbar = fig.colorbar(im)
#    fig.savefig(f'{ctapipe_output}/output_plots/image_{img}_sim.png',bbox_inches='tight')
#    axbig.remove()
#
#    fig.clf()
#    axbig = fig.add_subplot()
#    label_x = 'X'
#    label_y = 'Y'
#    axbig.set_xlabel(label_x)
#    axbig.set_ylabel(label_y)
#    im = axbig.imshow(truth_image_2d,origin='lower')
#    cbar = fig.colorbar(im)
#    fig.savefig(f'{ctapipe_output}/output_plots/image_{img}_truth.png',bbox_inches='tight')
#    axbig.remove()
#
#    exit()

