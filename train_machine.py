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

training_id_list = []
training_telesc_position_matrix = []
training_truth_shower_position_matrix = []
train_cam_axes = []
big_training_image_matrix = []
big_training_param_matrix = []
for path in range(0,len(training_sample_path)):
    source = SimTelEventSource(training_sample_path[path], focal_length_choice='EQUIVALENT')
    subarray = source.subarray
    ob_keys = source.observation_blocks.keys()
    run_id = list(ob_keys)[0]
    output_filename = f'{ctapipe_output}/output_samples/training_sample_truth_dirty_repose_run{run_id}.pkl'
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

print ('Compute big matrix SVD...')
big_training_image_matrix = np.array(big_training_image_matrix)
big_training_param_matrix = np.array(big_training_param_matrix)


# neural network for image cleaning
learning_rate = 0.1
n_node = 100
nn_cleaner = NeuralNetwork(big_training_image_matrix[0],big_training_truth_image_matrix[0],learning_rate,n_node)
training_error = nn_cleaner.train(big_training_image_matrix, big_training_truth_image_matrix, 10000)

fig.clf()
axbig = fig.add_subplot()
label_x = 'Iterations'
label_y = 'Error'
axbig.set_xlabel(label_x)
axbig.set_ylabel(label_y)
axbig.plot(training_error)
fig.savefig(f'{ctapipe_output}/output_plots/training_error_cleaning.png',bbox_inches='tight')
axbig.remove()



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

fig.clf()
axbig = fig.add_subplot()
label_x = 'Rank'
label_y = 'Signular value'
axbig.set_xlabel(label_x)
axbig.set_ylabel(label_y)
axbig.set_xlim(0,100)
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
im = axbig.imshow(lookup_table_arrival.zaxis[:,:].T,origin='lower',extent=(xmin,xmax,ymin,ymax),aspect='auto')
cbar = fig.colorbar(im)
fig.savefig(f'{ctapipe_output}/output_plots/lookup_table_arrival.png',bbox_inches='tight')
axbig.remove()

exit()

