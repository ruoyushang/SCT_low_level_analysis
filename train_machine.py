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

ctapipe_output = os.environ.get("CTAPIPE_OUTPUT_PATH")

print ('loading pickle trainging sample data... ')
output_filename = f'{ctapipe_output}/output_machines/training_sample.pkl'
training_sample = pickle.load(open(output_filename, "rb"))

training_id_list = training_sample[0]
training_telesc_position_matrix = training_sample[1]
training_truth_shower_position_matrix = training_sample[2]
train_cam_axes = training_sample[3]
big_training_image_matrix = training_sample[4]
big_training_param_matrix = training_sample[5]

print ('Compute big matrix SVD...')
big_training_image_matrix = np.array(big_training_image_matrix)
big_training_param_matrix = np.array(big_training_param_matrix)

# Calculate the unweighted pseudo-inverse
U_full, S_full, VT_full = np.linalg.svd(big_training_image_matrix,full_matrices=False)
rank = 100
U_eco = U_full[:, :rank]
VT_eco = VT_full[:rank, :]
S_pseudo = np.diag(1 / S_full[:rank])
inv_M_pseudo = VT_eco.T @ S_pseudo @ U_eco.T
# Compute the weighted least-squares solution
svd_image2param = inv_M_pseudo @ big_training_param_matrix

u_full, s_full, vT_full = np.linalg.svd(big_training_param_matrix,full_matrices=False)
#u_eco = u_full[:, :rank]
#vT_eco = vT_full[:rank, :]
#s_pseudo = np.diag(1 / s_full[:rank])
u_eco = u_full
vT_eco = vT_full
s_pseudo = np.diag(1 / s_full)
inv_m_pseudo = vT_eco.T @ s_pseudo @ u_eco.T
svd_param2image = inv_m_pseudo @ big_training_image_matrix

print ('Train a Neural Network...')
tic = time.perf_counter()
learning_rate = 0.1
nn_param2image = NeuralNetwork(big_training_param_matrix[0],big_training_image_matrix[0],learning_rate,20)
training_error_param2image = nn_param2image.train(big_training_param_matrix, big_training_image_matrix, 10000)
nn_image2param = NeuralNetwork(big_training_image_matrix[0],big_training_param_matrix[0],learning_rate,20)
training_error_image2param = nn_image2param.train(big_training_image_matrix, big_training_param_matrix, 10000)
toc = time.perf_counter()
print (f'Neural Network training completed in {toc - tic:0.4f} seconds')



fig, ax = plt.subplots()
figsize_x = 8.6
figsize_y = 6.4
fig.set_figheight(figsize_y)
fig.set_figwidth(figsize_x)

fig.clf()
axbig = fig.add_subplot()
label_x = 'Iterations'
label_y = 'Error'
axbig.set_xlabel(label_x)
axbig.set_ylabel(label_y)
axbig.plot(training_error_param2image)
fig.savefig(f'{ctapipe_output}/output_plots/training_error_param2image.png',bbox_inches='tight')
axbig.remove()

fig.clf()
axbig = fig.add_subplot()
label_x = 'Iterations'
label_y = 'Error'
axbig.set_xlabel(label_x)
axbig.set_ylabel(label_y)
axbig.plot(training_error_image2param)
fig.savefig(f'{ctapipe_output}/output_plots/training_error_image2param.png',bbox_inches='tight')
axbig.remove()

output_filename = f'{ctapipe_output}/output_machines/svd_param2image.pkl'
with open(output_filename,"wb") as file:
    pickle.dump(svd_param2image, file)

output_filename = f'{ctapipe_output}/output_machines/svd_image2param.pkl'
with open(output_filename,"wb") as file:
    pickle.dump(svd_image2param, file)

output_filename = f'{ctapipe_output}/output_machines/nn_param2image.pkl'
with open(output_filename,"wb") as file:
    pickle.dump(nn_param2image, file)

output_filename = f'{ctapipe_output}/output_machines/nn_image2param.pkl'
with open(output_filename,"wb") as file:
    pickle.dump(nn_image2param, file)

