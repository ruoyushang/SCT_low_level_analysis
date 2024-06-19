
import os, sys
import subprocess
import glob
import tracemalloc

import pickle
from matplotlib import pyplot as plt
from matplotlib import colors
import numpy as np
from astropy import units as u

from ctapipe.utils.datasets import get_dataset_path
from ctapipe.io import EventSource, SimTelEventSource, HDF5TableWriter

import torch
import torch.nn.functional as torchF

import common_functions
MyArray3D = common_functions.MyArray3D
linear_regression = common_functions.linear_regression
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


ctapipe_output = os.environ.get("CTAPIPE_OUTPUT_PATH")
ctapipe_input = os.environ.get("CTAPIPE_SVC_PATH")
print (f'ctapipe_output = {ctapipe_output}')

sim_files = 'sim_files.txt'
#sim_files = 'sim_files_diffuse_gamma.txt'
#sim_files = 'sim_files_merged_point_20deg.txt'

make_movie = True

telescope_type = sys.argv[1]

training_sample_path = []
n_samples = 0
with open(f'{ctapipe_input}/{sim_files}', 'r') as file:
    for line in file:
        #if n_samples>7: continue
        training_sample_path += [get_dataset_path(line.strip('\n'))]
        n_samples += 1

# start memory profiling
tracemalloc.start()

big_truth_matrix = []
big_moment_matrix = []
big_image_matrix = []
big_time_matrix = []
big_movie_matrix = []
for path in range(0,len(training_sample_path)):

    #if path>100: break

    source = SimTelEventSource(training_sample_path[path], focal_length_choice='EQUIVALENT')
    subarray = source.subarray
    ob_keys = source.observation_blocks.keys()
    run_id = list(ob_keys)[0]
    output_filename = f'{ctapipe_output}/output_samples/training_sample_run{run_id}_{telescope_type}.pkl'
    print (f'loading pickle trainging sample data: {output_filename}')
    if not os.path.exists(output_filename):
        print (f'file does not exist.')
        continue
    training_sample = pickle.load(open(output_filename, "rb"))

    big_truth_matrix += training_sample[0]
    big_moment_matrix += training_sample[1]
    big_image_matrix += training_sample[2]
    big_time_matrix += training_sample[3]
    if make_movie:
        big_movie_matrix += training_sample[4]

    print(f'memory usage (current,peak) = {tracemalloc.get_traced_memory()}')

print (f'saving big matrices to {ctapipe_output}/output_machines...')

output_filename = f'{ctapipe_output}/output_machines/big_truth_matrix_{telescope_type}.pkl'
with open(output_filename,"wb") as file:
    pickle.dump(big_truth_matrix, file)

output_filename = f'{ctapipe_output}/output_machines/big_moment_matrix_{telescope_type}.pkl'
with open(output_filename,"wb") as file:
    pickle.dump(big_moment_matrix, file)

output_filename = f'{ctapipe_output}/output_machines/big_image_matrix_{telescope_type}.pkl'
with open(output_filename,"wb") as file:
    pickle.dump(big_image_matrix, file)

output_filename = f'{ctapipe_output}/output_machines/big_time_matrix_{telescope_type}.pkl'
with open(output_filename,"wb") as file:
    pickle.dump(big_time_matrix, file)

if make_movie:
    output_filename = f'{ctapipe_output}/output_machines/big_movie_matrix_{telescope_type}.pkl'
    with open(output_filename,"wb") as file:
        pickle.dump(big_movie_matrix, file)


tracemalloc.stop()
