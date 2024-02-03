
import os
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

import common_functions
MyArray3D = common_functions.MyArray3D

ctapipe_output = os.environ.get("CTAPIPE_OUTPUT_PATH")
ctapipe_input = os.environ.get("CTAPIPE_SVC_PATH")
print (f'ctapipe_output = {ctapipe_output}')

training_sample_path = []
#training_sample_path += [get_dataset_path("gamma_40deg_0deg_run2006___cta-prod3-sct_desert-2150m-Paranal-SCT.simtel.gz")]
with open('%s/sim_files.txt'%(ctapipe_input), 'r') as file:
    for line in file:
        training_sample_path += [get_dataset_path(line.strip('\n'))]

# start memory profiling
tracemalloc.start()

big_truth_matrix = []
big_moment_matrix = []
big_movie_matrix = []
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

    big_truth_matrix += training_sample[0]
    big_moment_matrix += training_sample[1]
    big_movie_matrix += training_sample[2]

    print(f'memory usage (current,peak) = {tracemalloc.get_traced_memory()}')

big_movie_matrix = np.array(big_movie_matrix)

fig, ax = plt.subplots()
figsize_x = 8.6
figsize_y = 6.4
fig.set_figheight(figsize_y)
fig.set_figwidth(figsize_x)

n_bins_arrival = 40
arrival_lower = 0.
arrival_upper = 0.4
n_bins_impact = 40
impact_lower = 0.
impact_upper = 0.4
n_bins_xmax = 10
xmax_lower = 150.
xmax_upper = 375.
n_bins_energy = 10
log_energy_lower = -1.
log_energy_upper = 2.

print ('Compute big matrix SVD...')
n_images, n_pixels = big_movie_matrix.shape
print (f'n_images = {n_images}, n_pixels = {n_pixels}')
image_rank = 30
U_full, S_full, VT_full = np.linalg.svd(big_movie_matrix,full_matrices=False)
U_eco = U_full[:, :image_rank]
VT_eco = VT_full[:image_rank, :]

print(f'memory usage (current,peak) = {tracemalloc.get_traced_memory()}')

print (f'saving eigenvector to {ctapipe_output}/output_machines...')
output_filename = f'{ctapipe_output}/output_machines/eigen_vectors.pkl'
with open(output_filename,"wb") as file:
    pickle.dump(VT_eco, file)

fig.clf()
axbig = fig.add_subplot()
label_x = 'Rank'
label_y = 'Signular value'
axbig.set_xlabel(label_x)
axbig.set_ylabel(label_y)
axbig.set_xlim(0,100)
axbig.plot(S_full)
fig.savefig(f'{ctapipe_output}/output_plots/training_movie_signularvalue.png',bbox_inches='tight')
axbig.remove()

lookup_table = []
lookup_table_norm = MyArray3D(x_bins=n_bins_arrival,start_x=arrival_lower,end_x=arrival_upper,y_bins=n_bins_xmax,start_y=xmax_lower,end_y=xmax_upper,z_bins=n_bins_energy,start_z=log_energy_lower,end_z=log_energy_upper)

list_impact = []
list_arrival = []
list_log_energy = []
list_height = []
list_xmax = []
list_image_qual = []

for r in range(0,image_rank):
    lookup_table += [MyArray3D(x_bins=n_bins_arrival,start_x=arrival_lower,end_x=arrival_upper,y_bins=n_bins_xmax,start_y=xmax_lower,end_y=xmax_upper,z_bins=n_bins_energy,start_z=log_energy_lower,end_z=log_energy_upper)]


for img in range(0,len(big_movie_matrix)):

    image_center_x = big_moment_matrix[img][1]
    image_center_y = big_moment_matrix[img][2]
    time_direction = big_moment_matrix[img][6]
    image_direction = big_moment_matrix[img][7]
    image_qual = abs(image_direction+time_direction)

    truth_energy = float(big_truth_matrix[img][0]/u.TeV)
    truth_height = float(big_truth_matrix[img][5]/u.m)
    truth_x_max = float(big_truth_matrix[img][6]/(u.g/(u.cm*u.cm)))
    star_cam_x = big_truth_matrix[img][7]
    star_cam_y = big_truth_matrix[img][8]
    impact_x = big_truth_matrix[img][9]
    impact_y = big_truth_matrix[img][10]

    arrival = pow(pow(star_cam_x-image_center_x,2)+pow(star_cam_y-image_center_y,2),0.5)
    impact = pow(impact_x*impact_x+impact_y*impact_y,0.5)
    log_energy = np.log10(truth_energy)

    list_log_energy += [log_energy]
    list_height += [truth_height]
    list_xmax += [truth_x_max]
    list_arrival += [arrival]
    list_impact += [impact]

    movie = np.array(big_movie_matrix[img])
    latent_space = VT_eco @ movie
    for r in range(0,image_rank):
        lookup_table[r].fill(arrival,truth_x_max,log_energy,weight=latent_space[r])

    lookup_table_norm.fill(arrival,truth_x_max,log_energy,weight=1.)

for r in range(0,image_rank):
    lookup_table[r].divide(lookup_table_norm)

print (f'saving table to {ctapipe_output}/output_machines...')
output_filename = f'{ctapipe_output}/output_machines/lookup_table.pkl'
with open(output_filename,"wb") as file:
    pickle.dump(lookup_table, file)


print (f'saving plots to {ctapipe_output}/output_plots...')
fig.clf()
axbig = fig.add_subplot()
label_x = 'log10 Energy [TeV]'
label_y = 'Height [m]'
axbig.set_xlabel(label_x)
axbig.set_ylabel(label_y)
axbig.scatter(list_log_energy, list_height, s=90, c='b', marker='+', alpha=0.1)
fig.savefig(f'{ctapipe_output}/output_plots/scatter_log_energy_vs_height.png',bbox_inches='tight')
axbig.remove()

fig.clf()
axbig = fig.add_subplot()
label_x = 'log10 Energy [TeV]'
label_y = 'X max [g/cm2]'
axbig.set_xlabel(label_x)
axbig.set_ylabel(label_y)
axbig.scatter(list_log_energy, list_xmax, s=90, c='b', marker='+', alpha=0.1)
fig.savefig(f'{ctapipe_output}/output_plots/scatter_log_energy_vs_xmax.png',bbox_inches='tight')
axbig.remove()

fig.clf()
axbig = fig.add_subplot()
label_x = 'Arrival [m]'
label_y = 'Impact [m]'
axbig.set_xlabel(label_x)
axbig.set_ylabel(label_y)
axbig.scatter(list_arrival, list_impact, s=90, c='b', marker='+', alpha=0.1)
fig.savefig(f'{ctapipe_output}/output_plots/scatter_arrival_vs_impact.png',bbox_inches='tight')
axbig.remove()

fig.clf()
axbig = fig.add_subplot()
label_x = 'Arrival [m]'
label_y = 'X max [g/cm2]'
axbig.set_xlabel(label_x)
axbig.set_ylabel(label_y)
axbig.scatter(list_arrival, list_xmax, s=90, c='b', marker='+', alpha=0.1)
fig.savefig(f'{ctapipe_output}/output_plots/scatter_arrival_vs_xmax.png',bbox_inches='tight')
axbig.remove()

tracemalloc.stop()
