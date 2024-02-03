
import os, sys
import subprocess
import glob

import numpy as np
from astropy import units as u
from matplotlib import pyplot as plt
from matplotlib import colors
import pickle

from ctapipe.utils.datasets import get_dataset_path
from ctapipe.io import EventSource, SimTelEventSource, HDF5TableWriter

ctapipe_output = os.environ.get("CTAPIPE_OUTPUT_PATH")
ctapipe_input = os.environ.get("CTAPIPE_SVC_PATH")

fig, ax = plt.subplots()
figsize_x = 8.6
figsize_y = 6.4
fig.set_figheight(figsize_y)
fig.set_figwidth(figsize_x)

font = {'family': 'serif', 'color':  'black', 'weight': 'normal', 'size': 10, 'rotation': 0.,}

training_sample_path = []
particle_type = []
with open('%s/sim_files.txt'%(ctapipe_input), 'r') as file:
    for line in file:
        training_sample_path += [get_dataset_path(line.strip('\n'))]
        particle_type += [0]

def plot_monotel_analysis():

    list_log10_image_size = []
    list_log10_image_qual = []
    list_delta_energy = []
    list_delta_arrival = []

    for path in range(0,len(training_sample_path)):
    
        source = SimTelEventSource(training_sample_path[path], focal_length_choice='EQUIVALENT')
        subarray = source.subarray
        ob_keys = source.observation_blocks.keys()
        run_id = list(ob_keys)[0]
    
        ana_tag = 'monotel_ana'
        input_filename = f'{ctapipe_output}/output_analysis/{ana_tag}_run{run_id}.pkl'
        print (f'loading pickle analysis data: {input_filename}')
        if not os.path.exists(input_filename):
            print (f'file does not exist.')
            continue
        analysis_result = pickle.load(open(input_filename, "rb"))
    
        for img in range(0,len(analysis_result)):

            image_size = analysis_result[img][0]
            image_qual = analysis_result[img][1]
            truth_energy = analysis_result[img][2]
            fit_energy = analysis_result[img][3]
            truth_alt = analysis_result[img][4]
            truth_az = analysis_result[img][5]
            fit_alt = analysis_result[img][6]
            fit_az = analysis_result[img][7]

            if image_size<200.: continue
            if image_qual<2.: continue

            delta_energy = (fit_energy - truth_energy) / truth_energy
            delta_alt = abs(fit_alt - truth_alt)*180./np.pi
            delta_az = abs(fit_az - truth_az)*180./np.pi
            delta_az = min(delta_az,abs(360.-delta_az))

            list_log10_image_size += [np.log10(image_size)]
            list_log10_image_qual += [np.log10(image_qual)]
            list_delta_energy += [delta_energy]
            list_delta_arrival += [pow(delta_alt*delta_alt+delta_az*delta_az,0.5)]

    fig.clf()
    axbig = fig.add_subplot()
    label_x = 'log10 image size'
    label_y = 'arrival error [deg]'
    axbig.set_xlabel(label_x)
    axbig.set_ylabel(label_y)
    axbig.scatter(list_log10_image_size, list_delta_arrival, s=90, c='r', marker='+', alpha=0.2)
    fig.savefig(f'{ctapipe_output}/output_plots/monotel_arrival_error_vs_size.png',bbox_inches='tight')
    axbig.remove()

    fig.clf()
    axbig = fig.add_subplot()
    label_x = 'log10 direction quality'
    label_y = 'arrival error [deg]'
    axbig.set_xlabel(label_x)
    axbig.set_ylabel(label_y)
    axbig.scatter(list_log10_image_qual, list_delta_arrival, s=90, c='r', marker='+', alpha=0.2)
    fig.savefig(f'{ctapipe_output}/output_plots/monotel_arrival_error_vs_size.png',bbox_inches='tight')
    axbig.remove()

plot_monotel_analysis()
