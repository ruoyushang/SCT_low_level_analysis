
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
from load_cta_data import rank_brightest_telescope
from load_cta_data import image_translation
from load_cta_data import image_rotation
from load_cta_data import smooth_map
from load_cta_data import NeuralNetwork

ctapipe_output = os.environ.get("CTAPIPE_OUTPUT_PATH")
subprocess.call(['sh', './clean.sh'])

testing_sample_path = []
#testing_sample_path += [get_dataset_path("gamma_40deg_0deg_run1933___cta-prod3-sct_desert-2150m-Paranal-SCT_cone10.simtel.gz")]
#testing_sample_path += [get_dataset_path("gamma_20deg_0deg_run742___cta-prod3-sct_desert-2150m-Paranal-SCT.simtel.gz")]
#testing_sample_path += [get_dataset_path("gamma_20deg_0deg_run876___cta-prod3-sct_desert-2150m-Paranal-SCT.simtel.gz")]
#testing_sample_path += [get_dataset_path("gamma_20deg_0deg_run860___cta-prod3-sct_desert-2150m-Paranal-SCT.simtel.gz")]
#testing_sample_path += [get_dataset_path("gamma_20deg_0deg_run859___cta-prod3-sct_desert-2150m-Paranal-SCT.simtel.gz")]
#testing_sample_path += [get_dataset_path("gamma_20deg_0deg_run853___cta-prod3-sct_desert-2150m-Paranal-SCT.simtel.gz")]

with open('test_sim_files.txt', 'r') as file:
    for line in file:
        testing_sample_path += [get_dataset_path(line.strip('\n'))]

print ('loading testing data... ')
load_training_samples(testing_sample_path,False,min_energy=0.1,max_energy=100.0,max_evt=1e10)

#testing_id_list, big_telesc_position_matrix, big_truth_shower_position_matrix, test_cam_axes, big_testing_image_matrix, big_testing_param_matrix = load_training_samples(testing_sample_path,False,min_energy=0.1,max_energy=100.0,max_evt=1e1)

#output_filename = f'{ctapipe_output}/output_machines/testing_sample.pkl'
#with open(output_filename,"wb") as file:
#    pickle.dump([testing_id_list, big_telesc_position_matrix, big_truth_shower_position_matrix, test_cam_axes, big_testing_image_matrix, big_testing_param_matrix], file)

