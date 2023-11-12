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

training_sample_path = []
#training_sample_path += [get_dataset_path("gamma_40deg_0deg_run1933___cta-prod3-sct_desert-2150m-Paranal-SCT_cone10.simtel.gz")]
training_sample_path += [get_dataset_path("gamma_20deg_0deg_run876___cta-prod3-sct_desert-2150m-Paranal-SCT.simtel.gz")]
training_sample_path += [get_dataset_path("gamma_20deg_0deg_run860___cta-prod3-sct_desert-2150m-Paranal-SCT.simtel.gz")]
training_sample_path += [get_dataset_path("gamma_20deg_0deg_run859___cta-prod3-sct_desert-2150m-Paranal-SCT.simtel.gz")]
training_sample_path += [get_dataset_path("gamma_20deg_0deg_run853___cta-prod3-sct_desert-2150m-Paranal-SCT.simtel.gz")]


print ('loading training data... ')
training_id_list, training_telesc_position_matrix, training_truth_shower_position_matrix, train_cam_axes, big_training_image_matrix, big_training_param_matrix = load_training_samples(training_sample_path,True)

output_filename = f'{ctapipe_output}/output_machines/training_sample.pkl'
with open(output_filename,"wb") as file:
    pickle.dump([training_id_list, training_telesc_position_matrix, training_truth_shower_position_matrix, train_cam_axes, big_training_image_matrix, big_training_param_matrix], file)


