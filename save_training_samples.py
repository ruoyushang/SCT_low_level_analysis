import os, sys
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

training_sample_path = sys.argv[1]

ctapipe_output = os.environ.get("CTAPIPE_OUTPUT_PATH")
subprocess.call(['sh', './clean_plots.sh'])
#subprocess.call(['sh', './clean_samples.sh'])

load_training_samples(training_sample_path,is_training=True,use_truth=False,do_cleaning=True,do_reposition=True,min_energy=0.1,max_energy=100.0,max_evt=1e10)
load_training_samples(training_sample_path,is_training=False,use_truth=False,do_cleaning=True,do_reposition=False,min_energy=0.1,max_energy=100.0,max_evt=1e10)

#load_training_samples(training_sample_path,is_gamma=False,is_training=True,use_truth=False,do_cleaning=True,do_reposition=True,min_energy=0.1,max_energy=100.0,max_evt=1e10)
#load_training_samples(training_sample_path,is_gamma=False,is_training=False,use_truth=False,do_cleaning=True,do_reposition=False,min_energy=0.1,max_energy=100.0,max_evt=1e10)



