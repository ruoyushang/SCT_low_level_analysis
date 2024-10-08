
import os
import subprocess
import pickle
import numpy as np
from ctapipe import utils
from ctapipe.utils.datasets import get_dataset_path
from ctapipe.io import EventSource, SimTelEventSource, HDF5TableWriter

from ctapipe.reco.veritas_utilities import run_save_training_matrix
import ctapipe.reco.veritas_utilities as veritas_utilities
image_size_bins = veritas_utilities.image_size_bins
image_size_cut_analysis = veritas_utilities.image_size_cut_analysis
mask_size_cut = veritas_utilities.mask_size_cut
frac_leakage_intensity_cut_analysis = veritas_utilities.frac_leakage_intensity_cut_analysis
image_direction_significance_cut = veritas_utilities.image_direction_significance_cut

ctapipe_output = os.environ.get("CTAPIPE_OUTPUT_PATH")
ctapipe_input = os.environ.get("CTAPIPE_SVC_PATH")

subprocess.call(f'rm {ctapipe_output}/output_plots/*.png', shell=True)

#sim_files = 'sct_onaxis_train.txt'
#telescope_type = 'MST_SCT_SCTCam'

sim_files = 'mst_mix_train.txt'
#sim_files = 'mst_diffuse_train.txt'
telescope_type = 'MST_MST_NectarCam'
#telescope_type = 'MST_MST_FlashCam'
#telescope_type = 'SST_1M_DigiCam'
#telescope_type = 'SST_ASTRI_ASTRICam'
#telescope_type = 'SST_GCT_CHEC'
#telescope_type = 'LST_LST_LSTCam'

#sim_files = 'sct_onaxis_test.txt'
#sim_files = 'sct_diffuse_all.txt'
#sim_files = 'mst_onaxis_test.txt'
#sim_files = 'mst_diffuse_all.txt'

big_truth_matrix = []
big_moment_matrix = []
big_image_matrix = []
big_time_matrix = []
big_movie_matrix = []
for idx in range(0,len(image_size_bins)-1):
    big_truth_matrix += [[]]
    big_moment_matrix += [[]]
    big_image_matrix += [[]]
    big_time_matrix += [[]]
    big_movie_matrix += [[]]

with open(f'{ctapipe_input}/{sim_files}', 'r') as file:
    for line in file:

        #training_sample_path = get_dataset_path(line.strip('\n'))
        #source = SimTelEventSource(training_sample_path, focal_length_choice='EQUIVALENT')
        #subarray = source.subarray
        #ob_keys = source.observation_blocks.keys()
        #run_id = list(ob_keys)[0]

        training_sample_path = line.strip('\n')
        run_id = training_sample_path.split("_")[3].strip("run")
        output_filename = f'{ctapipe_output}/output_samples/training_sample_run{run_id}_{telescope_type}.pkl'
        print (f'loading pickle trainging sample data: {output_filename}')
        if not os.path.exists(output_filename):
            print (f'file does not exist.')
            continue
        training_sample = pickle.load(open(output_filename, "rb"))

        truth_matrix = training_sample[0]
        moment_matrix = training_sample[1]
        image_matrix = training_sample[2]
        time_matrix = training_sample[3]
        movie_matrix = training_sample[4]

        for evt in range(0,len(moment_matrix)):
            mask_size = moment_matrix[evt][0]
            image_size = moment_matrix[evt][14]
            frac_leakage_intensity = moment_matrix[evt][15]
            if mask_size < mask_size_cut: continue
            if image_size < image_size_cut_analysis: continue
            if image_size > image_size_bins[len(image_size_bins)-1]: continue
            if frac_leakage_intensity>frac_leakage_intensity_cut_analysis: continue

            image_direction_err = moment_matrix[evt][6]
            image_direction = moment_matrix[evt][7]
            image_direction_significance = abs(image_direction)/image_direction_err
            #if image_direction_significance<image_direction_significance_cut: continue


            image_idx = 0
            for idx in range(0,len(image_size_bins)-1):
                if image_size>=image_size_bins[idx] and image_size<image_size_bins[idx+1]:
                    image_idx = idx

            big_truth_matrix[image_idx] += [truth_matrix[evt]]
            big_moment_matrix[image_idx] += [moment_matrix[evt]]
            big_image_matrix[image_idx] += [image_matrix[evt]]
            big_time_matrix[image_idx] += [time_matrix[evt]]
            big_movie_matrix[image_idx] += [movie_matrix[evt]]


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

output_filename = f'{ctapipe_output}/output_machines/big_movie_matrix_{telescope_type}.pkl'
with open(output_filename,"wb") as file:
    pickle.dump(big_movie_matrix, file)

