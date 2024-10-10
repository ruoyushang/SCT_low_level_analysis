
import os
import subprocess
import pickle
from ctapipe import utils
from ctapipe.utils.datasets import get_dataset_path
from ctapipe.io import EventSource, SimTelEventSource, HDF5TableWriter

from ctapipe.reco.veritas_utilities import BigMatrixSVD
from ctapipe.reco.veritas_utilities import mapping_physical_params_to_latent_params
import ctapipe.reco.veritas_utilities as veritas_utilities
matrix_rank = veritas_utilities.matrix_rank

ctapipe_output = os.environ.get("CTAPIPE_OUTPUT_PATH")
ctapipe_input = os.environ.get("CTAPIPE_SVC_PATH")

subprocess.call(f'rm {ctapipe_output}/output_plots/*.png', shell=True)

#telescope_type = 'MST_SCT_SCTCam'
telescope_type = 'MST_MST_NectarCam'
#telescope_type = 'MST_MST_FlashCam'
#telescope_type = 'SST_1M_DigiCam'
#telescope_type = 'SST_ASTRI_ASTRICam'
#telescope_type = 'SST_GCT_CHEC'
#telescope_type = 'LST_LST_LSTCam'


output_filename = f'{ctapipe_output}/output_machines/big_truth_matrix_{telescope_type}.pkl'
big_truth_matrix = pickle.load(open(output_filename, "rb"))

output_filename = f'{ctapipe_output}/output_machines/big_moment_matrix_{telescope_type}.pkl'
big_moment_matrix = pickle.load(open(output_filename, "rb"))

output_filename = f'{ctapipe_output}/output_machines/big_image_matrix_{telescope_type}.pkl'
big_image_matrix = pickle.load(open(output_filename, "rb"))

output_filename = f'{ctapipe_output}/output_machines/big_time_matrix_{telescope_type}.pkl'
big_time_matrix= pickle.load(open(output_filename, "rb"))

output_filename = f'{ctapipe_output}/output_machines/big_movie_matrix_{telescope_type}.pkl'
big_movie_matrix = pickle.load(open(output_filename, "rb"))

print ('Compute movie matrix SVD...')
movie_eigenvectors, physics_eigenvectors, physics_mean_rms = BigMatrixSVD(ctapipe_output,telescope_type,big_movie_matrix,big_moment_matrix,big_truth_matrix,matrix_rank,'movie')
print ('Compute image matrix SVD...')
image_eigenvectors, physics_eigenvectors, physics_mean_rms = BigMatrixSVD(ctapipe_output,telescope_type,big_image_matrix,big_moment_matrix,big_truth_matrix,matrix_rank,'image')
print ('Compute time matrix SVD...')
time_eigenvectors, physics_eigenvectors, physics_mean_rms = BigMatrixSVD(ctapipe_output,telescope_type,big_time_matrix,big_moment_matrix,big_truth_matrix,matrix_rank,'time')


#mapping_physical_params_to_latent_params(ctapipe_output,telescope_type,physics_eigenvectors,image_eigenvectors,big_image_matrix,big_moment_matrix,big_truth_matrix,matrix_rank,'image')
mapping_physical_params_to_latent_params(ctapipe_output,telescope_type,physics_eigenvectors,movie_eigenvectors,big_movie_matrix,big_moment_matrix,big_truth_matrix,matrix_rank,'movie')

