
import os
import subprocess
from ctapipe import utils
from ctapipe.utils.datasets import get_dataset_path
from ctapipe.io import EventSource, SimTelEventSource, HDF5TableWriter

from ctapipe.reco.veritas_utilities import run_save_training_matrix

ctapipe_output = os.environ.get("CTAPIPE_OUTPUT_PATH")
ctapipe_input = os.environ.get("CTAPIPE_SVC_PATH")

subprocess.call(f'rm {ctapipe_output}/output_plots/*.png', shell=True)

list_telescope_type = []

#sim_files = 'sct_onaxis_train.txt'
#list_telescope_type += ['MST_SCT_SCTCam']

sim_files = 'mst_onaxis_train.txt'
#sim_files = 'mst_mix_train.txt'
list_telescope_type += ['MST_MST_NectarCam']
#list_telescope_type += ['MST_MST_FlashCam']
#list_telescope_type += ['SST_1M_DigiCam']
#list_telescope_type += ['SST_ASTRI_ASTRICam']
#list_telescope_type += ['SST_GCT_CHEC']
#list_telescope_type += ['LST_LST_LSTCam']

#sim_files = 'sct_onaxis_test.txt'
#sim_files = 'sct_diffuse_all.txt'
#sim_files = 'mst_onaxis_test.txt'
#sim_files = 'mst_diffuse_all.txt'

with open(f'{ctapipe_input}/{sim_files}', 'r') as file:
    for line in file:
        training_sample_path = get_dataset_path(line.strip('\n'))
        print (f'training_sample_path = {training_sample_path}')
        run_id = line.split("_")[3].strip("run")
        print (f"run_id = {run_id}")
        #if run_id!="898": continue
        for tel in range(0,len(list_telescope_type)):
            telescope_type = list_telescope_type[tel]
            run_save_training_matrix(training_sample_path,telescope_type,ctapipe_output,use_star_coord=False)

