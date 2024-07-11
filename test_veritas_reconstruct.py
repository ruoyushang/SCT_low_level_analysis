
import os
import subprocess
import pickle
from ctapipe import utils
from ctapipe.utils.datasets import get_dataset_path
from ctapipe.io import EventSource, SimTelEventSource, HDF5TableWriter

from ctapipe.reco.veritas_utilities import loop_all_events

ctapipe_output = os.environ.get("CTAPIPE_OUTPUT_PATH")
ctapipe_input = os.environ.get("CTAPIPE_SVC_PATH")

subprocess.call(f'rm {ctapipe_output}/output_plots/*.png', shell=True)

telescope_type = []
telescope_type += ['MST_SCT_SCTCam']
#telescope_type += ['MST_MST_NectarCam']
#telescope_type += ['MST_MST_FlashCam']
#telescope_type += ['LST_LST_LSTCam']
#telescope_type += ['SST_ASTRI_ASTRICam']
#telescope_type += ['SST_GCT_CHEC']
#telescope_type += ['SST_1M_DigiCam']

sim_files = 'sct_onaxis_train.txt'
#sim_files = 'sct_onaxis_test.txt'
#sim_files = 'sct_diffuse_all.txt'
#sim_files = 'mst_onaxis_train.txt'
#sim_files = 'mst_onaxis_test.txt'
#sim_files = 'mst_diffuse_all.txt'

ana_tag = 'clean_param_2_5_2'
with open(f'{ctapipe_input}/{sim_files}', 'r') as file:
    for line in file:
        training_sample_path = get_dataset_path(line.strip('\n'))

        run_id = line.split("_")[3].strip("run")
        print (f"run_id = {run_id}")
        #if int(run_id)<=2000: continue

        loop_all_events(ana_tag,training_sample_path,ctapipe_output,telescope_type,save_output=True)

