
import os
import subprocess
import pickle
from ctapipe import utils
from ctapipe.utils.datasets import get_dataset_path
from ctapipe.io import EventSource, SimTelEventSource, HDF5TableWriter

from ctapipe.reco.veritas_utilities import loop_all_events

ctapipe_output = os.environ.get("CTAPIPE_OUTPUT_PATH")
ctapipe_input = os.environ.get("CTAPIPE_SVC_PATH")

#subprocess.call(f'rm {ctapipe_output}/output_plots/*.png', shell=True)

#diagnosis = False
diagnosis = True

select_evt = []
#[402, 886208, 1.1188316446304112, 0.16720824735226741]
#[403, 868404, 2.371480574261248, 0.24881383423026607]
#[405, 209203, 1.8347507338701627, 0.31281798691756796]
#[405, 614507, 1.511795814111109, 0.34615395820739164]
run_id, event_id = 403, 868404
select_evt = [run_id, event_id]

evt_selection = 'loose'
#evt_selection = 'freepact'

#array_type = 'LST_Nectar_ASTRI'
#array_type = 'SCT'
array_type = 'Nectar'
#array_type = 'Flash'
#array_type = 'LST'
#array_type = 'ASTRI'
#array_type = 'CHEC'
#array_type = 'DigiCam'
#array_type = 'MIX'

pointing = 'onaxis'
#pointing = 'diffuse'

template = 'yes'
#template = 'no'

ana_tag = f"{array_type}_{evt_selection}_{pointing}_{template}"

telescope_type = []
if 'SCT' in ana_tag:
    telescope_type += ['MST_SCT_SCTCam']
if 'Nectar' in ana_tag:
  telescope_type += ['MST_MST_NectarCam']
if 'Flash' in ana_tag:
  telescope_type += ['MST_MST_FlashCam']
if 'LST' in ana_tag:
  telescope_type += ['LST_LST_LSTCam']
if 'ASTRI' in ana_tag:
  telescope_type += ['SST_ASTRI_ASTRICam']
if 'CHEC' in ana_tag:
  telescope_type += ['SST_GCT_CHEC']
if 'DigiCam' in ana_tag:
    telescope_type += ['SST_1M_DigiCam']
if 'MIX' in ana_tag:
    telescope_type += ['MST_MST_NectarCam']
    telescope_type += ['MST_MST_FlashCam']
    telescope_type += ['LST_LST_LSTCam']
    telescope_type += ['SST_ASTRI_ASTRICam']
    telescope_type += ['SST_GCT_CHEC']
    telescope_type += ['SST_1M_DigiCam']

sim_files = None
if 'SCT' in ana_tag:
    if 'onaxis' in ana_tag:
        #sim_files = 'sct_onaxis_train.txt'
        #sim_files = 'sct_onaxis_test.txt'
        sim_files = 'sct_onaxis_all.txt'
    else:
        sim_files = 'sct_diffuse_all.txt'
else:
    if 'onaxis' in ana_tag:
        #sim_files = 'mst_onaxis_train.txt'
        sim_files = 'mst_onaxis_test.txt'
        #sim_files = 'mst_onaxis_all.txt'
    else:
        sim_files = 'mst_diffuse_all.txt'

with open(f'{ctapipe_input}/{sim_files}', 'r') as file:
    for line in file:
        training_sample_path = get_dataset_path(line.strip('\n'))

        run_id = line.split("_")[3].strip("run")
        print (f"run_id = {run_id}")

        if len(select_evt)>0:
            if int(run_id)<select_evt[0]: continue
            loop_all_events(ana_tag,training_sample_path,ctapipe_output,telescope_type,select_evt=select_evt,save_output=False)
        elif diagnosis:
            loop_all_events(ana_tag,training_sample_path,ctapipe_output,telescope_type,save_output=False)
        else:
            loop_all_events(ana_tag,training_sample_path,ctapipe_output,telescope_type,save_output=True)

