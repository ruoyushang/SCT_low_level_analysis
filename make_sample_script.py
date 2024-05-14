
import os, sys
import subprocess

from ctapipe import utils
from ctapipe.utils.datasets import get_dataset_path
from ctapipe.io import EventSource, SimTelEventSource, HDF5TableWriter

ctapipe_output = os.environ.get("CTAPIPE_OUTPUT_PATH")
ctapipe_work = os.environ.get("CTAPIPE_WORK_PATH")
ctapipe_input = os.environ.get("CTAPIPE_SVC_PATH")

sim_files = 'sim_files.txt'
#sim_files = 'sim_files_diffuse_gamma.txt'
#sim_files = 'sim_files_merged_point_20deg.txt'

list_tel_type = []
list_tel_type += ['MST_SCT_SCTCam']
#list_tel_type += ['MST_MST_NectarCam']
#list_tel_type += ['MST_MST_FlashCam']
#list_tel_type += ['SST_1M_DigiCam']
#list_tel_type += ['SST_ASTRI_ASTRICam']
#list_tel_type += ['SST_GCT_CHEC']
#list_tel_type += ['LST_LST_LSTCam']

n_samples = 0
runlist = []
with open(f'{ctapipe_input}/{sim_files}', 'r') as file:
    for line in file:
        #if n_samples==15: continue
        training_sample_path = get_dataset_path(line.strip('\n'))
        print (f'loading file: {training_sample_path}')
        source = SimTelEventSource(training_sample_path, focal_length_choice='EQUIVALENT')
        ob_keys = source.observation_blocks.keys()
        run_id = list(ob_keys)[0]
        runlist += [run_id]
        file = open("run/save_sample_run%s.sh"%(run_id),"w") 
        file.write('cd %s\n'%(ctapipe_work))
        for tel_type in range(0,len(list_tel_type)):
            file.write('python3 save_training_matrix.py "%s" "%s"\n'%(training_sample_path,list_tel_type[tel_type]))
        file.close() 
        n_samples += 1

job_counts = 0
qfile = open("run/condor_ctapipe_sample.sh","w") 
for s in range(0,len(runlist)):
    job_counts += 1
    run_id = runlist[s]
    qfile.write('universe = vanilla \n')
    qfile.write('getenv = true \n')
    qfile.write('executable = /bin/bash \n')
    qfile.write('arguments = save_sample_run%s.sh\n'%(run_id))
    qfile.write('request_cpus = 1 \n')
    qfile.write('request_memory = 1024M \n')
    qfile.write('request_disk = 1024M \n')
    qfile.write('output = sample_run%s.out\n'%(run_id))
    qfile.write('error = sample_run%s.err\n'%(run_id))
    qfile.write('log = sample_run%s.log\n'%(run_id))
    qfile.write('queue\n')
qfile.close() 

job_counts = 0
qfile = open("run/local_ctapipe_sample.sh","w") 
for s in range(0,len(runlist)):
    job_counts += 1
    run_id = runlist[s]
    qfile.write('sh save_sample_run%s.sh\n'%(run_id))
qfile.close() 

file = open("run/compile_big_matrices.sh"%(run_id),"w") 
file.write('cd %s\n'%(ctapipe_work))
for tel_type in range(0,len(list_tel_type)):
    file.write('python3 compile_big_matrices.py "%s"\n'%(list_tel_type[tel_type]))
file.close() 

file = open("run/build_eigenvectors.sh"%(run_id),"w") 
file.write('cd %s\n'%(ctapipe_work))
for tel_type in range(0,len(list_tel_type)):
    file.write('python3 build_eigenvectors.py "%s"\n'%(list_tel_type[tel_type]))
file.close() 

qfile = open("run/condor_ctapipe_eigenvector.sh","w") 
qfile.write('universe = vanilla \n')
qfile.write('getenv = true \n')
qfile.write('executable = /bin/bash \n')
qfile.write('arguments = build_eigenvectors.sh\n')
qfile.write('request_cpus = 1 \n')
qfile.write('request_memory = 1024M \n')
qfile.write('request_disk = 1024M \n')
qfile.write('output = eigenvector.out\n')
qfile.write('error = eigenvector.err\n')
qfile.write('log = eigenvector.log\n')
qfile.write('queue\n')
qfile.close() 


runlist = []
with open(f'{ctapipe_input}/{sim_files}', 'r') as file:
    for line in file:
        training_sample_path = get_dataset_path(line.strip('\n'))
        print (f'loading file: {training_sample_path}')
        source = SimTelEventSource(training_sample_path, focal_length_choice='EQUIVALENT')
        ob_keys = source.observation_blocks.keys()
        run_id = list(ob_keys)[0]
        runlist += [run_id]
        file = open("run/analyze_monotel_run%s.sh"%(run_id),"w") 
        file.write('cd %s\n'%(ctapipe_work))
        for tel_type in range(0,len(list_tel_type)):
            file.write('python3 monotel_analysis.py "%s" "%s"\n'%(training_sample_path,list_tel_type[tel_type]))
        file.close() 

job_counts = 0
qfile = open("run/condor_ctapipe_monotel.sh","w") 
for s in range(0,len(runlist)):
    job_counts += 1
    run_id = runlist[s]
    qfile.write('universe = vanilla \n')
    qfile.write('getenv = true \n')
    qfile.write('executable = /bin/bash \n')
    qfile.write('arguments = analyze_monotel_run%s.sh\n'%(run_id))
    qfile.write('request_cpus = 1 \n')
    qfile.write('request_memory = 1024M \n')
    qfile.write('request_disk = 1024M \n')
    qfile.write('output = monotel_run%s.out\n'%(run_id))
    qfile.write('error = monotel_run%s.err\n'%(run_id))
    qfile.write('log = monotel_run%s.log\n'%(run_id))
    qfile.write('queue\n')
qfile.close() 

job_counts = 0
qfile = open("run/local_ctapipe_monotel.sh","w") 
for s in range(0,len(runlist)):
    job_counts += 1
    run_id = runlist[s]
    qfile.write('sh analyze_monotel_run%s.sh\n'%(run_id))
qfile.close() 

