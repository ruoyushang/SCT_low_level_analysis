
import os
import subprocess
import glob
import tracemalloc

import pickle
from matplotlib import pyplot as plt
from matplotlib import colors
import numpy as np
from astropy import units as u

from ctapipe.utils.datasets import get_dataset_path
from ctapipe.io import EventSource, SimTelEventSource, HDF5TableWriter

import torch
import torch.nn.functional as torchF

import common_functions
MyArray3D = common_functions.MyArray3D
linear_regression = common_functions.linear_regression
linear_model = common_functions.linear_model


ctapipe_output = os.environ.get("CTAPIPE_OUTPUT_PATH")
ctapipe_input = os.environ.get("CTAPIPE_SVC_PATH")
print (f'ctapipe_output = {ctapipe_output}')

overwrite_file = True
#overwrite_file = False

training_sample_path = []
#training_sample_path += [get_dataset_path("gamma_40deg_0deg_run2006___cta-prod3-sct_desert-2150m-Paranal-SCT.simtel.gz")]
#training_sample_path += [get_dataset_path("gamma_20deg_0deg_run744___cta-prod3-sct_desert-2150m-Paranal-SCT.simtel.gz")]
n_samples = 0
with open('%s/sim_files.txt'%(ctapipe_input), 'r') as file:
    for line in file:
        if n_samples==15: continue
        training_sample_path += [get_dataset_path(line.strip('\n'))]
        n_samples += 1

# start memory profiling
tracemalloc.start()

big_truth_matrix = []
big_moment_matrix = []
big_movie_matrix = []
big_image_matrix = []
big_time_matrix = []
for path in range(0,len(training_sample_path)):
    source = SimTelEventSource(training_sample_path[path], focal_length_choice='EQUIVALENT')
    subarray = source.subarray
    ob_keys = source.observation_blocks.keys()
    run_id = list(ob_keys)[0]
    output_filename = f'{ctapipe_output}/output_samples/training_sample_run{run_id}.pkl'
    print (f'loading pickle trainging sample data: {output_filename}')
    if not os.path.exists(output_filename):
        print (f'file does not exist.')
        continue
    training_sample = pickle.load(open(output_filename, "rb"))

    big_truth_matrix += training_sample[0]
    big_moment_matrix += training_sample[1]
    big_image_matrix += training_sample[2]
    big_time_matrix += training_sample[3]
    big_movie_matrix += training_sample[4]

    print(f'memory usage (current,peak) = {tracemalloc.get_traced_memory()}')

big_movie_matrix = np.array(big_movie_matrix)
big_image_matrix = np.array(big_image_matrix)
big_time_matrix = np.array(big_time_matrix)

fig, ax = plt.subplots()
figsize_x = 8.6
figsize_y = 6.4
fig.set_figheight(figsize_y)
fig.set_figwidth(figsize_x)


def BigMatrixSVD(big_matrix,moment_matrix,truth_matrix,image_rank,pkl_name):

    n_images, n_pixels = big_matrix.shape
    print (f'n_images = {n_images}, n_pixels = {n_pixels}')

    U_full, S_full, VT_full = np.linalg.svd(big_matrix,full_matrices=False)
    U_eco = U_full[:, :image_rank]
    VT_eco = VT_full[:image_rank, :]
    print(f'memory usage (current,peak) = {tracemalloc.get_traced_memory()}')

    if overwrite_file:
        print (f'saving image eigenvector to {ctapipe_output}/output_machines...')
        output_filename = f'{ctapipe_output}/output_machines/{pkl_name}_eigen_vectors.pkl'
        with open(output_filename,"wb") as file:
            pickle.dump(VT_eco, file)

    fig.clf()
    axbig = fig.add_subplot()
    label_x = 'Rank'
    label_y = 'Signular value'
    axbig.set_xlabel(label_x)
    axbig.set_ylabel(label_y)
    #axbig.set_xlim(0,100)
    axbig.set_xscale('log')
    axbig.plot(S_full)
    fig.savefig(f'{ctapipe_output}/output_plots/training_{pkl_name}_signularvalue.png',bbox_inches='tight')
    axbig.remove()

    MakeLookupTable(VT_eco,big_matrix,moment_matrix,truth_matrix,image_rank,pkl_name+'_box3d',nvar=3)
    MakeLookupTable(VT_eco,big_matrix,moment_matrix,truth_matrix,image_rank,pkl_name+'_box2d',nvar=2)
    MakeLookupTable(VT_eco,big_matrix,moment_matrix,truth_matrix,image_rank,pkl_name+'_box1d',nvar=1)

    if pkl_name=='movie':
        MakeLookupTableNN(VT_eco,big_matrix,moment_matrix,truth_matrix,image_rank,pkl_name+'_nn')

#class NeuralNetwork(torch.nn.Module):
#    def __init__(self,x_dim=100,y_dim=10,nodes=1):
#        super().__init__()
#        self.weights_1st = torch.nn.Parameter(torch.randn(x_dim, y_dim) / np.sqrt(x_dim))
#        self.weights_2nd = torch.nn.Parameter(torch.randn(x_dim, y_dim) / np.sqrt(x_dim))
#        self.bias = torch.nn.Parameter(torch.zeros(y_dim))
#
#    def forward(self, xb):
#        return (xb**2) @ self.weights_2nd + xb @ self.weights_1st + self.bias

class NeuralNetwork(torch.nn.Module):
    def __init__(self,x_dim=100,y_dim=10,nodes=20):
        super().__init__()
        self.flatten = torch.nn.Flatten()
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(x_dim, y_dim),
            #torch.nn.Linear(x_dim, int(x_dim/2)),
            #torch.nn.Sigmoid(),
            #torch.nn.Linear(int(x_dim/2), y_dim),
        )

    def forward(self, x):
        #x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

def mse_loss(input, target, weight):
    #return torch.sum((input - target) ** 2)
    return torch.sum(weight*(input - target) ** 2)

def fit(model,loss_func,x_train,y_train):

    lr = 1e-4  # learning rate
    for i in range(len(x_train)):
        xb = x_train[i]
        yb = y_train[i]
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        with torch.no_grad():
            for p in model.parameters():
                p -= p.grad * lr
            model.zero_grad()

def normalize_data(list_data):

    data_mean = np.mean(np.array(list_data))
    data_rms = np.sqrt(np.mean(np.square(np.array(list_data)-data_mean)))
    list_data = (np.array(list_data)-data_mean)/data_rms


def MakeLookupTableNN(eigenvectors,big_matrix,moment_matrix,truth_matrix,image_rank,pkl_name):

    list_image_size = []
    list_evt_weight = []
    list_arrival = []
    list_impact = []
    list_log_energy = []
    list_latent_space = []

    for img in range(0,len(big_matrix)):
    
        image_size = moment_matrix[img][0]
        image_center_x = moment_matrix[img][1]
        image_center_y = moment_matrix[img][2]
        time_direction = moment_matrix[img][6]
        image_direction = moment_matrix[img][7]
        image_qual = abs(image_direction+time_direction)

        truth_energy = float(truth_matrix[img][0]/u.TeV)
        truth_height = float(truth_matrix[img][5]/u.m)
        truth_x_max = float(truth_matrix[img][6]/(u.g/(u.cm*u.cm)))
        star_cam_x = truth_matrix[img][7]
        star_cam_y = truth_matrix[img][8]
        impact_x = truth_matrix[img][9]
        impact_y = truth_matrix[img][10]

        arrival = pow(pow(star_cam_x-image_center_x,2)+pow(star_cam_y-image_center_y,2),0.5)
        impact = pow(impact_x*impact_x+impact_y*impact_y,0.5)
        log_energy = np.log10(truth_energy)

        image_1d = np.array(big_matrix[img])
        image_latent_space = eigenvectors @ image_1d
        list_latent_space += [image_latent_space]

        list_image_size += [image_size]
        list_evt_weight += [pow(image_size,0.5)]
        list_arrival += [image_size/arrival]
        list_impact += [image_size/impact]
        list_log_energy += [pow(10.,log_energy)]

    target = list_arrival
    model, chi2 = linear_regression(list_latent_space, target, list_evt_weight)
    print (f'pkl_name = {pkl_name}')
    for img in range(0,len(target)):
        if img % 20 != 0: continue
        predict = linear_model(list_latent_space[img], model)
        print (f'arrival target = {target[img]}, predict = {predict}')
    print (f'chi2 = {chi2}')

    if overwrite_file:
        output_filename = f'{ctapipe_output}/output_machines/{pkl_name}_lookup_table_arrival_poly.pkl'
        with open(output_filename,"wb") as file:
            pickle.dump(model, file)

    target = list_impact
    model, chi2 = linear_regression(list_latent_space, target, list_evt_weight)
    print (f'pkl_name = {pkl_name}')
    for img in range(0,len(target)):
        if img % 20 != 0: continue
        predict = linear_model(list_latent_space[img], model)
        print (f'impact target = {target[img]}, predict = {predict}')
    print (f'chi2 = {chi2}')

    if overwrite_file:
        output_filename = f'{ctapipe_output}/output_machines/{pkl_name}_lookup_table_impact_poly.pkl'
        with open(output_filename,"wb") as file:
            pickle.dump(model, file)

    target = list_log_energy
    model, chi2 = linear_regression(list_latent_space, target, list_evt_weight)
    print (f'pkl_name = {pkl_name}')
    for img in range(0,len(target)):
        if img % 20 != 0: continue
        predict = linear_model(list_latent_space[img], model)
        print (f'log_energy target = {target[img]}, predict = {predict}')
    print (f'chi2 = {chi2}')

    if overwrite_file:
        output_filename = f'{ctapipe_output}/output_machines/{pkl_name}_lookup_table_log_energy_poly.pkl'
        with open(output_filename,"wb") as file:
            pickle.dump(model, file)


    #mean_weight = np.mean(list_evt_weight)
    #normalize_data(list_arrival)
    #normalize_data(list_impact)
    #normalize_data(list_log_energy)
    ##normalize_data(list_latent_space)

    #list_latent_space = torch.tensor(list_latent_space,dtype=torch.float32)
    #list_evt_weight = torch.tensor(list_evt_weight,dtype=torch.float32)
    #list_arrival = torch.tensor(list_arrival,dtype=torch.float32)
    #list_impact = torch.tensor(list_impact,dtype=torch.float32)
    #list_log_energy = torch.tensor(list_log_energy,dtype=torch.float32)

    #print (f'list_latent_space.dtype = {list_latent_space.dtype}')
    #model = NeuralNetwork(x_dim=len(list_latent_space[0]),y_dim=len(list_arrival[0]))
    ##loss_func = torch.nn.MSELoss(reduction='sum')
    #loss_func = mse_loss
    #print(f'loss of model = {loss_func(model(list_latent_space), list_arrival, list_evt_weight)}')
   
    #lr = 1e-6 / mean_weight # learning rate
    #for t in range(2000):
    #    y_pred = model(list_latent_space)
    #    loss = loss_func(model(list_latent_space), list_arrival, list_evt_weight)
    #    if t % 100 == 99:
    #        print(t, loss.item())
    #    loss.backward()
    #    with torch.no_grad():
    #        for p in model.parameters():
    #            p -= p.grad * lr

    #for i in range(0,len(list_latent_space)):
    #    if i % 10 == 9:
    #        #print (f'x = {list_latent_space[i]}')
    #        print (f'truth = {list_arrival[i]}')
    #        print (f'model = {model(list_latent_space[i])}')


    #exit()

def MakeLookupTable(eigenvectors,big_matrix,moment_matrix,truth_matrix,image_rank,pkl_name,nvar=3):

    n_bins_arrival = 40
    arrival_lower = 0.
    arrival_upper = 0.4
    n_bins_impact = 40
    impact_lower = 0.
    impact_upper = 800.
    n_bins_xmax = 10
    xmax_lower = 150.
    xmax_upper = 375.
    n_bins_height = 30
    height_lower = 10000.
    height_upper = 70000.
    n_bins_energy = 15
    log_energy_lower = -1.
    log_energy_upper = 2.

    if nvar==2:
        n_bins_energy = 1
    if nvar==1:
        n_bins_energy = 1
        n_bins_impact = 1


    lookup_table = []
    lookup_table_norm = MyArray3D(x_bins=n_bins_arrival,start_x=arrival_lower,end_x=arrival_upper,y_bins=n_bins_impact,start_y=impact_lower,end_y=impact_upper,z_bins=n_bins_energy,start_z=log_energy_lower,end_z=log_energy_upper)

    list_impact = []
    list_arrival = []
    list_log_energy = []
    list_height = []
    list_xmax = []
    list_image_qual = []

    for r in range(0,image_rank):
        lookup_table += [MyArray3D(x_bins=n_bins_arrival,start_x=arrival_lower,end_x=arrival_upper,y_bins=n_bins_impact,start_y=impact_lower,end_y=impact_upper,z_bins=n_bins_energy,start_z=log_energy_lower,end_z=log_energy_upper)]

    for img in range(0,len(big_matrix)):
    
        image_center_x = moment_matrix[img][1]
        image_center_y = moment_matrix[img][2]
        time_direction = moment_matrix[img][6]
        image_direction = moment_matrix[img][7]
        image_qual = abs(image_direction+time_direction)

        truth_energy = float(truth_matrix[img][0]/u.TeV)
        truth_height = float(truth_matrix[img][5]/u.m)
        truth_x_max = float(truth_matrix[img][6]/(u.g/(u.cm*u.cm)))
        star_cam_x = truth_matrix[img][7]
        star_cam_y = truth_matrix[img][8]
        impact_x = truth_matrix[img][9]
        impact_y = truth_matrix[img][10]

        arrival = pow(pow(star_cam_x-image_center_x,2)+pow(star_cam_y-image_center_y,2),0.5)
        impact = pow(impact_x*impact_x+impact_y*impact_y,0.5)
        log_energy = np.log10(truth_energy)

        list_log_energy += [log_energy]
        list_height += [truth_height]
        list_xmax += [truth_x_max]
        list_arrival += [arrival]
        list_impact += [impact]

        image_1d = np.array(big_matrix[img])
        image_latent_space = eigenvectors @ image_1d
        for r in range(0,image_rank):
            lookup_table[r].fill(arrival,impact,log_energy,weight=image_latent_space[r])

        lookup_table_norm.fill(arrival,impact,log_energy,weight=1.)

    for r in range(0,image_rank):
        lookup_table[r].divide(lookup_table_norm)

    n_empty_cells = 0.
    n_filled_cells = 0.
    n_training_images = 0.
    for idx_x in range(0,len(lookup_table_norm.xaxis)-1):
        for idx_y in range(0,len(lookup_table_norm.yaxis)-1):
            for idx_z in range(0,len(lookup_table_norm.zaxis)-1):
                count = lookup_table_norm.waxis[idx_x,idx_y,idx_z]
                if count==0:
                    n_empty_cells += 1.
                else:
                    n_filled_cells += 1.
                n_training_images += count
    avg_images_per_cell = n_training_images/n_filled_cells
    print (f'n_empty_cells = {n_empty_cells}, n_filled_cells = {n_filled_cells}, n_training_images = {n_training_images}, avg_images_per_cell = {avg_images_per_cell:0.1f}')

    if overwrite_file:
        output_filename = f'{ctapipe_output}/output_machines/{pkl_name}_lookup_table.pkl'
        with open(output_filename,"wb") as file:
            pickle.dump(lookup_table, file)


    print (f'saving plots to {ctapipe_output}/output_plots...')

    fig.clf()
    axbig = fig.add_subplot()
    label_x = 'log10 Energy [TeV]'
    label_y = 'Height [m]'
    axbig.set_xlabel(label_x)
    axbig.set_ylabel(label_y)
    axbig.scatter(list_log_energy, list_height, s=90, c='b', marker='+', alpha=0.1)
    fig.savefig(f'{ctapipe_output}/output_plots/scatter_log_energy_vs_height.png',bbox_inches='tight')
    axbig.remove()
    
    fig.clf()
    axbig = fig.add_subplot()
    label_x = 'log10 Energy [TeV]'
    label_y = 'X max [g/cm2]'
    axbig.set_xlabel(label_x)
    axbig.set_ylabel(label_y)
    axbig.scatter(list_log_energy, list_xmax, s=90, c='b', marker='+', alpha=0.1)
    fig.savefig(f'{ctapipe_output}/output_plots/scatter_log_energy_vs_xmax.png',bbox_inches='tight')
    axbig.remove()
    
    fig.clf()
    axbig = fig.add_subplot()
    label_x = 'Arrival [m]'
    label_y = 'Impact [m]'
    axbig.set_xlabel(label_x)
    axbig.set_ylabel(label_y)
    axbig.scatter(list_arrival, list_impact, s=90, c='b', marker='+', alpha=0.1)
    fig.savefig(f'{ctapipe_output}/output_plots/scatter_arrival_vs_impact.png',bbox_inches='tight')
    axbig.remove()
    
    fig.clf()
    axbig = fig.add_subplot()
    label_x = 'Arrival [m]'
    label_y = 'X max [g/cm2]'
    axbig.set_xlabel(label_x)
    axbig.set_ylabel(label_y)
    axbig.scatter(list_arrival, list_xmax, s=90, c='b', marker='+', alpha=0.1)
    fig.savefig(f'{ctapipe_output}/output_plots/scatter_arrival_vs_xmax.png',bbox_inches='tight')
    axbig.remove()

    fig.clf()
    axbig = fig.add_subplot()
    label_x = 'log10 Energy [TeV]'
    label_y = 'Arrival [m]'
    axbig.set_xlabel(label_x)
    axbig.set_ylabel(label_y)
    axbig.scatter(list_log_energy, list_arrival, s=90, c='b', marker='+', alpha=0.1)
    fig.savefig(f'{ctapipe_output}/output_plots/scatter_log_energy_vs_arrival.png',bbox_inches='tight')
    axbig.remove()
    

print ('Compute movie matrix SVD...')
BigMatrixSVD(big_movie_matrix,big_moment_matrix,big_truth_matrix,50,'movie')
print ('Compute image matrix SVD...')
BigMatrixSVD(big_image_matrix,big_moment_matrix,big_truth_matrix,50,'image')
print ('Compute time matrix SVD...')
BigMatrixSVD(big_time_matrix,big_moment_matrix,big_truth_matrix,200,'time')



tracemalloc.stop()
