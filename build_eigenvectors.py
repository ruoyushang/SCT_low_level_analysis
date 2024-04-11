
import os
import subprocess
import glob
import tracemalloc

import pickle
from matplotlib import pyplot as plt
from matplotlib import colors
import numpy as np
from astropy import units as u
from scipy.interpolate import LinearNDInterpolator
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from ctapipe.utils.datasets import get_dataset_path
from ctapipe.io import EventSource, SimTelEventSource, HDF5TableWriter

import common_functions
MyArray3D = common_functions.MyArray3D

ctapipe_output = os.environ.get("CTAPIPE_OUTPUT_PATH")
ctapipe_input = os.environ.get("CTAPIPE_SVC_PATH")
print (f'ctapipe_output = {ctapipe_output}')

training_sample_path = []
#training_sample_path += [get_dataset_path("gamma_40deg_0deg_run2006___cta-prod3-sct_desert-2150m-Paranal-SCT.simtel.gz")]
#training_sample_path += [get_dataset_path("gamma_20deg_0deg_run744___cta-prod3-sct_desert-2150m-Paranal-SCT.simtel.gz")]
n_samples = 0
with open('%s/sim_files.txt'%(ctapipe_input), 'r') as file:
    for line in file:
        #if n_samples==15: continue
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


def make_spline_lookup_table(lookup_table,lookup_table_norm,rank):

    #xobs = []
    #yobs = []
    #zobs = []
    #wobs = []
    #for idx_x in range(0,len(lookup_table.xaxis)-1):
    #    for idx_y in range(0,len(lookup_table.yaxis)-1):
    #        for idx_z in range(0,len(lookup_table.zaxis)-1):
    #            if lookup_table_norm.waxis[idx_x,idx_y,idx_z]==0.: continue
    #            xobs += [lookup_table.xaxis[idx_x]]
    #            yobs += [lookup_table.yaxis[idx_y]]
    #            zobs += [lookup_table.zaxis[idx_z]]
    #            wobs += [lookup_table.waxis[idx_x,idx_y,idx_z]]

    #spline_lookup_table = LinearNDInterpolator(list(zip(np.array(xobs), np.array(yobs), np.array(zobs))), np.array(wobs))
    #return spline_lookup_table

    xobs = []
    wobs = []
    for idx_x in range(0,len(lookup_table.xaxis)-1):
        for idx_y in range(0,len(lookup_table.yaxis)-1):
            for idx_z in range(0,len(lookup_table.zaxis)-1):
                if lookup_table_norm.waxis[idx_x,idx_y,idx_z]==0.: continue
                xobs += [[lookup_table.xaxis[idx_x],lookup_table.yaxis[idx_y],lookup_table.zaxis[idx_z]]]
                wobs += [lookup_table.waxis[idx_x,idx_y,idx_z]]

    poly = PolynomialFeatures(degree=4)
    X_poly = poly.fit_transform(np.array(xobs))
    model = LinearRegression()
    model.fit(X_poly, np.array(wobs))

    if rank<5:
        print (f'inpertolating table rank {rank}')
        chi2 = 0.
        norm = 0.
        for idx_x in range(0,len(lookup_table.xaxis)-1):
            for idx_y in range(0,len(lookup_table.yaxis)-1):
                for idx_z in range(0,len(lookup_table.zaxis)-1):
                    if lookup_table.waxis[idx_x,idx_y,idx_z]==0.: continue
                    norm += pow(lookup_table.waxis[idx_x,idx_y,idx_z],2)
                    new_data = np.array([[lookup_table.xaxis[idx_x], lookup_table.yaxis[idx_y], lookup_table.zaxis[idx_z]]])
                    new_data_poly = poly.transform(new_data)
                    chi2 += pow(lookup_table.waxis[idx_x,idx_y,idx_z]-model.predict(new_data_poly)[0],2)
        print (f'chi2/norm = {chi2/norm:0.3f}')

    #new_data = np.array([[0.1, 100., 0.]])
    #new_data_poly = poly.transform(new_data)
    #predicted_y = model.predict(new_data_poly)
    #print (f'predicted_y = ')
    #print (f'{predicted_y}')
    
    return [poly,model]


def BigMatrixSVD(big_matrix,moment_matrix,truth_matrix,image_rank,pkl_name):

    n_images, n_pixels = big_matrix.shape
    print (f'n_images = {n_images}, n_pixels = {n_pixels}')

    U_full, S_full, VT_full = np.linalg.svd(big_matrix,full_matrices=False)
    U_eco = U_full[:, :image_rank]
    VT_eco = VT_full[:image_rank, :]
    print(f'memory usage (current,peak) = {tracemalloc.get_traced_memory()}')

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
        image_latent_space = VT_eco @ image_1d
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

    output_filename = f'{ctapipe_output}/output_machines/{pkl_name}_lookup_table.pkl'
    with open(output_filename,"wb") as file:
        pickle.dump(lookup_table, file)

    lookup_table_spline = []
    for rank in range(0,len(lookup_table)):
        lookup_table_spline += [make_spline_lookup_table(lookup_table[rank],lookup_table_norm,rank)]

    output_filename = f'{ctapipe_output}/output_machines/{pkl_name}_lookup_table_spline.pkl'
    with open(output_filename,"wb") as file:
        pickle.dump(lookup_table_spline, file)

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
