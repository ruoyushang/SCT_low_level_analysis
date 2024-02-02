import os
import subprocess
import glob

import time
import math
import numpy as np
from astropy import units as u
from scipy.optimize import least_squares, minimize, brute, dual_annealing
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib import patheffects
import pickle

from ctapipe import utils
from ctapipe.utils.datasets import get_dataset_path
from ctapipe.calib import CameraCalibrator
from ctapipe.image import hillas_parameters, tailcuts_clean, ImageProcessor
from ctapipe.reco import ShowerProcessor
from ctapipe.io import EventSource, SimTelEventSource, HDF5TableWriter
from ctapipe.visualization import CameraDisplay

from load_cta_data import load_training_samples
from load_cta_data import rank_brightest_telescope
from load_cta_data import image_translation
from load_cta_data import image_rotation
from load_cta_data import NeuralNetwork
from load_cta_data import sqaure_difference_between_1d_images
from load_cta_data import find_image_moments_guided
from load_cta_data import get_average
from load_cta_data import MyArray2D
from load_cta_data import MyArray3D
from load_cta_data import single_image_reconstruction

ctapipe_output = os.environ.get("CTAPIPE_OUTPUT_PATH")
ctapipe_input = os.environ.get("CTAPIPE_SVC_PATH")
subprocess.call(['sh', './clean_plots.sh'])

fig, ax = plt.subplots()
figsize_x = 8.6
figsize_y = 6.4
fig.set_figheight(figsize_y)
fig.set_figwidth(figsize_x)

font = {'family': 'serif', 'color':  'black', 'weight': 'normal', 'size': 10, 'rotation': 0.,}

training_sample_path = []
particle_type = []
with open('%s/sim_files.txt'%(ctapipe_input), 'r') as file:
    for line in file:
        training_sample_path += [get_dataset_path(line.strip('\n'))]
        particle_type += [0]

def plot_single_tel_analysis():

    arrival_error = []
    impact_error = []
    energy_error = []
    log_energy_truth = []
    impact_truth = []
    arrival_truth = []
    log_energy_guess = []
    impact_guess = []
    arrival_guess = []
    log_image_size = []
    delta_time = []
    image_center_r = []
    semi_major = []
    semi_minor = []
    semi_ratio = []
    impact_time_ratio = []

    for path in range(0,len(training_sample_path)):
    
        source = SimTelEventSource(training_sample_path[path], focal_length_choice='EQUIVALENT')
        subarray = source.subarray
        ob_keys = source.observation_blocks.keys()
        run_id = list(ob_keys)[0]
    
        input_filename = f'{ctapipe_output}/output_analysis/single_tel_ana_output_run{run_id}.pkl'
        print (f'loading pickle analysis data: {input_filename}')
        if not os.path.exists(input_filename):
            print (f'file does not exist.')
            #return
            continue
        analysis_result = pickle.load(open(input_filename, "rb"))
    
        for img in range(0,len(analysis_result[2])):
            img_arrival_error = float(analysis_result[0][img])
            img_log_energy_truth = float(analysis_result[1][img])
            img_impact_truth = float(analysis_result[2][img])
            img_arrival_truth = float(analysis_result[3][img])
            img_log_energy_guess = float(analysis_result[4][img])
            img_impact_guess = float(analysis_result[5][img])
            img_arrival_guess = float(analysis_result[6][img])
            img_log_image_size = float(analysis_result[7][img])
            img_delta_time = float(analysis_result[8][img])
            img_image_center_r = float(analysis_result[9][img])
            img_semi_major = float(analysis_result[10][img])
            img_semi_minor = float(analysis_result[11][img])
            img_semi_ratio = img_semi_major/img_semi_minor
            if img_delta_time<=0.: continue
            if img_image_center_r>0.3: continue
            #if img_impact_guess<=0.07: continue
            #if img_impact_guess>=0.20: continue

            arrival_error += [img_arrival_error]
            impact_error += [img_impact_guess-img_impact_truth]
            energy_error += [(pow(10.,img_log_energy_guess)-pow(10.,img_log_energy_truth))/pow(10.,img_log_energy_truth)]
            log_energy_truth += [img_log_energy_truth]
            impact_truth += [img_impact_truth]
            arrival_truth += [img_arrival_truth]
            log_energy_guess += [img_log_energy_guess]
            impact_guess += [img_impact_guess]
            arrival_guess += [img_arrival_guess]
            log_image_size += [img_log_image_size]
            delta_time += [img_delta_time]
            image_center_r += [img_image_center_r]
            semi_major += [img_semi_major]
            semi_minor += [img_semi_minor]
            semi_ratio += [img_semi_major/img_semi_minor]
            impact_time_ratio += [np.log10(img_impact_guess/img_delta_time)]

    
        max_ratio = float(int(10.*np.max(semi_ratio)))/10.
        min_ratio = float(int(10.*np.min(semi_ratio)))/10.
        nbin_ratio = 10
        delta_ratio = (max_ratio-min_ratio)/float(nbin_ratio)
        log_ratio_axis = []
        for x in range(0,nbin_ratio):
            log_ratio_axis += [(min_ratio+x*delta_ratio)]
        hist_sky_err_vs_ratio = get_average(semi_ratio,pow(np.array(arrival_error),2),log_ratio_axis)
        hist_sky_err_vs_ratio.yaxis = pow(np.array(hist_sky_err_vs_ratio.yaxis),0.5)

        max_energy = float(int(10.*np.max(log_energy_truth)))/10.
        min_energy = float(int(10.*np.min(log_energy_truth)))/10.
        nbin_energy = 10
        delta_energy = (max_energy-min_energy)/float(nbin_energy)
        log_energy_axis = []
        for x in range(0,nbin_energy):
            log_energy_axis += [(min_energy+x*delta_energy)]
        hist_energy_err_vs_energy = get_average(log_energy_truth,pow(np.array(energy_error),2),log_energy_axis)
        hist_energy_err_vs_energy.yaxis = pow(np.array(hist_energy_err_vs_energy.yaxis),0.5)
        hist_sky_err_vs_energy = get_average(log_energy_truth,pow(np.array(arrival_error),2),log_energy_axis)
        hist_sky_err_vs_energy.yaxis = pow(np.array(hist_sky_err_vs_energy.yaxis),0.5)
        print (f'hist_sky_err_vs_energy.yaxis = {hist_sky_err_vs_energy.yaxis}')
        
        max_size = float(int(10.*np.max(log_image_size)))/10.
        min_size = float(int(10.*np.min(log_image_size)))/10.
        nbin_size = 10
        delta_size = (max_size-min_size)/float(nbin_size)
        log_size_axis = []
        for x in range(0,nbin_size):
            log_size_axis += [(min_size+x*delta_size)]
        hist_sky_err_vs_size = get_average(log_image_size,pow(np.array(arrival_error),2),log_size_axis)
        hist_sky_err_vs_size.yaxis = pow(np.array(hist_sky_err_vs_size.yaxis),0.5)

        output_filename = f'{ctapipe_output}/output_machines/template_reconstruction_error.pkl'
        with open(output_filename,"wb") as file:
            pickle.dump(hist_sky_err_vs_size, file)
        
        max_impact = float(int(100.*np.max(impact_guess)))/100.
        min_impact = float(int(100.*np.min(impact_guess)))/100.
        nbin_impact = 10
        hist_sky_err_vs_size_impact = MyArray2D(x_bins=nbin_size,start_x=min_size,end_x=max_size,y_bins=nbin_impact,start_y=min_impact,end_y=max_impact)
        hist_norm_vs_size_impact = MyArray2D(x_bins=nbin_size,start_x=min_size,end_x=max_size,y_bins=nbin_impact,start_y=min_impact,end_y=max_impact)
        for img in range(0,len(arrival_error)):
            img_arrival_error = arrival_error[img]
            img_log_image_size = log_image_size[img]
            img_impact_guess = impact_guess[img]
            hist_sky_err_vs_size_impact.fill(img_log_image_size,img_impact_guess,weight=pow(img_arrival_error,2))
            hist_norm_vs_size_impact.fill(img_log_image_size,img_impact_guess,weight=1.)
        hist_sky_err_vs_size_impact.divide(hist_norm_vs_size_impact)
        hist_sky_err_vs_size_impact.zaxis = np.sqrt(hist_sky_err_vs_size_impact.zaxis)
        
        hist_sky_err_vs_size_ratio = MyArray2D(x_bins=nbin_size,start_x=min_size,end_x=max_size,y_bins=nbin_ratio,start_y=min_ratio,end_y=max_ratio)
        hist_norm_vs_size_ratio = MyArray2D(x_bins=nbin_size,start_x=min_size,end_x=max_size,y_bins=nbin_ratio,start_y=min_ratio,end_y=max_ratio)
        for img in range(0,len(arrival_error)):
            img_arrival_error = arrival_error[img]
            img_log_image_size = log_image_size[img]
            img_ratio = semi_ratio[img]
            hist_sky_err_vs_size_ratio.fill(img_log_image_size,img_ratio,weight=pow(img_arrival_error,2))
            hist_norm_vs_size_ratio.fill(img_log_image_size,img_ratio,weight=1.)
        hist_sky_err_vs_size_ratio.divide(hist_norm_vs_size_ratio)
        hist_sky_err_vs_size_ratio.zaxis = np.sqrt(hist_sky_err_vs_size_ratio.zaxis)
        
        output_filename = f'{ctapipe_output}/output_machines/template_reconstruction_error_size_impact_2d.pkl'
        with open(output_filename,"wb") as file:
            pickle.dump(hist_sky_err_vs_size_impact, file)
    
        output_filename = f'{ctapipe_output}/output_machines/template_reconstruction_error_size_ratio_2d.pkl'
        with open(output_filename,"wb") as file:
            pickle.dump(hist_sky_err_vs_size_ratio, file)

        #fig.clf()
        #axbig = fig.add_subplot()
        #label_x = 'log energy truth'
        #label_y = 'log energy predict'
        #axbig.set_xlabel(label_x)
        #axbig.set_ylabel(label_y)
        #axbig.scatter(log_energy_truth, log_energy_guess, s=90, c='r', marker='+', alpha=0.2)
        #fig.savefig(f'{ctapipe_output}/output_plots/reconstruction_energy_guess.png',bbox_inches='tight')
        #axbig.remove()
        #
        #fig.clf()
        #axbig = fig.add_subplot()
        #label_x = 'impact distance truth'
        #label_y = 'impact distance predict'
        #axbig.set_xlabel(label_x)
        #axbig.set_ylabel(label_y)
        #axbig.scatter(impact_truth, impact_guess, s=90, c='r', marker='+', alpha=0.2)
        #fig.savefig(f'{ctapipe_output}/output_plots/reconstruction_impact_guess.png',bbox_inches='tight')
        #axbig.remove()
        #
        #fig.clf()
        #axbig = fig.add_subplot()
        #label_x = 'arrival truth'
        #label_y = 'arrival predict'
        #axbig.set_xlabel(label_x)
        #axbig.set_ylabel(label_y)
        #axbig.scatter(arrival_truth, arrival_guess, s=90, c='r', marker='+', alpha=0.2)
        #fig.savefig(f'{ctapipe_output}/output_plots/reconstruction_arrival_guess.png',bbox_inches='tight')
        #axbig.remove()

        fig.clf()
        axbig = fig.add_subplot()
        label_x = 'log10 size'
        label_y = 'impact'
        axbig.set_xlabel(label_x)
        axbig.set_ylabel(label_y)
        xmin = hist_sky_err_vs_size_impact.xaxis.min()
        xmax = hist_sky_err_vs_size_impact.xaxis.max()
        ymin = hist_sky_err_vs_size_impact.yaxis.min()
        ymax = hist_sky_err_vs_size_impact.yaxis.max()
        im = axbig.imshow(hist_sky_err_vs_size_impact.zaxis[:,:].T,origin='lower',extent=(xmin,xmax,ymin,ymax),aspect='auto',vmin=0.,vmax=0.3)
        cbar = fig.colorbar(im)
        fig.savefig(f'{ctapipe_output}/output_plots/hist_sky_err_vs_size_impact.png',bbox_inches='tight')
        axbig.remove()
        
        fig.clf()
        axbig = fig.add_subplot()
        label_x = 'log10 size'
        label_y = 'semi-axis ratio'
        axbig.set_xlabel(label_x)
        axbig.set_ylabel(label_y)
        xmin = hist_sky_err_vs_size_ratio.xaxis.min()
        xmax = hist_sky_err_vs_size_ratio.xaxis.max()
        ymin = hist_sky_err_vs_size_ratio.yaxis.min()
        ymax = hist_sky_err_vs_size_ratio.yaxis.max()
        im = axbig.imshow(hist_sky_err_vs_size_ratio.zaxis[:,:].T,origin='lower',extent=(xmin,xmax,ymin,ymax),aspect='auto',vmin=0.,vmax=0.3)
        cbar = fig.colorbar(im)
        fig.savefig(f'{ctapipe_output}/output_plots/hist_sky_err_vs_size_ratio.png',bbox_inches='tight')
        axbig.remove()
        
        fig.clf()
        axbig = fig.add_subplot()
        label_x = 'log10 size'
        label_y = 'impact'
        axbig.set_xlabel(label_x)
        axbig.set_ylabel(label_y)
        xmin = hist_sky_err_vs_size_impact.xaxis.min()
        xmax = hist_sky_err_vs_size_impact.xaxis.max()
        ymin = hist_sky_err_vs_size_impact.yaxis.min()
        ymax = hist_sky_err_vs_size_impact.yaxis.max()
        im = axbig.imshow(hist_norm_vs_size_impact.zaxis[:,:].T,origin='lower',extent=(xmin,xmax,ymin,ymax),aspect='auto')
        cbar = fig.colorbar(im)
        fig.savefig(f'{ctapipe_output}/output_plots/hist_norm_vs_size_impact.png',bbox_inches='tight')
        axbig.remove()
        
        fig.clf()
        axbig = fig.add_subplot()
        label_x = 'log10 size'
        label_y = 'semi-axis ratio'
        axbig.set_xlabel(label_x)
        axbig.set_ylabel(label_y)
        xmin = hist_sky_err_vs_size_ratio.xaxis.min()
        xmax = hist_sky_err_vs_size_ratio.xaxis.max()
        ymin = hist_sky_err_vs_size_ratio.yaxis.min()
        ymax = hist_sky_err_vs_size_ratio.yaxis.max()
        im = axbig.imshow(hist_norm_vs_size_ratio.zaxis[:,:].T,origin='lower',extent=(xmin,xmax,ymin,ymax),aspect='auto')
        cbar = fig.colorbar(im)
        fig.savefig(f'{ctapipe_output}/output_plots/hist_norm_vs_size_ratio.png',bbox_inches='tight')
        axbig.remove()
        
        fig.clf()
        axbig = fig.add_subplot()
        label_x = 'log10 image size'
        label_y = 'impact error [km]'
        axbig.set_xlabel(label_x)
        axbig.set_ylabel(label_y)
        axbig.scatter(log_image_size, impact_error, s=90, c='r', marker='+', alpha=0.2)
        fig.savefig(f'{ctapipe_output}/output_plots/reconstruction_impact_error_vs_size.png',bbox_inches='tight')
        axbig.remove()
    
        fig.clf()
        axbig = fig.add_subplot()
        label_x = 'log10 image size'
        label_y = 'arrival error [deg]'
        axbig.set_xlabel(label_x)
        axbig.set_ylabel(label_y)
        axbig.scatter(log_image_size, arrival_error, s=90, c='r', marker='+', alpha=0.2)
        axbig.errorbar(hist_sky_err_vs_size.xaxis,np.zeros_like(hist_sky_err_vs_size.xaxis),yerr=hist_sky_err_vs_size.yaxis,c='k')
        for x in range(0,len(hist_sky_err_vs_size.xaxis)):
            axbig.text(hist_sky_err_vs_size.xaxis[x], hist_sky_err_vs_size.yaxis[x], '%0.2f'%(hist_sky_err_vs_size.yaxis[x]))
        fig.savefig(f'{ctapipe_output}/output_plots/reconstruction_arrival_error_vs_size.png',bbox_inches='tight')
        axbig.remove()
    
        fig.clf()
        axbig = fig.add_subplot()
        label_x = 'impact / delta time'
        label_y = 'arrival error [deg]'
        axbig.set_xlabel(label_x)
        axbig.set_ylabel(label_y)
        axbig.scatter(impact_time_ratio, arrival_error, s=90, c='r', marker='+', alpha=0.2)
        fig.savefig(f'{ctapipe_output}/output_plots/reconstruction_arrival_error_vs_speed.png',bbox_inches='tight')
        axbig.remove()
    
        fig.clf()
        axbig = fig.add_subplot()
        label_x = 'image center r'
        label_y = 'arrival error [deg]'
        axbig.set_xlabel(label_x)
        axbig.set_ylabel(label_y)
        axbig.scatter(image_center_r, arrival_error, s=90, c='r', marker='+', alpha=0.2)
        fig.savefig(f'{ctapipe_output}/output_plots/reconstruction_arrival_error_vs_center.png',bbox_inches='tight')
        axbig.remove()
    
        fig.clf()
        axbig = fig.add_subplot()
        label_x = 'log10 energy (truth)'
        label_y = 'arrival error [deg]'
        axbig.set_xlabel(label_x)
        axbig.set_ylabel(label_y)
        axbig.scatter(log_energy_truth, arrival_error, s=90, c='r', marker='+', alpha=0.2)
        axbig.errorbar(hist_sky_err_vs_energy.xaxis,np.zeros_like(hist_sky_err_vs_energy.xaxis),yerr=hist_sky_err_vs_energy.yaxis,c='k')
        for x in range(0,len(hist_sky_err_vs_energy.xaxis)):
            axbig.text(hist_sky_err_vs_energy.xaxis[x], hist_sky_err_vs_energy.yaxis[x], '%0.2f'%(hist_sky_err_vs_energy.yaxis[x]))
        fig.savefig(f'{ctapipe_output}/output_plots/reconstruction_arrival_error_vs_energy.png',bbox_inches='tight')
        axbig.remove()
    
        fig.clf()
        axbig = fig.add_subplot()
        label_x = 'log10 energy (truth)'
        label_y = '$\delta E / E$'
        axbig.set_xlabel(label_x)
        axbig.set_ylabel(label_y)
        axbig.scatter(log_energy_truth, energy_error, s=90, c='r', marker='+', alpha=0.2)
        axbig.errorbar(hist_energy_err_vs_energy.xaxis,np.zeros_like(hist_energy_err_vs_energy.xaxis),yerr=hist_energy_err_vs_energy.yaxis,c='k')
        for x in range(0,len(hist_energy_err_vs_energy.xaxis)):
            axbig.text(hist_energy_err_vs_energy.xaxis[x], hist_energy_err_vs_energy.yaxis[x], '%0.2f'%(hist_energy_err_vs_energy.yaxis[x]))
        fig.savefig(f'{ctapipe_output}/output_plots/reconstruction_energy_error_vs_energy.png',bbox_inches='tight')
        axbig.remove()
    
        fig.clf()
        axbig = fig.add_subplot()
        label_x = 'semi major'
        label_y = 'arrival error [deg]'
        axbig.set_xlabel(label_x)
        axbig.set_ylabel(label_y)
        axbig.scatter(semi_major, arrival_error, s=90, c='r', marker='+', alpha=0.2)
        fig.savefig(f'{ctapipe_output}/output_plots/reconstruction_arrival_error_vs_major.png',bbox_inches='tight')
        axbig.remove()
    
        fig.clf()
        axbig = fig.add_subplot()
        label_x = 'semi-axis ratio'
        label_y = 'arrival error [deg]'
        axbig.set_xlabel(label_x)
        axbig.set_ylabel(label_y)
        axbig.scatter(semi_ratio, arrival_error, s=90, c='r', marker='+', alpha=0.2)
        axbig.errorbar(hist_sky_err_vs_ratio.xaxis,np.zeros_like(hist_sky_err_vs_ratio.xaxis),yerr=hist_sky_err_vs_ratio.yaxis,c='k')
        for x in range(0,len(hist_sky_err_vs_ratio.xaxis)):
            axbig.text(hist_sky_err_vs_ratio.xaxis[x], hist_sky_err_vs_ratio.yaxis[x], '%0.2f'%(hist_sky_err_vs_ratio.yaxis[x]))
        fig.savefig(f'{ctapipe_output}/output_plots/reconstruction_arrival_error_vs_ratio.png',bbox_inches='tight')
        axbig.remove()
    
        fig.clf()
        axbig = fig.add_subplot()
        label_x = 'delta time'
        label_y = 'arrival [deg]'
        axbig.set_xlabel(label_x)
        axbig.set_ylabel(label_y)
        axbig.scatter(delta_time, arrival_truth, s=90, c='r', marker='+', alpha=0.2)
        fig.savefig(f'{ctapipe_output}/output_plots/truth_arrival_vs_time.png',bbox_inches='tight')
        axbig.remove()
    
        fig.clf()
        axbig = fig.add_subplot()
        label_x = 'semi minor'
        label_y = 'arrival error [deg]'
        axbig.set_xlabel(label_x)
        axbig.set_ylabel(label_y)
        axbig.scatter(semi_minor, arrival_error, s=90, c='r', marker='+', alpha=0.2)
        fig.savefig(f'{ctapipe_output}/output_plots/reconstruction_arrival_error_vs_minor.png',bbox_inches='tight')
        axbig.remove()
    
        fig.clf()
        axbig = fig.add_subplot()
        label_x = 'arrival (reconstuct)'
        label_y = 'arrival error [deg]'
        axbig.set_xlabel(label_x)
        axbig.set_ylabel(label_y)
        axbig.scatter(arrival_guess, arrival_error, s=90, c='r', marker='+', alpha=0.2)
        fig.savefig(f'{ctapipe_output}/output_plots/reconstruction_arrival_error_vs_arrival.png',bbox_inches='tight')
        axbig.remove()
    
        fig.clf()
        axbig = fig.add_subplot()
        label_x = 'impact (reconstuct)'
        label_y = 'arrival error [deg]'
        axbig.set_xlabel(label_x)
        axbig.set_ylabel(label_y)
        axbig.scatter(impact_guess, arrival_error, s=90, c='r', marker='+', alpha=0.2)
        axbig.grid()
        fig.savefig(f'{ctapipe_output}/output_plots/reconstruction_arrival_error_vs_impact.png',bbox_inches='tight')
        axbig.remove()

def gaussian(x, y, sigma, A):
    x0 = 0.
    y0 = 0.
    return A * np.exp(-((x-x0)**2+(y-y0)**2)/(2*sigma*sigma))/(2*np.pi*sigma*sigma)
# https://scipython.com/blog/non-linear-least-squares-fitting-of-a-two-dimensional-data/
# This is the callable that is passed to curve_fit. M is a (2,N) array
# where N is the total number of data points in Z, which will be ravelled
# to one dimension.
def _gaussian(M, *args):
    x, y = M
    arr = np.zeros(x.shape)
    n_var = 2
    for i in range(len(args)//n_var):
       arr += gaussian(x, y, *args[i*n_var:i*n_var+n_var])
    return arr

def plot_cta_array_analysis():

    event_count = 0
    comb_fit_psf = MyArray3D(x_bins=80,start_x=-0.4,end_x=0.4,y_bins=80,start_y=-0.4,end_y=0.4,z_bins=1,start_z=-1.,end_z=2.,overflow=False)

    all_truth_energy = []
    all_hillas_valid = []
    all_temp_valid = []
    all_line_valid = []
    all_comb_valid = []
    hillas_truth_energy = []
    temp_truth_energy = []
    line_truth_energy = []
    comb_truth_energy = []
    hillas_sky_err = []
    line_fit_sky_err = []
    temp_fit_sky_err = []
    comb_fit_sky_err = []
    hillas_open_angle = []
    line_open_angle = []
    temp_open_angle = []
    comb_open_angle = []
    hillas_image_size = []
    line_image_size = []
    temp_image_size = []
    comb_image_size = []
    line_fit_cam_x_err = []
    line_fit_cam_y_err = []
    temp_fit_cam_x_err = []
    temp_fit_cam_y_err = []
    comb_fit_cam_x_err = []
    comb_fit_cam_y_err = []
    temp_fit_evt_err = []
    temp_fit_evt_rms = []
    for path in range(0,len(training_sample_path)):
    
        source = SimTelEventSource(training_sample_path[path], focal_length_choice='EQUIVALENT')
        subarray = source.subarray
        ob_keys = source.observation_blocks.keys()
        run_id = list(ob_keys)[0]
    
        input_filename = f'{ctapipe_output}/output_analysis/cta_array_ana_output_run{run_id}.pkl'
        print (f'loading pickle analysis data: {input_filename}')
        if not os.path.exists(input_filename):
            print (f'file does not exist.')
            #return
            continue
        analysis_result = pickle.load(open(input_filename, "rb"))

        event_count += len(analysis_result[0])
        print (f'event_count = {event_count}')

        for evt in range(0,len(analysis_result[0])):
            evt_truth_energy = analysis_result[0][evt]
            hillas_valid = analysis_result[1][evt]
            evt_temp_fit_energy = analysis_result[2][evt]
            evt_truth_sky_alt = analysis_result[3][evt]
            evt_hillas_sky_alt = analysis_result[4][evt]
            evt_line_fit_sky_alt = analysis_result[5][evt]
            evt_temp_fit_sky_alt = analysis_result[6][evt]
            evt_truth_sky_az = analysis_result[7][evt]
            evt_hillas_sky_az = analysis_result[8][evt]
            evt_line_fit_sky_az = analysis_result[9][evt]
            evt_temp_fit_sky_az = analysis_result[10][evt]
            evt_open_angle = analysis_result[11][evt]
            evt_line_impact = analysis_result[12][evt]
            evt_temp_impact = analysis_result[13][evt]
            evt_image_size = analysis_result[14][evt]
            evt_ntel = analysis_result[17][evt]
            evt_image_line_rms = analysis_result[18][evt]
            evt_image_temp_rms = analysis_result[19][evt]
            evt_image_temp_err = analysis_result[20][evt]
            line_valid = 1
            temp_valid = 1
            comb_valid = 1

            if evt_ntel<3: continue
            if evt_image_line_rms==0.: evt_image_line_rms = 1e10
            if evt_image_temp_rms==0.: evt_image_temp_rms = 1e10
            if evt_image_temp_err==0.: evt_image_temp_err = 1e10
            line_fit_weight = 1./evt_image_line_rms
            temp_fit_weight = 1./evt_image_temp_err
            total_weight = line_fit_weight+temp_fit_weight

            all_truth_energy += [evt_truth_energy]

            evt_hillas_sky_err = pow(pow(evt_truth_sky_alt-evt_hillas_sky_alt,2)+pow(evt_truth_sky_az-evt_hillas_sky_az,2),0.5)*180./math.pi
            evt_line_fit_sky_err = pow(pow(evt_truth_sky_alt-evt_line_fit_sky_alt,2)+pow(evt_truth_sky_az-evt_line_fit_sky_az,2),0.5)*180./math.pi
            evt_temp_fit_sky_err = pow(pow(evt_truth_sky_alt-evt_temp_fit_sky_alt,2)+pow(evt_truth_sky_az-evt_temp_fit_sky_az,2),0.5)*180./math.pi

            if evt_line_fit_sky_err>4.0: line_valid = 0
            if evt_temp_fit_sky_err>4.0: temp_valid = 0
            if evt_hillas_sky_err>4.0: hillas_valid = 0

            if evt_open_angle<100. or evt_image_size<30. or evt_image_temp_err>0.023 or evt_image_temp_rms>0.023:
                temp_valid = 0
            if evt_open_angle<200. or evt_image_line_rms>0.023:
                line_valid = 0

            #if evt_open_angle<100. or evt_image_size<30. or evt_image_temp_err>0.015 or evt_image_temp_rms>0.023:
            #    temp_valid = 0
            #if evt_open_angle<200. or evt_image_line_rms>0.023:
            #    line_valid = 0

            if temp_valid==0 and line_valid==0:
                comb_valid = 0

            if hillas_valid==1:
                all_hillas_valid += [1]
                hillas_sky_err += [evt_hillas_sky_err]
                hillas_truth_energy += [evt_truth_energy]
                hillas_open_angle += [evt_open_angle]
                hillas_image_size += [evt_image_size]
            else:
                all_hillas_valid += [0]

            if temp_valid==1:
                all_temp_valid += [1]
                temp_fit_sky_err += [evt_temp_fit_sky_err]
                temp_fit_cam_x_err += [(evt_truth_sky_alt-evt_temp_fit_sky_alt)*180./math.pi]
                temp_fit_cam_y_err += [(evt_truth_sky_az-evt_temp_fit_sky_az)*180./math.pi]
                temp_truth_energy += [evt_truth_energy]
                temp_open_angle += [evt_open_angle]
                temp_image_size += [evt_image_size]
                temp_fit_evt_err += [evt_image_temp_err]
                temp_fit_evt_rms += [evt_image_temp_rms]
            else:
                all_temp_valid += [0]

            if line_valid==1:
                all_line_valid += [1]
                line_fit_sky_err += [evt_line_fit_sky_err]
                line_fit_cam_x_err += [(evt_truth_sky_alt-evt_line_fit_sky_alt)*180./math.pi]
                line_fit_cam_y_err += [(evt_truth_sky_az-evt_line_fit_sky_az)*180./math.pi]
                line_truth_energy += [evt_truth_energy]
                line_open_angle += [evt_open_angle]
                line_image_size += [evt_image_size]
            else:
                all_line_valid += [0]

            if comb_valid==1:
                all_comb_valid += [1]
                evt_comb_fit_sky_alt = (temp_fit_weight*evt_temp_fit_sky_alt+line_fit_weight*evt_line_fit_sky_alt)/total_weight
                evt_comb_fit_sky_az = (temp_fit_weight*evt_temp_fit_sky_az+line_fit_weight*evt_line_fit_sky_az)/total_weight
                if temp_valid==0:
                    evt_comb_fit_sky_alt = evt_line_fit_sky_alt
                    evt_comb_fit_sky_az = evt_line_fit_sky_az
                elif line_valid==0:
                    evt_comb_fit_sky_alt = evt_temp_fit_sky_alt
                    evt_comb_fit_sky_az = evt_temp_fit_sky_az
                evt_comb_fit_sky_err = pow(pow(evt_truth_sky_alt-evt_comb_fit_sky_alt,2)+pow(evt_truth_sky_az-evt_comb_fit_sky_az,2),0.5)*180./math.pi
                comb_fit_sky_err += [evt_comb_fit_sky_err]
                comb_fit_cam_x_err += [(evt_truth_sky_alt-evt_comb_fit_sky_alt)*180./math.pi]
                comb_fit_cam_y_err += [(evt_truth_sky_az-evt_comb_fit_sky_az)*180./math.pi]
                comb_truth_energy += [evt_truth_energy]
                comb_open_angle += [evt_open_angle]
                comb_image_size += [evt_image_size]
                comb_fit_psf.fill((evt_truth_sky_alt-evt_temp_fit_sky_alt)*180./math.pi,(evt_truth_sky_az-evt_temp_fit_sky_az)*180./math.pi,np.log10(evt_truth_energy))
            else:
                all_comb_valid += [0]

    
        max_energy = float(int(10.*np.max(np.log10(all_truth_energy))))/10.
        min_energy = float(int(10.*np.min(np.log10(all_truth_energy))))/10.
        nbin_energy = 4
        delta_energy = (max_energy-min_energy)/float(nbin_energy)
        log_energy_axis = []
        for x in range(0,nbin_energy):
            log_energy_axis += [(min_energy+x*delta_energy)]

        hillas_sky_err_vs_energy = get_average(hillas_truth_energy,pow(np.array(hillas_sky_err),2),pow(10.,np.array(log_energy_axis)))
        hillas_sky_err_vs_energy.yaxis = pow(np.array(hillas_sky_err_vs_energy.yaxis),0.5)
        print (f'hillas_sky_err_vs_energy.yaxis = {hillas_sky_err_vs_energy.yaxis}')

        line_fit_sky_err_vs_energy = get_average(line_truth_energy,pow(np.array(line_fit_sky_err),2),pow(10.,np.array(log_energy_axis)))
        line_fit_sky_err_vs_energy.yaxis = pow(np.array(line_fit_sky_err_vs_energy.yaxis),0.5)
        print (f'line_fit_sky_err_vs_energy.yaxis = {line_fit_sky_err_vs_energy.yaxis}')

        temp_fit_sky_err_vs_energy = get_average(temp_truth_energy,pow(np.array(temp_fit_sky_err),2),pow(10.,np.array(log_energy_axis)))
        temp_fit_sky_err_vs_energy.yaxis = pow(np.array(temp_fit_sky_err_vs_energy.yaxis),0.5)
        print (f'temp_fit_sky_err_vs_energy.yaxis = {temp_fit_sky_err_vs_energy.yaxis}')

        comb_fit_sky_err_vs_energy = get_average(comb_truth_energy,pow(np.array(comb_fit_sky_err),2),pow(10.,np.array(log_energy_axis)))
        comb_fit_sky_err_vs_energy.yaxis = pow(np.array(comb_fit_sky_err_vs_energy.yaxis),0.5)
        print (f'comb_fit_sky_err_vs_energy.yaxis = {comb_fit_sky_err_vs_energy.yaxis}')

        hillas_frac_vs_energy = get_average(all_truth_energy,np.array(all_hillas_valid),pow(10.,np.array(log_energy_axis)))
        temp_frac_vs_energy = get_average(all_truth_energy,np.array(all_temp_valid),pow(10.,np.array(log_energy_axis)))
        line_frac_vs_energy = get_average(all_truth_energy,np.array(all_line_valid),pow(10.,np.array(log_energy_axis)))
        comb_frac_vs_energy = get_average(all_truth_energy,np.array(all_comb_valid),pow(10.,np.array(log_energy_axis)))
        

        for z in range(0,len(comb_fit_psf.zaxis)-1):

            xmin = comb_fit_psf.xaxis.min()
            xmax = comb_fit_psf.xaxis.max()
            ymin = comb_fit_psf.yaxis.min()
            ymax = comb_fit_psf.yaxis.max()
            nbins_x = len(comb_fit_psf.xaxis)
            nbins_y = len(comb_fit_psf.yaxis)

            initial_prms = [(0.1,100.)]
            bound_lower_prms = [(0.,0.)]
            bound_upper_prms = [(3.0,1e6)]
            # Flatten the initial guess parameter list.
            p0 = [p for prms in initial_prms for p in prms]
            p0_lower = [p for prms in bound_lower_prms for p in prms]
            p0_upper = [p for prms in bound_upper_prms for p in prms]

            x_axis = np.linspace(xmin,xmax,nbins_x)
            y_axis = np.linspace(ymin,ymax,nbins_y)
            X_grid, Y_grid = np.meshgrid(x_axis, y_axis)
            # We need to ravel the meshgrids of X, Y points to a pair of 1-D arrays.
            XY_stack = np.vstack((X_grid.ravel(), Y_grid.ravel()))
            popt, pcov = curve_fit(_gaussian, XY_stack, comb_fit_psf.waxis[:,:,z].ravel(), p0, bounds=(p0_lower,p0_upper))
            temp_psf = popt[0]
            print ('temp_psf = %0.3f deg'%(temp_psf))

            fig.clf()
            axbig = fig.add_subplot()
            label_x = 'X [deg]'
            label_y = 'Y [deg]'
            axbig.set_xlabel(label_x)
            axbig.set_ylabel(label_y)
            im = axbig.imshow(comb_fit_psf.waxis[:,:,z].T,origin='lower',extent=(xmin,xmax,ymin,ymax),aspect='auto')
            cbar = fig.colorbar(im)
            fig.savefig(f'{ctapipe_output}/output_plots/comb_fit_psf_z{z}.png',bbox_inches='tight')
            axbig.remove()
        

        fig.clf()
        axbig = fig.add_subplot()
        label_x = 'energy'
        label_y = 'Hillas reconstruction fraction'
        axbig.set_xlabel(label_x)
        axbig.set_ylabel(label_y)
        axbig.plot(hillas_frac_vs_energy.xaxis,hillas_frac_vs_energy.yaxis,label='Hillas')
        axbig.plot(line_frac_vs_energy.xaxis,line_frac_vs_energy.yaxis,label='Line')
        axbig.plot(temp_frac_vs_energy.xaxis,temp_frac_vs_energy.yaxis,label='Template')
        axbig.plot(comb_frac_vs_energy.xaxis,comb_frac_vs_energy.yaxis,label='Combined')
        axbig.set_xscale('log')
        #axbig.set_yscale('log')
        axbig.legend(loc='best')
        fig.savefig(f'{ctapipe_output}/output_plots/frac_vs_energy.png',bbox_inches='tight')
        axbig.remove()
    
        
        fig.clf()
        axbig = fig.add_subplot()
        label_x = 'energy'
        label_y = 'arrival error [deg]'
        axbig.set_xlabel(label_x)
        axbig.set_ylabel(label_y)
        axbig.scatter(hillas_truth_energy, hillas_sky_err, s=90, c='r', marker='+', alpha=0.2)
        axbig.plot(hillas_sky_err_vs_energy.xaxis,hillas_sky_err_vs_energy.yaxis,c='k')
        for x in range(0,len(hillas_sky_err_vs_energy.xaxis)):
            axbig.text(hillas_sky_err_vs_energy.xaxis[x], hillas_sky_err_vs_energy.yaxis[x], '%0.2f'%(hillas_sky_err_vs_energy.yaxis[x]))
        axbig.set_xscale('log')
        #axbig.set_yscale('log')
        #axbig.set_ylim(0, 0.4)
        fig.savefig(f'{ctapipe_output}/output_plots/sky_err_vs_energy_hillas.png',bbox_inches='tight')
        axbig.remove()
    
        fig.clf()
        axbig = fig.add_subplot()
        label_x = 'energy'
        label_y = 'arrival error [deg]'
        axbig.set_xlabel(label_x)
        axbig.set_ylabel(label_y)
        axbig.scatter(temp_truth_energy, temp_fit_sky_err, s=90, c='r', marker='+', alpha=0.2)
        axbig.plot(temp_fit_sky_err_vs_energy.xaxis,temp_fit_sky_err_vs_energy.yaxis,c='k')
        for x in range(0,len(temp_fit_sky_err_vs_energy.xaxis)):
            axbig.text(temp_fit_sky_err_vs_energy.xaxis[x], temp_fit_sky_err_vs_energy.yaxis[x], '%0.2f'%(temp_fit_sky_err_vs_energy.yaxis[x]))
        axbig.set_xscale('log')
        #axbig.set_yscale('log')
        #axbig.set_ylim(0, 0.4)
        fig.savefig(f'{ctapipe_output}/output_plots/sky_err_vs_energy_temp.png',bbox_inches='tight')
        axbig.remove()
    
        fig.clf()
        axbig = fig.add_subplot()
        label_x = 'energy'
        label_y = 'arrival error [deg]'
        axbig.set_xlabel(label_x)
        axbig.set_ylabel(label_y)
        axbig.scatter(comb_truth_energy, comb_fit_sky_err, s=90, c='r', marker='+', alpha=0.2)
        axbig.plot(comb_fit_sky_err_vs_energy.xaxis,comb_fit_sky_err_vs_energy.yaxis,c='k')
        for x in range(0,len(comb_fit_sky_err_vs_energy.xaxis)):
            axbig.text(comb_fit_sky_err_vs_energy.xaxis[x], comb_fit_sky_err_vs_energy.yaxis[x], '%0.2f'%(comb_fit_sky_err_vs_energy.yaxis[x]))
        axbig.set_xscale('log')
        #axbig.set_yscale('log')
        #axbig.set_ylim(0, 0.4)
        fig.savefig(f'{ctapipe_output}/output_plots/sky_err_vs_energy_comb.png',bbox_inches='tight')
        axbig.remove()
    
        fig.clf()
        axbig = fig.add_subplot()
        label_x = 'energy'
        label_y = 'arrival error [deg]'
        axbig.set_xlabel(label_x)
        axbig.set_ylabel(label_y)
        axbig.scatter(line_truth_energy, line_fit_sky_err, s=90, c='r', marker='+', alpha=0.2)
        axbig.plot(line_fit_sky_err_vs_energy.xaxis,line_fit_sky_err_vs_energy.yaxis,c='k')
        for x in range(0,len(line_fit_sky_err_vs_energy.xaxis)):
            axbig.text(line_fit_sky_err_vs_energy.xaxis[x], line_fit_sky_err_vs_energy.yaxis[x], '%0.2f'%(line_fit_sky_err_vs_energy.yaxis[x]))
        axbig.set_xscale('log')
        #axbig.set_yscale('log')
        #axbig.set_ylim(0, 0.4)
        fig.savefig(f'{ctapipe_output}/output_plots/sky_err_vs_energy_line.png',bbox_inches='tight')
        axbig.remove()
    
        fig.clf()
        axbig = fig.add_subplot()
        label_x = 'weighted open angle'
        label_y = 'arrival error [deg]'
        axbig.set_xlabel(label_x)
        axbig.set_ylabel(label_y)
        axbig.set_xscale('log')
        #axbig.set_yscale('log')
        axbig.scatter(hillas_open_angle, hillas_sky_err, s=90, c='r', marker='+', alpha=0.2)
        fig.savefig(f'{ctapipe_output}/output_plots/sky_err_vs_angle_hillas.png',bbox_inches='tight')
        axbig.remove()
    
        fig.clf()
        axbig = fig.add_subplot()
        label_x = 'weighted open angle'
        label_y = 'arrival error [deg]'
        axbig.set_xlabel(label_x)
        axbig.set_ylabel(label_y)
        axbig.set_xscale('log')
        #axbig.set_yscale('log')
        axbig.scatter(temp_open_angle, temp_fit_sky_err, s=90, c='r', marker='+', alpha=0.2)
        fig.savefig(f'{ctapipe_output}/output_plots/sky_err_vs_angle_temp.png',bbox_inches='tight')
        axbig.remove()
    
        fig.clf()
        axbig = fig.add_subplot()
        label_x = 'weighted open angle'
        label_y = 'arrival error [deg]'
        axbig.set_xlabel(label_x)
        axbig.set_ylabel(label_y)
        axbig.set_xscale('log')
        #axbig.set_yscale('log')
        axbig.scatter(line_open_angle, line_fit_sky_err, s=90, c='r', marker='+', alpha=0.2)
        fig.savefig(f'{ctapipe_output}/output_plots/sky_err_vs_angle_line.png',bbox_inches='tight')
        axbig.remove()
    
        fig.clf()
        axbig = fig.add_subplot()
        label_x = 'sum image size'
        label_y = 'arrival error [deg]'
        axbig.set_xlabel(label_x)
        axbig.set_ylabel(label_y)
        axbig.set_xscale('log')
        #axbig.set_yscale('log')
        axbig.scatter(hillas_image_size, hillas_sky_err, s=90, c='r', marker='+', alpha=0.2)
        fig.savefig(f'{ctapipe_output}/output_plots/sky_err_vs_size_hillas.png',bbox_inches='tight')
        axbig.remove()
    
        fig.clf()
        axbig = fig.add_subplot()
        label_x = 'sum image size'
        label_y = 'arrival error [deg]'
        axbig.set_xlabel(label_x)
        axbig.set_ylabel(label_y)
        axbig.set_xscale('log')
        #axbig.set_yscale('log')
        axbig.scatter(temp_image_size, temp_fit_sky_err, s=90, c='r', marker='+', alpha=0.2)
        fig.savefig(f'{ctapipe_output}/output_plots/sky_err_vs_size_temp.png',bbox_inches='tight')
        axbig.remove()
    
        fig.clf()
        axbig = fig.add_subplot()
        label_x = 'sum image size'
        label_y = 'arrival error [deg]'
        axbig.set_xlabel(label_x)
        axbig.set_ylabel(label_y)
        axbig.set_xscale('log')
        #axbig.set_yscale('log')
        axbig.scatter(line_image_size, line_fit_sky_err, s=90, c='r', marker='+', alpha=0.2)
        fig.savefig(f'{ctapipe_output}/output_plots/sky_err_vs_size_line.png',bbox_inches='tight')
        axbig.remove()

        fig.clf()
        axbig = fig.add_subplot()
        label_x = 'fit error'
        label_y = 'arrival error [deg]'
        axbig.set_xlabel(label_x)
        axbig.set_ylabel(label_y)
        #axbig.set_xscale('log')
        #axbig.set_xlim(0, 0.03)
        axbig.scatter(temp_fit_evt_err, temp_fit_sky_err, s=90, c='r', marker='+', alpha=0.2)
        fig.savefig(f'{ctapipe_output}/output_plots/sky_err_vs_fit_err_temp.png',bbox_inches='tight')
        axbig.remove()
    
        fig.clf()
        axbig = fig.add_subplot()
        label_x = 'fit rms'
        label_y = 'arrival error [deg]'
        axbig.set_xlabel(label_x)
        axbig.set_ylabel(label_y)
        #axbig.set_xscale('log')
        #axbig.set_xlim(0, 0.03)
        axbig.scatter(temp_fit_evt_rms, temp_fit_sky_err, s=90, c='r', marker='+', alpha=0.2)
        fig.savefig(f'{ctapipe_output}/output_plots/sky_err_vs_fit_rms_temp.png',bbox_inches='tight')
        axbig.remove()
    
        fig.clf()
        axbig = fig.add_subplot()
        label_x = 'cam x error'
        label_y = 'cam y error'
        axbig.set_xlabel(label_x)
        axbig.set_ylabel(label_y)
        axbig.scatter(line_fit_cam_x_err, line_fit_cam_y_err, s=90, c='r', marker='+', alpha=0.2)
        axbig.set_xlim(-0.5, 0.5)
        axbig.set_ylim(-0.5, 0.5)
        fig.savefig(f'{ctapipe_output}/output_plots/camx_vs_camy_line.png',bbox_inches='tight')
        axbig.remove()
    
        fig.clf()
        axbig = fig.add_subplot()
        label_x = 'cam x error'
        label_y = 'cam y error'
        axbig.set_xlabel(label_x)
        axbig.set_ylabel(label_y)
        axbig.scatter(temp_fit_cam_x_err, temp_fit_cam_y_err, s=90, c='r', marker='+', alpha=0.2)
        axbig.set_xlim(-0.5, 0.5)
        axbig.set_ylim(-0.5, 0.5)
        fig.savefig(f'{ctapipe_output}/output_plots/camx_vs_camy_temp.png',bbox_inches='tight')
        axbig.remove()
    
        fig.clf()
        axbig = fig.add_subplot()
        label_x = 'cam x error'
        label_y = 'cam y error'
        axbig.set_xlabel(label_x)
        axbig.set_ylabel(label_y)
        axbig.scatter(comb_fit_cam_x_err, comb_fit_cam_y_err, s=90, c='r', marker='+', alpha=0.2)
        axbig.set_xlim(-0.5, 0.5)
        axbig.set_ylim(-0.5, 0.5)
        fig.savefig(f'{ctapipe_output}/output_plots/camx_vs_camy_comb.png',bbox_inches='tight')
        axbig.remove()
    

#plot_single_tel_analysis()    
plot_cta_array_analysis()

