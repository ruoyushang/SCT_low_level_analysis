
import os, sys
import subprocess
import glob

import numpy as np
from scipy.optimize import curve_fit
from astropy import units as u
from matplotlib import pyplot as plt
from matplotlib import colors
import pickle

from ctapipe.utils import Histogram
from ctapipe.utils.datasets import get_dataset_path
from ctapipe.io import EventSource, SimTelEventSource, HDF5TableWriter

ctapipe_output = os.environ.get("CTAPIPE_OUTPUT_PATH")
ctapipe_input = os.environ.get("CTAPIPE_SVC_PATH")

fig, ax = plt.subplots()
figsize_x = 8.6
figsize_y = 6.4
fig.set_figheight(figsize_y)
fig.set_figwidth(figsize_x)

#ana_tag = 'image'
ana_tag = 'movie'

font = {'family': 'serif', 'color':  'black', 'weight': 'normal', 'size': 10, 'rotation': 0.,}

training_sample_path = []
particle_type = []
max_nfiles = 1e10
nfiles = 0
with open('%s/sim_files.txt'%(ctapipe_input), 'r') as file:
    for line in file:
        training_sample_path += [get_dataset_path(line.strip('\n'))]
        particle_type += [0]
        nfiles += 1
        if nfiles >= max_nfiles: break

def gauss_func(x,A,sigma):
    return A * np.exp(-((x-0.)**2)/(2*sigma*sigma))

def plot_monotel_analysis():

    list_log10_image_size = []
    list_log10_lightcone = []
    list_delta_energy = []
    list_truth_energy = []
    list_delta_arrival = []
    list_delta_camx = []
    list_delta_camy = []
    list_delta_camr = []
    list_delta_camr_weight = []

    for path in range(0,len(training_sample_path)):
    
        source = SimTelEventSource(training_sample_path[path], focal_length_choice='EQUIVALENT')
        subarray = source.subarray
        ob_keys = source.observation_blocks.keys()
        run_id = list(ob_keys)[0]
    
        input_filename = f'{ctapipe_output}/output_analysis/{ana_tag}_run{run_id}.pkl'
        print (f'loading pickle analysis data: {input_filename}')
        if not os.path.exists(input_filename):
            print (f'file does not exist.')
            continue
        analysis_result = pickle.load(open(input_filename, "rb"))
    
        for img in range(0,len(analysis_result)):

            image_size = analysis_result[img][0]
            lightcone = analysis_result[img][1]
            truth_energy = analysis_result[img][2]
            fit_energy = analysis_result[img][3]
            truth_alt = analysis_result[img][4]
            truth_az = analysis_result[img][5]
            fit_alt = analysis_result[img][6]
            fit_az = analysis_result[img][7]
            truth_camx = analysis_result[img][8]
            truth_camy = analysis_result[img][9]
            fit_camx = analysis_result[img][10]
            fit_camy = analysis_result[img][11]
            focal_length = float(analysis_result[img][12])

            #if image_size<200.: continue
            #if abs(lightcone)<1.: continue
            if abs(lightcone)<2.: continue
            #if abs(lightcone)<5.: continue
            #if abs(lightcone)<10.: continue

            delta_energy = abs(fit_energy - truth_energy) / truth_energy
            delta_alt = (fit_alt - truth_alt)*180./np.pi
            delta_az = (fit_az - truth_az)*180./np.pi
            if delta_az>180.:
                delta_az = delta_az - 360.
            if delta_az<-180.:
                delta_az = delta_az + 360.

            delta_camx = float((fit_camx-truth_camx)/focal_length*180./np.pi)
            delta_camy = float((fit_camy-truth_camy)/focal_length*180./np.pi)

            list_log10_image_size += [np.log10(image_size)]
            list_log10_lightcone += [lightcone]
            list_delta_arrival += [pow(delta_alt*delta_alt+delta_az*delta_az,0.5)]
            list_delta_camx += [delta_camx]
            list_delta_camy += [delta_camy]
            list_delta_camr += [pow(delta_camx*delta_camx+delta_camy*delta_camy,0.5)]
            list_delta_camr_weight += [1./pow(delta_camx*delta_camx+delta_camy*delta_camy,0.5)]

            list_truth_energy += [np.log10(truth_energy)]
            list_delta_energy += [delta_energy]

    hist_delta_camx = Histogram(nbins=(80), ranges=[[-4,4]])
    hist_delta_camx.fill(list_delta_camx)
    hist_delta_camy = Histogram(nbins=(80), ranges=[[-4,4]])
    hist_delta_camy.fill(list_delta_camy)
    hist_delta_camr = Histogram(nbins=(20), ranges=[[0,1]])
    hist_delta_camr.fill(list_delta_camr,weights=list_delta_camr_weight)

    hist_truth_energy = Histogram(nbins=(6), ranges=[[-1,2]])
    hist_truth_energy.fill(list_truth_energy)
    hist_delta_energy = Histogram(nbins=(6), ranges=[[-1,2]])
    hist_delta_energy.fill(list_truth_energy,weights=list_delta_energy)
    for e in range(0,len(hist_delta_energy.hist)):
        weighted_count = hist_delta_energy.hist[e]
        count = hist_truth_energy.hist[e]
        profile = 0.
        if count>0.:
            profile = weighted_count/count
        hist_delta_energy.hist[e] = profile

    init_A = 1000.
    init_sigma = 0.1
    start = (init_A,init_sigma)
    popt, pcov = curve_fit(gauss_func,hist_delta_camr.bin_centers(0),hist_delta_camr.hist,p0=start,bounds=((0, 0.),(np.inf, np.inf)))
    profile_fit = gauss_func(hist_delta_camr.bin_centers(0), *popt)
    residual = hist_delta_camr.hist - profile_fit
    print ('gaussian radius = %0.3f +/- %0.3f deg'%(popt[1],pow(pcov[1][1],0.5)))

    fig.clf()
    axbig = fig.add_subplot()
    label_x = 'log10 truth energy [TeV]'
    label_y = 'relative error'
    axbig.set_xlabel(label_x)
    axbig.set_ylabel(label_y)
    axbig.plot(hist_delta_energy.bin_centers(0), hist_delta_energy.hist)
    fig.savefig(f'{ctapipe_output}/output_plots/monotel_energy_error.png',bbox_inches='tight')
    axbig.remove()

    fig.clf()
    axbig = fig.add_subplot()
    label_x = 'Cam X error [deg]'
    label_y = 'count'
    axbig.set_xlabel(label_x)
    axbig.set_ylabel(label_y)
    axbig.plot(hist_delta_camx.bin_centers(0), hist_delta_camx.hist)
    fig.savefig(f'{ctapipe_output}/output_plots/monotel_camx_error.png',bbox_inches='tight')
    axbig.remove()

    fig.clf()
    axbig = fig.add_subplot()
    label_x = 'Cam Y error [deg]'
    label_y = 'count'
    axbig.set_xlabel(label_x)
    axbig.set_ylabel(label_y)
    axbig.plot(hist_delta_camy.bin_centers(0), hist_delta_camy.hist)
    fig.savefig(f'{ctapipe_output}/output_plots/monotel_camy_error.png',bbox_inches='tight')
    axbig.remove()

    fig.clf()
    axbig = fig.add_subplot()
    label_x = 'Cam R error [deg]'
    label_y = 'Surface brightness'
    axbig.set_xlabel(label_x)
    axbig.set_ylabel(label_y)
    axbig.plot(hist_delta_camr.bin_centers(0), hist_delta_camr.hist)
    fig.savefig(f'{ctapipe_output}/output_plots/monotel_camr_error.png',bbox_inches='tight')
    axbig.remove()

    delta_camr_per_energy = []
    for e in range(0,len(hist_delta_energy.bin_lower_edges[0])-1):
        energy_lo = hist_delta_energy.bin_lower_edges[0][e]
        energy_up = hist_delta_energy.bin_lower_edges[0][e+1]
        new_list_delta_camr = []
        new_list_delta_camr_weight = []
        for entry in range(0,len(list_truth_energy)):
            truth_energy = list_truth_energy[entry]
            if truth_energy<energy_lo: continue
            if truth_energy>energy_up: continue
            new_list_delta_camr += [list_delta_camr[entry]]
            new_list_delta_camr_weight += [list_delta_camr_weight[entry]]
        hist_delta_camr = Histogram(nbins=(20), ranges=[[0,1]])
        hist_delta_camr.fill(new_list_delta_camr,weights=new_list_delta_camr_weight)
        init_A = 1000.
        init_sigma = 0.1
        start = (init_A,init_sigma)
        popt, pcov = curve_fit(gauss_func,hist_delta_camr.bin_centers(0),hist_delta_camr.hist,p0=start,bounds=((0, 0.),(np.inf, np.inf)))
        profile_fit = gauss_func(hist_delta_camr.bin_centers(0), *popt)
        residual = hist_delta_camr.hist - profile_fit
        print ('E[e], gaussian radius = %0.3f +/- %0.3f deg'%(popt[1],pow(pcov[1][1],0.5)))
        delta_camr_per_energy += [popt[1]]

    fig.clf()
    axbig = fig.add_subplot()
    label_x = 'log10 truth energy [TeV]'
    label_y = 'Gaussian radius'
    axbig.set_xlabel(label_x)
    axbig.set_ylabel(label_y)
    axbig.plot(hist_delta_energy.bin_centers(0), delta_camr_per_energy)
    fig.savefig(f'{ctapipe_output}/output_plots/monotel_camr_per_energy_error.png',bbox_inches='tight')
    axbig.remove()

plot_monotel_analysis()
