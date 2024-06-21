
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

ana_tag = 'movie_box3d'
#ana_tag = 'image_box3d_fast'

telescope_type = 'MST_SCT_SCTCam'

sim_files = 'sim_files.txt'
#sim_files = 'sim_files_diffuse_gamma.txt'

font = {'family': 'serif', 'color':  'black', 'weight': 'normal', 'size': 10, 'rotation': 0.,}

training_sample_path = []
particle_type = []
max_nfiles = 1e10
nfiles = 0

image_dir_cut = 0.7
fit_chi2_cut = 1.2

def pass_quality(lightcone,image_direction,fit_chi2,image_size):

    #return True

    if abs(image_direction)<image_dir_cut: 
        return False

    #return True

    if fit_chi2/image_size>fit_chi2_cut:
        return False

    return True

with open(f'{ctapipe_input}/{sim_files}', 'r') as file:
    for line in file:
        training_sample_path += [get_dataset_path(line.strip('\n'))]
        particle_type += [0]
        nfiles += 1
        if nfiles >= max_nfiles: break

def gauss_func(x,A,sigma):
    return A * np.exp(-((x-0.)**2)/(2*sigma*sigma))

def plot_monotel_analysis():

    list_image_size = []
    list_fit_chi2 = []
    list_image_direction = []
    list_time_direction = []
    list_angle_err = []
    list_bad_fit_chi2 = []
    list_bad_image_size = []
    list_bad_image_direction = []
    list_lightcone = []
    list_truth_projection = []
    list_delta_energy = []
    list_truth_energy = []
    list_model_energy = []
    list_delta_arrival = []
    list_delta_camx = []
    list_delta_camy = []
    list_delta_camr = []
    list_delta_camr_weight = []

    total_images = 0
    pass_images = 0

    for path in range(0,len(training_sample_path)):
    
        source = SimTelEventSource(training_sample_path[path], focal_length_choice='EQUIVALENT')
        subarray = source.subarray
        ob_keys = source.observation_blocks.keys()
        run_id = list(ob_keys)[0]
    
        input_filename = f'{ctapipe_output}/output_analysis/{ana_tag}_run{run_id}_{telescope_type}.pkl'
        print (f'loading pickle analysis data: {input_filename}')
        if not os.path.exists(input_filename):
            print (f'file does not exist.')
            continue
        analysis_result = pickle.load(open(input_filename, "rb"))
    
        for img in range(0,len(analysis_result)):

            img_header = analysis_result[img][0]
            img_geometry = analysis_result[img][1]
            img_truth = analysis_result[img][2]
            img_model = analysis_result[img][3]

            image_size = img_geometry[0]
            lightcone = img_geometry[1]
            focal_length = img_geometry[2]
            image_direction = img_geometry[3]
            time_direction = img_geometry[4]
            angle_err = img_geometry[5]
            truth_energy = img_truth[0]
            truth_alt = img_truth[1]
            truth_az = img_truth[2]
            truth_camx = img_truth[3]/focal_length*180./np.pi
            truth_camy = img_truth[4]/focal_length*180./np.pi
            truth_projection = img_truth[5]
            fit_energy = img_model[0]
            fit_alt = img_model[1]
            fit_az = img_model[2]
            fit_camx = img_model[3]/focal_length*180./np.pi
            fit_camy = img_model[4]/focal_length*180./np.pi
            fit_chi2 = img_model[5]

            total_images += 1

            #if image_size<200.: continue
            if not pass_quality(lightcone,image_direction,fit_chi2,image_size): continue
            #if truth_projection<0.: continue

            pass_images += 1

            delta_energy = abs(fit_energy - truth_energy) / truth_energy
            delta_alt = (fit_alt - truth_alt)*180./np.pi
            delta_az = (fit_az - truth_az)*180./np.pi
            if delta_az>180.:
                delta_az = delta_az - 360.
            if delta_az<-180.:
                delta_az = delta_az + 360.

            delta_camx = float((fit_camx-truth_camx))
            delta_camy = float((fit_camy-truth_camy))

            delta_camr = pow(delta_camx*delta_camx+delta_camy*delta_camy,0.5)
            if delta_camr>4.0 and image_size>5000.:
            #if delta_camr<0.05 and image_size>5000. and lightcone>0.:
                print (f'file {img_header[0]}, evt_id {img_header[1]}, tel_id {img_header[2]}')
                print (f'delta_camr = {delta_camr:0.2f}, image_size = {image_size:0.1f}, lightcone = {lightcone:0.2f}, image_direction = {image_direction:0.2f}')


            list_image_size += [image_size]
            list_fit_chi2 += [fit_chi2/image_size]
            list_image_direction += [abs(image_direction)]
            list_time_direction += [abs(time_direction)]
            list_angle_err += [abs(angle_err)]
            list_lightcone += [lightcone]
            list_truth_projection += [truth_projection]
            list_delta_arrival += [pow(delta_alt*delta_alt+delta_az*delta_az,0.5)]
            list_delta_camx += [delta_camx]
            list_delta_camy += [delta_camy]
            list_delta_camr += [pow(delta_camx*delta_camx+delta_camy*delta_camy,0.5)]
            list_delta_camr_weight += [1./pow(delta_camx*delta_camx+delta_camy*delta_camy,0.5)]

            if delta_camr>0.5:
                list_bad_fit_chi2 += [fit_chi2/image_size]
                list_bad_image_size += [image_size]
                list_bad_image_direction += [abs(image_direction)]

            list_truth_energy += [np.log10(truth_energy)]
            list_model_energy += [np.log10(fit_energy)]
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
    label_x = 'log10 truth Energy [TeV]'
    label_y = 'log10 model Energy [TeV]'
    axbig.set_xlabel(label_x)
    axbig.set_ylabel(label_y)
    axbig.scatter(list_truth_energy, list_model_energy, s=90, c='b', marker='+', alpha=0.1)
    fig.savefig(f'{ctapipe_output}/output_plots/truth_vs_model_log_energy_{ana_tag}.png',bbox_inches='tight')
    axbig.remove()

    fig.clf()
    axbig = fig.add_subplot()
    label_x = 'Fit chi square'
    label_y = 'Cam delta r'
    axbig.set_xlabel(label_x)
    axbig.set_ylabel(label_y)
    axbig.scatter(list_fit_chi2, list_delta_camr, s=90, c='b', marker='+', alpha=0.3)
    axbig.axvline(x=fit_chi2_cut)
    fig.savefig(f'{ctapipe_output}/output_plots/fit_chi2_vs_camr_{ana_tag}.png',bbox_inches='tight')
    axbig.remove()

    fig.clf()
    axbig = fig.add_subplot()
    label_x = 'Image direction'
    label_y = 'Fit chi square'
    axbig.set_xlabel(label_x)
    axbig.set_ylabel(label_y)
    axbig.scatter(list_image_direction, list_fit_chi2, s=90, c='b', marker='+', alpha=0.3)
    axbig.scatter(list_bad_image_direction, list_bad_fit_chi2, s=90, c='r', marker='+', alpha=0.3)
    axbig.axvline(x=image_dir_cut)
    axbig.axhline(y=fit_chi2_cut)
    axbig.set_xscale('log')
    axbig.set_yscale('log')
    fig.savefig(f'{ctapipe_output}/output_plots/image_dir_vs_chi2_{ana_tag}.png',bbox_inches='tight')
    axbig.remove()

    fig.clf()
    axbig = fig.add_subplot()
    label_x = 'Image direction'
    label_y = 'Image size'
    axbig.set_xlabel(label_x)
    axbig.set_ylabel(label_y)
    axbig.scatter(list_image_direction, list_image_size, s=90, c='b', marker='+', alpha=0.3)
    axbig.scatter(list_bad_image_direction, list_bad_image_size, s=90, c='r', marker='+', alpha=0.3)
    axbig.axvline(x=image_dir_cut)
    axbig.set_xscale('log')
    axbig.set_yscale('log')
    fig.savefig(f'{ctapipe_output}/output_plots/image_dir_vs_size_{ana_tag}.png',bbox_inches='tight')
    axbig.remove()


    fig.clf()
    axbig = fig.add_subplot()
    label_x = 'Lightcone'
    label_y = 'Cam delta r'
    axbig.set_xlabel(label_x)
    axbig.set_ylabel(label_y)
    axbig.scatter(list_lightcone, list_delta_camr, s=90, c='b', marker='+', alpha=0.3)
    fig.savefig(f'{ctapipe_output}/output_plots/lightcone_vs_camr_{ana_tag}.png',bbox_inches='tight')
    axbig.remove()

    fig.clf()
    axbig = fig.add_subplot()
    label_x = 'Image direction'
    label_y = 'Truth projection'
    axbig.set_xlabel(label_x)
    axbig.set_ylabel(label_y)
    axbig.scatter(list_image_direction, list_truth_projection, s=90, c='b', marker='+', alpha=0.3)
    axbig.set_xscale('log')
    fig.savefig(f'{ctapipe_output}/output_plots/image_dir_vs_projection_{ana_tag}.png',bbox_inches='tight')
    axbig.remove()

    fig.clf()
    axbig = fig.add_subplot()
    label_x = 'Image direction'
    label_y = 'Cam delta r'
    axbig.set_xlabel(label_x)
    axbig.set_ylabel(label_y)
    axbig.scatter(list_image_direction, list_delta_camr, s=90, c='b', marker='+', alpha=0.3)
    axbig.axvline(x=image_dir_cut)
    axbig.set_xscale('log')
    fig.savefig(f'{ctapipe_output}/output_plots/image_dir_vs_camr_{ana_tag}.png',bbox_inches='tight')
    axbig.remove()

    fig.clf()
    axbig = fig.add_subplot()
    label_x = 'Time direction'
    label_y = 'Cam delta r'
    axbig.set_xlabel(label_x)
    axbig.set_ylabel(label_y)
    axbig.scatter(list_time_direction, list_delta_camr, s=90, c='b', marker='+', alpha=0.3)
    axbig.set_xscale('log')
    fig.savefig(f'{ctapipe_output}/output_plots/time_dir_vs_camr_{ana_tag}.png',bbox_inches='tight')
    axbig.remove()

    fig.clf()
    axbig = fig.add_subplot()
    label_x = 'Angle error'
    label_y = 'Cam delta r'
    axbig.set_xlabel(label_x)
    axbig.set_ylabel(label_y)
    axbig.scatter(list_angle_err, list_delta_camr, s=90, c='b', marker='+', alpha=0.3)
    axbig.set_xscale('log')
    fig.savefig(f'{ctapipe_output}/output_plots/angle_err_vs_camr_{ana_tag}.png',bbox_inches='tight')
    axbig.remove()

    fig.clf()
    axbig = fig.add_subplot()
    label_x = 'log10 truth energy [TeV]'
    label_y = 'relative error'
    axbig.set_xlabel(label_x)
    axbig.set_ylabel(label_y)
    axbig.plot(hist_delta_energy.bin_centers(0), hist_delta_energy.hist)
    fig.savefig(f'{ctapipe_output}/output_plots/monotel_energy_error_{ana_tag}.png',bbox_inches='tight')
    axbig.remove()

    fig.clf()
    axbig = fig.add_subplot()
    label_x = 'Cam X error [deg]'
    label_y = 'count'
    axbig.set_xlabel(label_x)
    axbig.set_ylabel(label_y)
    axbig.plot(hist_delta_camx.bin_centers(0), hist_delta_camx.hist)
    fig.savefig(f'{ctapipe_output}/output_plots/monotel_camx_error_{ana_tag}.png',bbox_inches='tight')
    axbig.remove()

    fig.clf()
    axbig = fig.add_subplot()
    label_x = 'Cam Y error [deg]'
    label_y = 'count'
    axbig.set_xlabel(label_x)
    axbig.set_ylabel(label_y)
    axbig.plot(hist_delta_camy.bin_centers(0), hist_delta_camy.hist)
    fig.savefig(f'{ctapipe_output}/output_plots/monotel_camy_error_{ana_tag}.png',bbox_inches='tight')
    axbig.remove()

    fig.clf()
    axbig = fig.add_subplot()
    label_x = 'Cam R error [deg]'
    label_y = 'Surface brightness'
    axbig.set_xlabel(label_x)
    axbig.set_ylabel(label_y)
    axbig.plot(hist_delta_camr.bin_centers(0), hist_delta_camr.hist)
    fig.savefig(f'{ctapipe_output}/output_plots/monotel_camr_error_{ana_tag}.png',bbox_inches='tight')
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
        low_energy = pow(10.,hist_delta_energy.bin_lower_edges[0][e])
        high_energy = pow(10.,hist_delta_energy.bin_lower_edges[0][e+1])
        print (f'E = {low_energy:0.2f}-{high_energy:0.2f} TeV, gaussian radius = %0.3f +/- %0.3f deg'%(popt[1],pow(pcov[1][1],0.5)))
        delta_camr_per_energy += [popt[1]]

    fig.clf()
    axbig = fig.add_subplot()
    label_x = 'log10 truth energy [TeV]'
    label_y = 'Gaussian radius'
    axbig.set_xlabel(label_x)
    axbig.set_ylabel(label_y)
    axbig.plot(hist_delta_energy.bin_centers(0), delta_camr_per_energy)
    fig.savefig(f'{ctapipe_output}/output_plots/monotel_camr_per_energy_error_{ana_tag}.png',bbox_inches='tight')
    axbig.remove()

    print (f'total_images = {total_images} / pass_images = {pass_images}')

plot_monotel_analysis()
