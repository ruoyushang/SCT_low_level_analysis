
import os, sys
import subprocess
import glob

import numpy as np
from scipy.optimize import curve_fit
from astropy import units as u
from matplotlib import pyplot as plt
from matplotlib import colors
import pickle
import math

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

#ana_tag = 'SCT_freepact_selection'
#ana_tag = 'SCT_loose_selection'
#ana_tag = 'MIX_loose_selection'

#sim_files = 'sct_onaxis_train.txt'
#sim_files = 'sct_onaxis_test.txt'
#sim_files = 'sct_onaxis_all.txt'
#sim_files = 'sct_diffuse_all.txt'
#sim_files = 'mst_onaxis_train.txt'
#sim_files = 'mst_onaxis_test.txt'
#sim_files = 'mst_onaxis_all.txt'
#sim_files = 'mst_diffuse_all.txt'

ana_tag = 'SCT_loose_onaxis'
#ana_tag = 'SCT_loose_diffuse'
sim_files = None
if 'SCT' in ana_tag:
    if 'onaxis' in ana_tag:
        sim_files = 'sct_onaxis_all.txt'
        #sim_files = 'sct_onaxis_test.txt'
    else:
        sim_files = 'sct_diffuse_all.txt'
else:
    if 'onaxis' in ana_tag:
        sim_files = 'mst_onaxis_all.txt'
    else:
        sim_files = 'mst_diffuse_all.txt'


font = {'family': 'serif', 'color':  'black', 'weight': 'normal', 'size': 10, 'rotation': 0.,}

training_sample_path = []
max_nfiles = 1e10
nfiles = 0

hillas_unc_cut = 1e10

off_angle_cut = 1.0
#off_angle_cut = 4.0
#off_angle_cut = 1e10

hist_edge = 0.5

energy_cut = 0.001
#energy_cut = 1.0

ref_name = []
list_ref_unc_cut = []
#ref_name += ['default ($\sigma<0.8^{\circ}$)']
#list_ref_unc_cut += [0.8]
#ref_name += ['default ($\sigma<0.4^{\circ}$)']
#list_ref_unc_cut += [0.4]
#ref_name += ['default ($\sigma<0.2^{\circ}$)']
#list_ref_unc_cut += [0.2]
#ref_name += ['default ($\sigma<0.1^{\circ}$)']
#list_ref_unc_cut += [0.1]
#ref_name += ['least square ($\sigma<0.8^{\circ}$)']
#list_ref_unc_cut += [0.8]
#ref_name += ['least square ($\sigma<0.4^{\circ}$)']
#list_ref_unc_cut += [0.4]
ref_name += ['least square ($\sigma<0.2^{\circ}$)']
list_ref_unc_cut += [0.2]
#ref_name += ['least square ($\sigma<0.1^{\circ}$)']
#list_ref_unc_cut += [0.1]
#ref_name += ['least square ($\sigma<0.05^{\circ}$)']
#list_ref_unc_cut += [0.05]
#ref_name += ['template result ($\sigma<0.8^{\circ}$)']
#list_ref_unc_cut += [0.8]
#ref_name += ['template result ($\sigma<0.4^{\circ}$)']
#list_ref_unc_cut += [0.4]
#ref_name += ['template result ($\sigma<0.2^{\circ}$)']
#list_ref_unc_cut += [0.2]
#ref_name += ['combined result ($\sigma<0.8^{\circ}$)']
#list_ref_unc_cut += [0.8]
#ref_name += ['combined result ($\sigma<0.4^{\circ}$)']
#list_ref_unc_cut += [0.4]
#ref_name += ['combined result ($\sigma<0.2^{\circ}$)']
#list_ref_unc_cut += [0.2]

hist_truth_norm = Histogram(nbins=(4), ranges=[[-1,1]])
hist_ref_norm = []
hist_ref_efficiency = []
hist_ref_off_angle = []
hist_ref_off_angle_err = []
for ref in range(0,len(ref_name)):
    hist_ref_norm += [Histogram(nbins=(4), ranges=[[-1,1]])]
    hist_ref_efficiency += [Histogram(nbins=(4), ranges=[[-1,1]])]
    hist_ref_off_angle += [Histogram(nbins=(4), ranges=[[-1,1]])]
    hist_ref_off_angle_err += [Histogram(nbins=(4), ranges=[[-1,1]])]


with open(f'{ctapipe_input}/{sim_files}', 'r') as file:
    for line in file:
        #training_sample_path += [get_dataset_path(line.strip('\n'))]
        training_sample_path += [line.strip('\n')]
        nfiles += 1
        if nfiles >= max_nfiles: break

def gauss_func(x,A,sigma):
    return A * np.exp(-((x-0.)**2)/(2*sigma*sigma))

def plot_analysis_result():

    list_all_truth_log_energy = []
    list_ref_truth_log_energy = []
    list_ref_truth_log_energy_pass = []
    list_ref_truth_log_energy_fail = []
    list_ref_unc = []
    list_ref_unc_pass = []
    list_ref_unc_fail = []
    list_ref_off_angle = []
    list_ref_off_angle_pass = []
    list_ref_off_angle_fail = []
    list_ref_xing_outlier = []
    for ref in range(0,len(ref_name)):
        list_ref_truth_log_energy += [[]]
        list_ref_truth_log_energy_pass += [[]]
        list_ref_truth_log_energy_fail += [[]]
        list_ref_unc += [[]]
        list_ref_unc_pass += [[]]
        list_ref_unc_fail += [[]]
        list_ref_off_angle += [[]]
        list_ref_off_angle_pass += [[]]
        list_ref_off_angle_fail += [[]]
        list_ref_xing_outlier += [[]]

    for path in range(0,len(training_sample_path)):
    
        run_id = training_sample_path[path].split("_")[3].strip("run")
        print (f"run_id = {run_id}")
    
        input_filename = f'{ctapipe_output}/output_analysis/{ana_tag}_run{run_id}.pkl'
        print (f'loading pickle analysis data: {input_filename}')
        if not os.path.exists(input_filename):
            print (f'file does not exist.')
            continue
        analysis_result = pickle.load(open(input_filename, "rb"))

        hillas_result = analysis_result[0]
        xing_result = analysis_result[1]
        template_result = analysis_result[2]
        combine_result = analysis_result[3]
        truth_result = analysis_result[4]

        for evt in range(0,len(truth_result)):
            truth_energy = truth_result[evt][0].value
            list_all_truth_log_energy += [np.log10(truth_energy)]

        for ref in range(0,len(ref_name)):
            if 'default' in ref_name[ref]:
                for evt in range(0,len(hillas_result)):
                    truth_energy = hillas_result[evt][0].value
                    if truth_energy<energy_cut: continue
                    off_angle = hillas_result[evt][2]
                    if math.isnan(off_angle): continue
                    if off_angle>off_angle_cut: continue
                    unc = hillas_result[evt][3]
                    if unc==0.:
                        unc = 4.0
                    list_ref_off_angle[ref] += [off_angle*off_angle]
                    list_ref_unc[ref] += [unc]
                    list_ref_truth_log_energy[ref] += [np.log10(truth_energy)]
                    if unc<list_ref_unc_cut[ref]:
                        list_ref_off_angle_pass[ref] += [off_angle*off_angle]
                        list_ref_unc_pass[ref] += [unc]
                        list_ref_truth_log_energy_pass[ref] += [np.log10(truth_energy)]
                    else:
                        list_ref_off_angle_fail[ref] += [off_angle*off_angle]
                        list_ref_unc_fail[ref] += [unc]
                        list_ref_truth_log_energy_fail[ref] += [np.log10(truth_energy)]
            if 'least square' in ref_name[ref]:
                for evt in range(0,len(xing_result)):
                    truth_energy = xing_result[evt][0].value
                    if truth_energy<energy_cut: continue
                    off_angle = xing_result[evt][2]
                    if math.isnan(off_angle): continue
                    if off_angle>off_angle_cut: continue
                    unc = xing_result[evt][3]
                    if unc==0.:
                        unc = 0.00001
                    list_ref_off_angle[ref] += [off_angle*off_angle]
                    list_ref_unc[ref] += [unc]
                    list_ref_truth_log_energy[ref] += [np.log10(truth_energy)]
                    if unc<list_ref_unc_cut[ref]:
                        list_ref_off_angle_pass[ref] += [off_angle*off_angle]
                        list_ref_unc_pass[ref] += [unc]
                        list_ref_truth_log_energy_pass[ref] += [np.log10(truth_energy)]
                    else:
                        list_ref_off_angle_fail[ref] += [off_angle*off_angle]
                        list_ref_unc_fail[ref] += [unc]
                        list_ref_truth_log_energy_fail[ref] += [np.log10(truth_energy)]
                    run_id = xing_result[evt][6]
                    event_id = xing_result[evt][7]
                    if off_angle/unc>4.:
                        list_ref_xing_outlier[ref] += [[run_id,event_id]]
            if 'template' in ref_name[ref]:
                for evt in range(0,len(template_result)):
                    truth_energy = template_result[evt][0].value
                    if truth_energy<energy_cut: continue
                    off_angle = template_result[evt][1]
                    if math.isnan(off_angle): continue
                    if off_angle>off_angle_cut: continue
                    unc = template_result[evt][2]
                    if unc==0.:
                        unc = 0.00001
                    list_ref_off_angle[ref] += [off_angle*off_angle]
                    list_ref_unc[ref] += [unc]
                    list_ref_truth_log_energy[ref] += [np.log10(truth_energy)]
                    if unc<list_ref_unc_cut[ref]:
                        list_ref_off_angle_pass[ref] += [off_angle*off_angle]
                        list_ref_unc_pass[ref] += [unc]
                        list_ref_truth_log_energy_pass[ref] += [np.log10(truth_energy)]
                    else:
                        list_ref_off_angle_fail[ref] += [off_angle*off_angle]
                        list_ref_unc_fail[ref] += [unc]
                        list_ref_truth_log_energy_fail[ref] += [np.log10(truth_energy)]
            if 'combined' in ref_name[ref]:
                for evt in range(0,len(combine_result)):
                    truth_energy = combine_result[evt][0].value
                    if truth_energy<energy_cut: continue
                    off_angle = combine_result[evt][1]
                    if math.isnan(off_angle): continue
                    if off_angle>off_angle_cut: continue
                    unc = combine_result[evt][2]
                    if unc==0.:
                        unc = 0.00001
                    list_ref_off_angle[ref] += [off_angle*off_angle]
                    list_ref_unc[ref] += [unc]
                    list_ref_truth_log_energy[ref] += [np.log10(truth_energy)]
                    if unc<list_ref_unc_cut[ref]:
                        list_ref_off_angle_pass[ref] += [off_angle*off_angle]
                        list_ref_unc_pass[ref] += [unc]
                        list_ref_truth_log_energy_pass[ref] += [np.log10(truth_energy)]
                    else:
                        list_ref_off_angle_fail[ref] += [off_angle*off_angle]
                        list_ref_unc_fail[ref] += [unc]
                        list_ref_truth_log_energy_fail[ref] += [np.log10(truth_energy)]

    

    hist_truth_norm.fill(list_all_truth_log_energy)

    for ref in range(0,len(ref_name)):
        hist_ref_norm[ref].fill(list_ref_truth_log_energy_pass[ref])
        hist_ref_off_angle[ref].fill(list_ref_truth_log_energy_pass[ref],weights=list_ref_off_angle_pass[ref])
        hist_ref_off_angle[ref].data = np.sqrt(hist_ref_off_angle[ref].data / hist_ref_norm[ref].data)
        hist_ref_off_angle_err[ref].data =  hist_ref_off_angle[ref].data / np.sqrt(hist_ref_norm[ref].data)
        hist_ref_efficiency[ref].data = hist_ref_norm[ref].data / hist_truth_norm.data

    for ref in range(0,len(ref_name)):
        print (f"list_ref_xing_outlier = {list_ref_xing_outlier[ref]}")

    fig, ax = plt.subplots()
    figsize_x = 6.4
    figsize_y = 4.6
    fig.set_figheight(figsize_y)
    fig.set_figwidth(figsize_x)
    label_x = "log10 energy [TeV]"
    label_y = "reconstruction efficiency"
    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)
    for ref in range(0,len(ref_name)):
        ax.plot(hist_ref_efficiency[ref].bin_centers(0),hist_ref_efficiency[ref].data,label=ref_name[ref])
    ax.legend(loc='best')
    ax.set_yscale('log')
    fig.savefig(
        f"{ctapipe_output}/output_plots/analysis_efficiency_{ana_tag}.png",
        bbox_inches="tight",
    )
    del fig
    del ax
    plt.close()

    fig, ax = plt.subplots()
    figsize_x = 6.4
    figsize_y = 4.6
    fig.set_figheight(figsize_y)
    fig.set_figwidth(figsize_x)
    label_x = "log10 energy [TeV]"
    label_y = "68% containment [deg]"
    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)
    for ref in range(0,len(ref_name)):
        ax.errorbar(hist_ref_norm[ref].bin_centers(0),hist_ref_off_angle[ref].data,hist_ref_off_angle_err[ref].data,label=ref_name[ref])
    ax.legend(loc='best')
    ax.set_yscale('log')
    fig.savefig(
        f"{ctapipe_output}/output_plots/analysis_off_angle_{ana_tag}.png",
        bbox_inches="tight",
    )
    del fig
    del ax
    plt.close()


    for ref in range(0,len(ref_name)):
        ref_mean = np.sqrt(np.mean(np.array(list_ref_off_angle_pass[ref])))
        print ("============================================")
        print (f"ref_name = {ref_name[ref]}")
        print (f"ref_mean = {ref_mean}")


    fig, ax = plt.subplots()
    figsize_x = 6.4
    figsize_y = 4.6
    fig.set_figheight(figsize_y)
    fig.set_figwidth(figsize_x)
    label_x = "angular distance [deg]"
    label_y = "count"
    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)
    for ref in range(0,len(ref_name)):
        list_with_limit = []
        for entry in range(0,len(list_ref_off_angle[ref])):
            list_with_limit += [min(np.sqrt(list_ref_off_angle[ref][entry]),hist_edge-0.001)]
        ax.hist(list_with_limit,histtype='step',bins=25,range=(0.,hist_edge),label=ref_name[ref].split('(')[0])
    ax.legend(loc='best')
    fig.savefig(
        f"{ctapipe_output}/output_plots/reconstruction_off_angle_incl_{ana_tag}.png",
        bbox_inches="tight",
    )
    del fig
    del ax
    plt.close()

    fig, ax = plt.subplots()
    figsize_x = 6.4
    figsize_y = 4.6
    fig.set_figheight(figsize_y)
    fig.set_figwidth(figsize_x)
    label_x = "angular distance [deg]"
    label_y = "count"
    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)
    for ref in range(0,len(ref_name)):
        list_with_limit = []
        for entry in range(0,len(list_ref_off_angle_pass[ref])):
            list_with_limit += [min(np.sqrt(list_ref_off_angle_pass[ref][entry]),hist_edge-0.001)]
        ax.hist(list_with_limit,histtype='step',bins=25,range=(0.,hist_edge),label=ref_name[ref].split('(')[0])
    ax.legend(loc='best')
    fig.savefig(
        f"{ctapipe_output}/output_plots/reconstruction_off_angle_pass_{ana_tag}.png",
        bbox_inches="tight",
    )
    del fig
    del ax
    plt.close()

    fig, ax = plt.subplots()
    figsize_x = 6.4
    figsize_y = 4.6
    fig.set_figheight(figsize_y)
    fig.set_figwidth(figsize_x)
    label_x = "angular distance [deg]"
    label_y = "count"
    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)
    for ref in range(0,len(ref_name)):
        list_with_limit = []
        for entry in range(0,len(list_ref_off_angle_fail[ref])):
            list_with_limit += [min(np.sqrt(list_ref_off_angle_fail[ref][entry]),hist_edge-0.001)]
        ax.hist(list_with_limit,histtype='step',bins=25,range=(0.,hist_edge),label=ref_name[ref].split('(')[0])
    ax.legend(loc='best')
    fig.savefig(
        f"{ctapipe_output}/output_plots/reconstruction_off_angle_fail_{ana_tag}.png",
        bbox_inches="tight",
    )
    del fig
    del ax
    plt.close()


    for ref in range(0,len(ref_name)):
        fig, ax = plt.subplots()
        figsize_x = 6.4
        figsize_y = 6.4
        fig.set_figheight(figsize_y)
        fig.set_figwidth(figsize_x)
        label_x = "off angle [deg]"
        label_y = "uncertainty [deg]"
        ax.set_xlabel(label_x)
        ax.set_ylabel(label_y)
        #max_unc = np.max(list_ref_unc_pass[ref])
        #min_unc = np.min(list_ref_unc_pass[ref])
        max_unc = list_ref_unc_cut[ref]
        min_unc = 0.
        ax.scatter(
            np.sqrt(list_ref_off_angle_pass[ref]),
            list_ref_unc_pass[ref],
            s=90,
            facecolors="none",
            c="r",
            alpha=0.2,
            marker="+",
            label=ref_name[ref],
        )
        n_intervals = 5
        for u in range(0,n_intervals):
            unc_axis = []
            off_angle_mean = []
            new_list_off_angle = []
            delta_unc = list_ref_unc_cut[ref]/float(n_intervals)
            lower_unc = float(u)*delta_unc + min_unc
            upper_unc = float(u+1)*delta_unc + min_unc
            for evt in range(0,len(list_ref_unc_pass[ref])):
                unc = list_ref_unc_pass[ref][evt]
                off_angle = list_ref_off_angle_pass[ref][evt]
                if unc < lower_unc or unc > upper_unc: continue
                new_list_off_angle += [off_angle]
            mean = np.sqrt(np.mean(new_list_off_angle))
            unc_axis += [0.5*(lower_unc+upper_unc)]
            off_angle_mean += [0.]
            unc_axis += [0.5*(lower_unc+upper_unc)]
            off_angle_mean += [1.4*mean]
            ax.plot(off_angle_mean,unc_axis,color='k',marker='|')
        ax.set_xlim(0., max_unc)
        ax.set_ylim(0., max_unc)
        ax.legend(loc='best')
        fig.savefig(
            f"{ctapipe_output}/output_plots/off_angle_vs_unc_ref{ref}_{ana_tag}.png",
            bbox_inches="tight",
        )
        del fig
        del ax
        plt.close()

plot_analysis_result()
