
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


energy_cut = 0.001
#energy_cut = 1.0

evt_selection = 'loose'
#evt_selection = 'freepact'

#array_type = 'LST_Nectar_ASTRI'
array_type = 'SCT'
#array_type = 'Nectar'
#array_type = 'Flash'
#array_type = 'LST'
#array_type = 'ASTRI'
#array_type = 'CHEC'
#array_type = 'DigiCam'
#array_type = 'MIX'

pointing = 'onaxis'
#pointing = 'diffuse'



#ana_container = []
#plot_name = 'default_vs_ls'
#pointing = 'onaxis'
##pointing = 'diffuse'
##array_type = 'SCT'
#array_type = 'Nectar'
##array_type = 'Flash'
#template = 'yes'
##template = 'no'
#ana_tag = f"{array_type}_{evt_selection}_{pointing}_{template}"
##ana_unc_cut = 4.0
##ana_name = f"{array_type}"+" (default)"
##ana_container += [[ana_tag,ana_name,ana_unc_cut]]
#ana_unc_cut = 0.3
#ana_name = f"{array_type}"+" (least square $\sigma<%0.2f^{\circ}$)"%(ana_unc_cut)
#ana_container += [[ana_tag,ana_name,ana_unc_cut]]
#ana_unc_cut = 0.2
#ana_name = f"{array_type}"+" (least square $\sigma<%0.2f^{\circ}$)"%(ana_unc_cut)
#ana_container += [[ana_tag,ana_name,ana_unc_cut]]
#ana_unc_cut = 0.1
#ana_name = f"{array_type}"+" (least square $\sigma<%0.2f^{\circ}$)"%(ana_unc_cut)
#ana_container += [[ana_tag,ana_name,ana_unc_cut]]

#ana_container = []
#plot_name = 'sct_vs_nectar'
#pointing = 'onaxis'
##pointing = 'diffuse'
#ana_unc_cut = 0.1
#array_type = 'SCT'
#template = 'yes'
#template = 'no'
#ana_tag = f"{array_type}_{evt_selection}_{pointing}_{template}"
#ana_name = f"{array_type}"+" (least square $\sigma<%0.2f^{\circ}$)"%(ana_unc_cut)
#ana_container += [[ana_tag,ana_name,ana_unc_cut]]
##array_type = 'Nectar'
##ana_tag = f"{array_type}_{evt_selection}_{pointing}_{template}"
##ana_name = f"{array_type}"+" (least square $\sigma<%0.2f^{\circ}$)"%(ana_unc_cut)
##ana_container += [[ana_tag,ana_name,ana_unc_cut]]
#array_type = 'Flash'
#ana_tag = f"{array_type}_{evt_selection}_{pointing}_{template}"
#ana_name = f"{array_type}"+" (least square $\sigma<%0.2f^{\circ}$)"%(ana_unc_cut)
#ana_container += [[ana_tag,ana_name,ana_unc_cut]]

ana_container = []
plot_name = 'template_improve'
pointing = 'onaxis'
#pointing = 'diffuse'
#array_type = 'SCT'
#array_type = 'Flash'
array_type = 'Nectar'
template = 'yes'
ana_tag = f"{array_type}_{evt_selection}_{pointing}_{template}"
#ana_unc_cut = 4.0
#ana_name = f"{array_type}"+" (default)"
#ana_container += [[ana_tag,ana_name,ana_unc_cut]]
ana_unc_cut = 0.3
ana_name = f"{array_type}"+" (least square $\sigma<%0.2f^{\circ}$)"%(ana_unc_cut)
ana_container += [[ana_tag,ana_name,ana_unc_cut]]
ana_name = f"{array_type}"+" (template $\sigma<%0.2f^{\circ}$)"%(ana_unc_cut)
ana_container += [[ana_tag,ana_name,ana_unc_cut]]



font = {'family': 'serif', 'color':  'black', 'weight': 'normal', 'size': 10, 'rotation': 0.,}


hillas_unc_cut = 1e10

#off_angle_cut = 1.0
#off_angle_cut = 5.0
off_angle_cut = 1e10

hist_edge = 0.5


hist_truth_norm = []
hist_ref_norm = []
hist_ref_efficiency = []
hist_ref_off_angle = []
hist_ref_off_angle_err = []
for ref in range(0,len(ana_container)):
    hist_truth_norm += [Histogram(nbins=(6), ranges=[[-1,2]])]
    hist_ref_norm += [Histogram(nbins=(6), ranges=[[-1,2]])]
    hist_ref_efficiency += [Histogram(nbins=(6), ranges=[[-1,2]])]
    hist_ref_off_angle += [Histogram(nbins=(6), ranges=[[-1,2]])]
    hist_ref_off_angle_err += [Histogram(nbins=(6), ranges=[[-1,2]])]


def gauss_func(x,A,sigma):
    return A * np.exp(-((x-0.)**2)/(2*sigma*sigma))

def plot_analysis_result():

    list_tel_off_angle = []
    list_tel_unc = []
    list_tel_correlation = []

    list_all_truth_log_energy = []
    list_ref_truth_log_energy = []
    list_ref_truth_log_energy_pass = []
    list_ref_truth_log_energy_fail = []
    list_ref_alt = []
    list_ref_az = []
    list_ref_unc = []
    list_ref_unc_pass = []
    list_ref_unc_fail = []
    list_ref_off_angle = []
    list_ref_off_angle_pass = []
    list_ref_off_angle_fail = []
    list_ref_xing_outlier = []
    list_ref_temp_outlier = []
    for ref in range(0,len(ana_container)):
        list_all_truth_log_energy += [[]]
        list_ref_truth_log_energy += [[]]
        list_ref_truth_log_energy_pass += [[]]
        list_ref_truth_log_energy_fail += [[]]
        list_ref_alt += [[]]
        list_ref_az += [[]]
        list_ref_unc += [[]]
        list_ref_unc_pass += [[]]
        list_ref_unc_fail += [[]]
        list_ref_off_angle += [[]]
        list_ref_off_angle_pass += [[]]
        list_ref_off_angle_fail += [[]]
        list_ref_xing_outlier += [[]]
        list_ref_temp_outlier += [[]]

    for ref in range(0,len(ana_container)):

        ana_tag = ana_container[ref][0]
        ana_name = ana_container[ref][1]
        ana_cut = ana_container[ref][2]

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

        training_sample_path = []
        max_nfiles = 1e10
        nfiles = 0
        with open(f'{ctapipe_input}/{sim_files}', 'r') as file:
            for line in file:
                #training_sample_path += [get_dataset_path(line.strip('\n'))]
                training_sample_path += [line.strip('\n')]
                nfiles += 1
                if nfiles >= max_nfiles: break

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
                list_all_truth_log_energy[ref] += [np.log10(truth_energy)]

            if 'default' in ana_name:
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
                    if unc<ana_cut and unc>1e-04:
                        list_ref_off_angle_pass[ref] += [off_angle*off_angle]
                        list_ref_unc_pass[ref] += [unc]
                        list_ref_truth_log_energy_pass[ref] += [np.log10(truth_energy)]
                    else:
                        list_ref_off_angle_fail[ref] += [off_angle*off_angle]
                        list_ref_unc_fail[ref] += [unc]
                        list_ref_truth_log_energy_fail[ref] += [np.log10(truth_energy)]
            if 'least square' in ana_name:
                for evt in range(0,len(xing_result)):
                    truth_energy = xing_result[evt][0].value
                    if truth_energy<energy_cut: continue
                    off_angle = xing_result[evt][2]
                    if math.isnan(off_angle): continue
                    if off_angle>off_angle_cut: continue
                    unc = xing_result[evt][3]
                    alt = xing_result[evt][4]
                    az = xing_result[evt][5]
                    if unc==0.:
                        unc = 0.00001
                    run_id = xing_result[evt][6]
                    event_id = xing_result[evt][7]
                    if off_angle/unc>4. and unc>1e-04:
                        list_ref_xing_outlier[ref] += [[run_id,event_id,off_angle,unc]]
                    list_ref_off_angle[ref] += [off_angle*off_angle]
                    list_ref_alt[ref] += [alt]
                    list_ref_az[ref] += [az]
                    list_ref_unc[ref] += [unc]
                    list_ref_truth_log_energy[ref] += [np.log10(truth_energy)]
                    if unc<ana_cut and unc>1e-04:
                        list_ref_off_angle_pass[ref] += [off_angle*off_angle]
                        list_ref_unc_pass[ref] += [unc]
                        list_ref_truth_log_energy_pass[ref] += [np.log10(truth_energy)]
                    else:
                        list_ref_off_angle_fail[ref] += [off_angle*off_angle]
                        list_ref_unc_fail[ref] += [unc]
                        list_ref_truth_log_energy_fail[ref] += [np.log10(truth_energy)]
            if 'template' in ana_name:
                for evt in range(0,len(template_result)):
                    run_id = template_result[evt][7]
                    event_id = template_result[evt][8]
                    truth_energy = pow(10.,template_result[evt][4].value)
                    if truth_energy<energy_cut: continue
                    off_angle = template_result[evt][0]
                    if math.isnan(off_angle): continue
                    if off_angle>off_angle_cut: continue
                    unc = template_result[evt][1]
                    alt = template_result[evt][2]
                    az = template_result[evt][3]
                    if unc==0.:
                        unc = 0.00001
                    if off_angle/unc>4. and unc>1e-04:
                        list_ref_temp_outlier[ref] += [[run_id,event_id,off_angle,unc]]
                    list_ref_off_angle[ref] += [off_angle*off_angle]
                    list_ref_alt[ref] += [alt]
                    list_ref_az[ref] += [az]
                    list_ref_unc[ref] += [unc]
                    list_ref_truth_log_energy[ref] += [np.log10(truth_energy)]
                    if unc<ana_cut and unc>1e-04:
                        list_ref_off_angle_pass[ref] += [off_angle*off_angle]
                        list_ref_unc_pass[ref] += [unc]
                        list_ref_truth_log_energy_pass[ref] += [np.log10(truth_energy)]
                    else:
                        list_ref_off_angle_fail[ref] += [off_angle*off_angle]
                        list_ref_unc_fail[ref] += [unc]
                        list_ref_truth_log_energy_fail[ref] += [np.log10(truth_energy)]

                    tel_ana = template_result[evt][9]
                    tel_off_angle = tel_ana[5]
                    tel_unc = tel_ana[6]
                    tel_correlation = tel_ana[7]
                    for img in range(0,len(tel_off_angle)):
                        #if tel_correlation[img]<0.8: continue
                        list_tel_off_angle += [pow(float(tel_off_angle[img])*180./np.pi,2)]
                        list_tel_unc += [float(tel_unc[img])*180./np.pi]
                        list_tel_correlation += [float(tel_correlation[img])]

            if 'combined' in ana_name:
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
                    if unc<ana_cut and unc>1e-04:
                        list_ref_off_angle_pass[ref] += [off_angle*off_angle]
                        list_ref_unc_pass[ref] += [unc]
                        list_ref_truth_log_energy_pass[ref] += [np.log10(truth_energy)]
                    else:
                        list_ref_off_angle_fail[ref] += [off_angle*off_angle]
                        list_ref_unc_fail[ref] += [unc]
                        list_ref_truth_log_energy_fail[ref] += [np.log10(truth_energy)]

    


    for ref in range(0,len(ana_container)):
        hist_truth_norm[ref].fill(list_all_truth_log_energy[ref])
        hist_ref_norm[ref].fill(list_ref_truth_log_energy_pass[ref])
        hist_ref_off_angle[ref].fill(list_ref_truth_log_energy_pass[ref],weights=list_ref_off_angle_pass[ref])
        hist_ref_off_angle[ref].data = np.sqrt(hist_ref_off_angle[ref].data / hist_ref_norm[ref].data)
        hist_ref_off_angle_err[ref].data =  hist_ref_off_angle[ref].data / np.sqrt(hist_ref_norm[ref].data)
        hist_ref_efficiency[ref].data = hist_ref_norm[ref].data / hist_truth_norm[ref].data

    for ref in range(0,len(ana_container)):
        print ("xing_outlier:")
        for evt in range(0,len(list_ref_xing_outlier[ref])):
            print (f"{list_ref_xing_outlier[ref][evt]}")
        print ("temp_outlier:")
        for evt in range(0,len(list_ref_temp_outlier[ref])):
            print (f"{list_ref_temp_outlier[ref][evt]}")

    fig, ax = plt.subplots()
    figsize_x = 6.4
    figsize_y = 4.6
    fig.set_figheight(figsize_y)
    fig.set_figwidth(figsize_x)
    label_x = "log10 energy [TeV]"
    label_y = "reconstruction efficiency"
    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)
    for ref in range(0,len(ana_container)):
        ana_name = ana_container[ref][1]
        if 'default' in ana_name:
            ax.plot(hist_ref_efficiency[ref].bin_centers(0),hist_ref_efficiency[ref].data,c='k',ls='solid',label=ana_name)
        else:
            ax.plot(hist_ref_efficiency[ref].bin_centers(0),hist_ref_efficiency[ref].data,ls='dashed',label=ana_name)
    ax.legend(loc='best')
    ax.set_yscale('log')
    fig.savefig(
        f"{ctapipe_output}/output_plots/analysis_efficiency_{plot_name}.png",
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
    for ref in range(0,len(ana_container)):
        ana_name = ana_container[ref][1]
        if 'default' in ana_name:
            ax.errorbar(hist_ref_norm[ref].bin_centers(0),hist_ref_off_angle[ref].data,hist_ref_off_angle_err[ref].data,c='k',ls='solid',label=ana_name)
        else:
            ax.errorbar(hist_ref_norm[ref].bin_centers(0),hist_ref_off_angle[ref].data,hist_ref_off_angle_err[ref].data,ls='dashed',label=ana_name)
    ax.legend(loc='best')
    ax.set_yscale('log')
    fig.savefig(
        f"{ctapipe_output}/output_plots/analysis_off_angle_{plot_name}.png",
        bbox_inches="tight",
    )
    del fig
    del ax
    plt.close()


    for ref in range(0,len(ana_container)):
        ana_name = ana_container[ref][1]
        ref_mean = np.sqrt(np.mean(np.array(list_ref_off_angle_pass[ref])))
        print ("============================================")
        print (f"ana_name = {ana_name}")
        print (f"ana_mean = {ref_mean}")


    fig, ax = plt.subplots()
    figsize_x = 6.4
    figsize_y = 4.6
    fig.set_figheight(figsize_y)
    fig.set_figwidth(figsize_x)
    label_x = "angular distance [deg]"
    label_y = "count"
    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)
    for ref in range(0,len(ana_container)):
        ana_name = ana_container[ref][1]
        list_with_limit = []
        for entry in range(0,len(list_ref_off_angle[ref])):
            list_with_limit += [min(np.sqrt(list_ref_off_angle[ref][entry]),hist_edge-0.001)]
        ax.hist(list_with_limit,histtype='step',bins=25,range=(0.,hist_edge),label=ana_name)
    ax.legend(loc='best')
    fig.savefig(
        f"{ctapipe_output}/output_plots/reconstruction_off_angle_incl_{plot_name}.png",
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
    for ref in range(0,len(ana_container)):
        ana_name = ana_container[ref][1]
        list_with_limit = []
        for entry in range(0,len(list_ref_off_angle_pass[ref])):
            list_with_limit += [min(np.sqrt(list_ref_off_angle_pass[ref][entry]),hist_edge-0.001)]
        ax.hist(list_with_limit,histtype='step',bins=25,range=(0.,hist_edge),label=ana_name)
    ax.legend(loc='best')
    fig.savefig(
        f"{ctapipe_output}/output_plots/reconstruction_off_angle_pass_{plot_name}.png",
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
    for ref in range(0,len(ana_container)):
        ana_name = ana_container[ref][1]
        list_with_limit = []
        for entry in range(0,len(list_ref_off_angle_fail[ref])):
            list_with_limit += [min(np.sqrt(list_ref_off_angle_fail[ref][entry]),hist_edge-0.001)]
        ax.hist(list_with_limit,histtype='step',bins=25,range=(0.,hist_edge),label=ana_name)
    ax.legend(loc='best')
    fig.savefig(
        f"{ctapipe_output}/output_plots/reconstruction_off_angle_fail_{plot_name}.png",
        bbox_inches="tight",
    )
    del fig
    del ax
    plt.close()

    fig, ax = plt.subplots()
    figsize_x = 6.4
    figsize_y = 6.4
    fig.set_figheight(figsize_y)
    fig.set_figwidth(figsize_x)
    label_x = "observed error [deg]"
    label_y = "estimated uncertainty [deg]"
    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)
    max_unc = 0.5
    min_unc = 0.
    ax.scatter(
        np.sqrt(list_tel_off_angle),
        list_tel_unc,
        s=90,
        facecolors="none",
        c="r",
        alpha=0.2,
        marker="+",
    )
    n_intervals = 10
    for u in range(0,n_intervals):
        unc_axis = []
        off_angle_mean = []
        new_list_off_angle = []
        delta_unc = max_unc/float(n_intervals)
        lower_unc = float(u)*delta_unc + min_unc
        upper_unc = float(u+1)*delta_unc + min_unc
        for evt in range(0,len(list_tel_off_angle)):
            unc = list_tel_unc[evt]
            off_angle = list_tel_off_angle[evt]
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
    fig.savefig(
        f"{ctapipe_output}/output_plots/monotel_off_angle_vs_unc_{plot_name}.png",
        bbox_inches="tight",
    )
    del fig
    del ax
    plt.close()

    fig, ax = plt.subplots()
    figsize_x = 6.4
    figsize_y = 6.4
    fig.set_figheight(figsize_y)
    fig.set_figwidth(figsize_x)
    label_x = "observed error [deg]"
    label_y = "correlation"
    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)
    max_unc = 0.5
    min_unc = 0.
    ax.scatter(
        np.sqrt(list_tel_off_angle),
        list_tel_correlation,
        s=90,
        facecolors="none",
        c="r",
        alpha=0.2,
        marker="+",
    )
    ax.set_xlim(0., max_unc)
    ax.set_ylim(0., 1.)
    fig.savefig(
        f"{ctapipe_output}/output_plots/monotel_off_angle_vs_correlation_{plot_name}.png",
        bbox_inches="tight",
    )
    del fig
    del ax
    plt.close()


    for ref in range(0,len(ana_container)):
        ana_name = ana_container[ref][1]
        ana_cut = ana_container[ref][2]
        fig, ax = plt.subplots()
        figsize_x = 6.4
        figsize_y = 6.4
        fig.set_figheight(figsize_y)
        fig.set_figwidth(figsize_x)
        label_x = "observed error [deg]"
        label_y = "estimated uncertainty [deg]"
        ax.set_xlabel(label_x)
        ax.set_ylabel(label_y)
        max_unc = ana_cut
        if 'default' in ana_name:
            mean = np.sqrt(np.mean(list_ref_off_angle_pass[ref]))
            max_unc = np.max(2.*1.4*mean)
        min_unc = 0.
        ax.scatter(
            np.sqrt(list_ref_off_angle_pass[ref]),
            list_ref_unc_pass[ref],
            s=90,
            facecolors="none",
            c="r",
            alpha=0.2,
            marker="+",
            label=ana_name,
        )
        n_intervals = 10
        for u in range(0,n_intervals):
            unc_axis = []
            off_angle_mean = []
            new_list_off_angle = []
            delta_unc = ana_cut/float(n_intervals)
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
        ax.set_ylim(0., ana_cut)
        ax.legend(loc='best')
        fig.savefig(
            f"{ctapipe_output}/output_plots/off_angle_vs_unc_ref{ref}_{plot_name}.png",
            bbox_inches="tight",
        )
        del fig
        del ax
        plt.close()

    for ref in range(0,len(ana_container)):
        ana_name = ana_container[ref][1]
        ana_cut = ana_container[ref][2]
        fig, ax = plt.subplots()
        figsize_x = 6.4
        figsize_y = 4.6
        fig.set_figheight(figsize_y)
        fig.set_figwidth(figsize_x)
        label_x = "$\chi^{2}$ (error / uncertainty)"
        label_y = "count"
        ax.set_xlabel(label_x)
        ax.set_ylabel(label_y)
        list_chi2 = []
        for evt in range(0,len(list_ref_unc[ref])):
            error = np.sqrt(list_ref_off_angle[ref][evt])
            unc = list_ref_unc[ref][evt]/1.
            if unc<ana_cut:
                list_chi2 += [pow(error/unc,2)]
        list_chi2 = np.array(list_chi2)
        hist = ax.hist(list_chi2, bins=100, range=[0., 10.])
        print ('==========================================================================')
        Sum = float(len(list_chi2))
        Mean = 0.
        Variance = 0.
        for entry in range(0,len(hist[0])):
            X = hist[1][entry]
            f = hist[0][entry] / Sum
            Mean += X * f
            Variance += X * X * f
        Variance = Variance - Mean*Mean
        print (f"ana_cut = {ana_cut}")
        print (f"Sum = {Sum}")
        print (f"Mean = {Mean:0.4f}")
        print (f"Variance = {Variance:0.4f}")
        ax.set_yscale('log')
        fig.savefig(
            f"{ctapipe_output}/output_plots/multitel_chi2_distribution_ref{ref}_{plot_name}.png",
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
    label_x = "$\chi^{2}$ (error / uncertainty)"
    label_y = "count"
    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)
    list_chi2 = []
    for evt in range(0,len(list_tel_unc)):
        error = np.sqrt(list_tel_off_angle[evt])
        unc = list_tel_unc[evt]/1.
        if unc<0.5:
            list_chi2 += [pow(error/unc,2)]
    list_chi2 = np.array(list_chi2)
    hist = ax.hist(list_chi2, bins=100, range=[0., 10.])
    print ('==========================================================================')
    Sum = float(len(list_chi2))
    Mean = 0.
    Variance = 0.
    for entry in range(0,len(hist[0])):
        X = hist[1][entry]
        f = hist[0][entry] / Sum
        Mean += X * f
        Variance += X * X * f
    Variance = Variance - Mean*Mean
    print (f"Monoscopic")
    print (f"Sum = {Sum}")
    print (f"Mean = {Mean:0.4f}")
    print (f"Variance = {Variance:0.4f}")
    ax.set_yscale('log')
    fig.savefig(
        f"{ctapipe_output}/output_plots/monotel_chi2_distribution_{plot_name}.png",
        bbox_inches="tight",
    )
    del fig
    del ax
    plt.close()


    for ref in range(0,len(ana_container)):
        ana_name = ana_container[ref][1]
        ana_cut = ana_container[ref][2]
        list_weight = []
        for evt in range(0,len(list_ref_unc[ref])):
            if list_ref_unc[ref][evt]<ana_cut:
                list_weight += [1.]
            else:
                list_weight += [0.]
        fig, ax = plt.subplots()
        figsize_x = 4.6
        figsize_y = 4.6
        fig.set_figheight(figsize_y)
        fig.set_figwidth(figsize_x)
        label_x = "Az [deg]"
        label_y = "Alt [deg]"
        ax.set_xlabel(label_x)
        ax.set_ylabel(label_y)
        xmin = 0.0-0.5
        xmax = 0.0+0.5
        ymin = 70.0-0.2
        ymax = 70.0+0.2
        ax.hist2d(list_ref_az[ref][:], list_ref_alt[ref][:], weights=list_weight, bins=11, range=[[xmin, xmax], [ymin, ymax]], cmap='Greys')
        fig.savefig(
            f"{ctapipe_output}/output_plots/skymap_ref{ref}_{plot_name}.png",
            bbox_inches="tight",
        )
        del fig
        del ax
        plt.close()

plot_analysis_result()
