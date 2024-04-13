
import os, sys
import subprocess
import glob

from operator import itemgetter

import pickle
import numpy as np
from astropy import units as u
from matplotlib import pyplot as plt
from matplotlib import colors
from scipy.optimize import least_squares, minimize
from scipy.interpolate import LinearNDInterpolator
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from ctapipe.io import EventSource, SimTelEventSource, HDF5TableWriter
from ctapipe.calib import CameraCalibrator
from ctapipe.reco import ShowerProcessor
from ctapipe.image import ImageProcessor
from traitlets.config import Config

import common_functions
training_event_select = common_functions.training_event_select
image_size_cut = common_functions.image_size_cut
remove_nan_pixels = common_functions.remove_nan_pixels
reset_time = common_functions.reset_time
find_image_moments = common_functions.find_image_moments
image_translation = common_functions.image_translation
image_rotation = common_functions.image_rotation
find_image_truth = common_functions.find_image_truth
make_a_movie = common_functions.make_a_movie
make_standard_image = common_functions.make_standard_image
plot_monotel_reconstruction = common_functions.plot_monotel_reconstruction
camxy_to_altaz = common_functions.camxy_to_altaz
image_simulation = common_functions.image_simulation
movie_simulation = common_functions.movie_simulation
display_a_movie = common_functions.display_a_movie

fig, ax = plt.subplots()
figsize_x = 8.6
figsize_y = 6.4
fig.set_figheight(figsize_y)
fig.set_figwidth(figsize_x)

time_weight_ratio = 1.
use_poly = False
#use_poly = True

#ana_tag = 'movie'
ana_tag = 'image'

select_event_id = 0
#select_event_id = 39705

def sqaure_difference_between_1d_images(init_params,image_1d_data,lookup_table,eigen_vectors,full_table=False,use_poisson=False):

    fit_arrival = init_params[0]
    fit_impact = init_params[1]
    fit_log_energy = init_params[2]
    if not full_table:
        if lookup_table[0].get_bin_content(fit_arrival,fit_impact,fit_log_energy)==0.:
            return 1e10

    fit_latent_space = []
    for r in range(0,len(lookup_table)):
        fit_latent_space += [lookup_table[r].get_bin_content(fit_arrival,fit_impact,fit_log_energy)]
    fit_latent_space = np.array(fit_latent_space)


    sum_chi2 = 0.
    if use_poisson:
        image_1d_fit = eigen_vectors.T @ fit_latent_space
        n_rows = len(image_1d_fit)
        for row in range(0,n_rows):
            n_expect = max(0.0001,image_1d_fit[row])
            n_data = max(0.,image_1d_data[row])
            if n_data==0.:
                sum_chi2 += n_expect
            else:
                sum_chi2 += -1.*(n_data*np.log(n_expect) - n_expect - (n_data*np.log(n_data)-n_data))
    else:
        data_latent_space = eigen_vectors @ image_1d_data
        n_rows = len(data_latent_space)
        for row in range(0,n_rows):
            if data_latent_space[row]==0. and fit_latent_space[row]==0.: continue
            diff = data_latent_space[row] - fit_latent_space[row]
            sum_chi2 += diff*diff

    return sum_chi2

def combined_sqaure_difference_between_1d_images_spline(init_params,image_1d_data,image_lookup_table_spline,image_eigen_vectors,time_1d_data,time_lookup_table_spline,time_eigen_vectors):

    image_weight = 1./np.sum(np.array(image_1d_data)*np.array(image_1d_data))
    time_weight = time_weight_ratio/np.sum(np.array(time_1d_data)*np.array(time_1d_data))

    image_chi2 = sqaure_difference_between_1d_images_spline(init_params,image_1d_data,image_lookup_table_spline,image_eigen_vectors)
    time_chi2 = sqaure_difference_between_1d_images_spline(init_params,time_1d_data,time_lookup_table_spline,time_eigen_vectors)

    fit_arrival = init_params[0]
    fit_impact = init_params[1]
    fit_log_energy = init_params[2]
    sum_chi2 = image_weight*image_chi2 + time_weight*time_chi2

    #print (f'try fit_arrival = {fit_arrival:0.3f}, fit_impact = {fit_impact:0.1f}, fit_log_energy = {fit_log_energy:0.3f}')
    #print (f'sum_chi2 = {sum_chi2}')

    return sum_chi2


def sqaure_difference_between_1d_images_spline(init_params,image_1d_data,lookup_table_spline,eigen_vectors):

    fit_arrival = init_params[0]
    fit_impact = init_params[1]
    fit_log_energy = init_params[2]

    #print (f'try fit_arrival = {fit_arrival:0.3f}, fit_impact = {fit_impact:0.1f}, fit_log_energy = {fit_log_energy:0.3f}')

    fit_latent_space = []
    for r in range(0,len(lookup_table_spline)):
        new_params_poly = lookup_table_spline[r][0].transform(np.array([[fit_arrival, fit_impact, fit_log_energy]]))
        fit_latent_space += [lookup_table_spline[r][1].predict(new_params_poly)[0]]
        #fit_latent_space += [lookup_table_spline[r].get_bin_content(fit_arrival,fit_impact,fit_log_energy)]
    fit_latent_space = np.array(fit_latent_space)

    sum_fit_latent_space = np.sum(fit_latent_space)
    #print (f'sum_fit_latent_space_spline = {sum_fit_latent_space}')
    if np.isnan(sum_fit_latent_space):
        return 1e10

    data_latent_space = eigen_vectors @ image_1d_data
    sum_chi2 = 0.
    n_rows = len(data_latent_space)
    for row in range(0,n_rows):
        if data_latent_space[row]==0. and fit_latent_space[row]==0.: continue
        diff = data_latent_space[row] - fit_latent_space[row]
        sum_chi2 += diff*diff

    #print (f'sum_chi2 = {sum_chi2}')

    return sum_chi2

def sortFirst(val):
    return val[0]

def single_movie_reconstruction(input_image_1d,image_lookup_table,image_eigen_vectors,input_time_1d,time_lookup_table,time_eigen_vectors,input_movie_1d,movie_lookup_table,movie_eigen_vectors, use_movie=True):

    image_weight = 1./np.sum(np.array(input_image_1d)*np.array(input_image_1d))
    time_weight = time_weight_ratio/np.sum(np.array(input_time_1d)*np.array(input_time_1d))

    n_bins_arrival = len(movie_lookup_table[0].xaxis)
    n_bins_impact = len(movie_lookup_table[0].yaxis)
    n_bins_energy = len(movie_lookup_table[0].zaxis)

    fit_arrival = 0.15
    fit_impact = 100.
    fit_log_energy = 0.
    init_params = [fit_arrival,fit_impact,fit_log_energy]
    fit_chi2_image = image_weight*sqaure_difference_between_1d_images(init_params,input_image_1d,image_lookup_table,image_eigen_vectors)
    fit_chi2_time = time_weight*sqaure_difference_between_1d_images(init_params,input_time_1d,time_lookup_table,time_eigen_vectors)
    fit_chi2 = fit_chi2_image + fit_chi2_time

    fit_coord = []
    fit_idx_x = 0
    fit_idx_y = 0
    fit_idx_z = 0
    for idx_x  in range(0,n_bins_arrival):
        for idx_y  in range(0,n_bins_impact):
            for idx_z  in range(0,n_bins_energy):

                try_arrival = image_lookup_table[0].xaxis[idx_x]
                try_impact = image_lookup_table[0].yaxis[idx_y]
                try_log_energy = image_lookup_table[0].zaxis[idx_z]
                init_params = [try_arrival,try_impact,try_log_energy]

                try_chi2_image = image_weight*sqaure_difference_between_1d_images(init_params,input_image_1d,image_lookup_table,image_eigen_vectors)
                try_chi2_time = time_weight*sqaure_difference_between_1d_images(init_params,input_time_1d,time_lookup_table,time_eigen_vectors)
                try_chi2 = try_chi2_image + try_chi2_time

                fit_coord += [(try_chi2,try_arrival,try_impact,try_log_energy)]

                if try_chi2<fit_chi2:
                    fit_chi2 = try_chi2
                    fit_arrival = try_arrival
                    fit_impact = try_impact
                    fit_log_energy = try_log_energy
                    fit_idx_x = idx_x
                    fit_idx_y = idx_y
                    fit_idx_z = idx_z

    cov_arrival = 0.
    cov_impact = 0.
    cov_log_energy = 0.

    if not use_movie:
        return fit_arrival+0.005, fit_impact+0.005, fit_log_energy+0.05, pow(cov_arrival,0.5), pow(cov_impact,0.5), pow(cov_log_energy,0.5), fit_chi2

    #fit_arrival = 0.15
    #fit_impact = 100.
    #fit_log_energy = 0.
    #init_params = [fit_arrival,fit_impact,fit_log_energy]
    #image_fit_chi2 = sqaure_difference_between_1d_images(init_params,input_image_1d,image_lookup_table,image_eigen_vectors)
    #movie_fit_chi2 = sqaure_difference_between_1d_images(init_params,input_movie_1d,movie_lookup_table,movie_eigen_vectors)
    #fit_chi2 = image_weight*image_fit_chi2 + movie_weight*movie_fit_chi2

    #fit_coord = []
    #for idx_x  in range(0,n_bins_arrival):
    #    for idx_y  in range(0,n_bins_impact):
    #        for idx_z  in range(0,n_bins_energy):

    #            try_arrival = movie_lookup_table[0].xaxis[idx_x]
    #            try_impact = movie_lookup_table[0].yaxis[idx_y]
    #            try_log_energy = movie_lookup_table[0].zaxis[idx_z]
    #            init_params = [try_arrival,try_impact,try_log_energy]

    #            image_try_chi2 = sqaure_difference_between_1d_images(init_params,input_image_1d,image_lookup_table,image_eigen_vectors)
    #            movie_try_chi2 = sqaure_difference_between_1d_images(init_params,input_movie_1d,movie_lookup_table,movie_eigen_vectors)
    #            try_chi2 = image_weight*image_try_chi2 + movie_weight*movie_try_chi2

    #            if try_chi2<1e10:
    #                fit_coord += [(try_chi2,try_arrival,try_impact,try_log_energy)]

    #            if try_chi2<fit_chi2:
    #                fit_chi2 = try_chi2
    #                fit_arrival = try_arrival
    #                fit_impact = try_impact
    #                fit_log_energy = try_log_energy

    image_weight = 1.
    movie_weight = 1.

    print (f'initial fit_arrival = {fit_arrival}, fit_impact = {fit_impact}, fit_log_energy = {fit_log_energy}')
    fit_coord.sort(key=sortFirst)
    fit_chi2 = 1e10
    #for entry in range(0,len(fit_coord)):
    for entry in range(0,min(10,len(fit_coord))):
        try_arrival = fit_coord[entry][1]
        try_impact = fit_coord[entry][2]
        try_log_energy = fit_coord[entry][3]
        init_params = [try_arrival,try_impact,try_log_energy]
        image_try_chi2 = sqaure_difference_between_1d_images(init_params,input_image_1d,image_lookup_table,image_eigen_vectors,use_poisson=True)
        movie_try_chi2 = sqaure_difference_between_1d_images(init_params,input_movie_1d,movie_lookup_table,movie_eigen_vectors,use_poisson=True)
        try_chi2 = image_weight*image_try_chi2 + movie_weight*movie_try_chi2
        if try_chi2<fit_chi2:
            fit_chi2 = try_chi2
            fit_arrival = try_arrival
            fit_impact = try_impact
            fit_log_energy = try_log_energy
    print (f'Final fit_arrival = {fit_arrival}, fit_impact = {fit_impact}, fit_log_energy = {fit_log_energy}')

    cov_arrival = 0.
    cov_impact = 0.
    cov_log_energy = 0.

    #n_sampling = 3
    #fit_coord.sort(key=sortFirst)
    #for entry in range(1,n_sampling+1):
    #    dchi2 = fit_coord[entry][0] - fit_coord[0][0]
    #    dx2 = pow(fit_coord[entry][1]-fit_coord[0][1],2)
    #    dy2 = pow(fit_coord[entry][2]-fit_coord[0][2],2)
    #    dz2 = pow(fit_coord[entry][3]-fit_coord[0][3],2)
    #    cov_arrival += dx2/dchi2*fit_chi2
    #    cov_impact = dy2/dchi2*fit_chi2
    #    cov_log_energy = dz2/dchi2*fit_chi2
    #cov_arrival = cov_arrival/float(n_sampling)
    #cov_impact = cov_impact/float(n_sampling)
    #cov_log_energy = cov_log_energy/float(n_sampling)

    #hessian_xx = 0.
    #hessian_yy = 0.
    #hessian_zz = 0.
    #delta_x = movie_lookup_table[0].xaxis[1]-movie_lookup_table[0].xaxis[0]
    #delta_y = movie_lookup_table[0].yaxis[1]-movie_lookup_table[0].yaxis[0]
    #delta_z = movie_lookup_table[0].zaxis[1]-movie_lookup_table[0].zaxis[0]
    #for idx_x  in range(0,n_bins_arrival):
    #    for idx_y  in range(0,n_bins_impact):
    #        for idx_z  in range(0,n_bins_energy):
    #            try_arrival = movie_lookup_table[0].xaxis[idx_x]
    #            try_impact = movie_lookup_table[0].yaxis[idx_y]
    #            try_log_energy = movie_lookup_table[0].zaxis[idx_z]
    #            init_params = [try_arrival,try_impact,try_log_energy]
    #            if idx_x==fit_idx_x:
    #                hessian_xx += -2.*image_weight*sqaure_difference_between_1d_images(init_params,input_image_1d,image_lookup_table,image_eigen_vectors)
    #                hessian_xx += -2.*movie_weight*sqaure_difference_between_1d_images(init_params,input_movie_1d,movie_lookup_table,movie_eigen_vectors)
    #            if idx_x==fit_idx_x-1 or idx_x==fit_idx_x+1:
    #                hessian_xx += 1.*image_weight*sqaure_difference_between_1d_images(init_params,input_image_1d,image_lookup_table,image_eigen_vectors)
    #                hessian_xx += 1.*movie_weight*sqaure_difference_between_1d_images(init_params,input_movie_1d,movie_lookup_table,movie_eigen_vectors)
    #            if idx_y==fit_idx_y:
    #                hessian_yy += -2.*image_weight*sqaure_difference_between_1d_images(init_params,input_image_1d,image_lookup_table,image_eigen_vectors)
    #                hessian_yy += -2.*movie_weight*sqaure_difference_between_1d_images(init_params,input_movie_1d,movie_lookup_table,movie_eigen_vectors)
    #            if idx_y==fit_idx_y-1 or idx_y==fit_idx_y+1:
    #                hessian_yy += 1.*image_weight*sqaure_difference_between_1d_images(init_params,input_image_1d,image_lookup_table,image_eigen_vectors)
    #                hessian_yy += 1.*movie_weight*sqaure_difference_between_1d_images(init_params,input_movie_1d,movie_lookup_table,movie_eigen_vectors)
    #            if idx_z==fit_idx_z:
    #                hessian_zz += -2.*image_weight*sqaure_difference_between_1d_images(init_params,input_image_1d,image_lookup_table,image_eigen_vectors)
    #                hessian_zz += -2.*movie_weight*sqaure_difference_between_1d_images(init_params,input_movie_1d,movie_lookup_table,movie_eigen_vectors)
    #            if idx_z==fit_idx_z-1 or idx_z==fit_idx_z+1:
    #                hessian_zz += 1.*image_weight*sqaure_difference_between_1d_images(init_params,input_image_1d,image_lookup_table,image_eigen_vectors)
    #                hessian_zz += 1.*movie_weight*sqaure_difference_between_1d_images(init_params,input_movie_1d,movie_lookup_table,movie_eigen_vectors)
    #cov_arrival = delta_x*delta_x/hessian_xx
    #cov_impact = delta_y*delta_y/hessian_yy
    #cov_log_energy = delta_z*delta_z/hessian_zz
    #print (f'cov_arrival = {cov_arrival:0.2e}, cov_impact = {cov_impact:0.2e}, cov_log_energy = {cov_log_energy:0.2e}')

    #print (f'initial fit_arrival = {fit_arrival:0.3f}, fit_impact = {fit_impact:0.1f}, fit_log_energy = {fit_log_energy:0.3f}')
    #arrival_lower = fit_arrival*0.5
    #arrival_upper = fit_arrival*2.0
    #impact_lower = fit_impact*0.5
    #impact_upper = fit_impact*2.0
    #log_energy_lower = fit_log_energy-np.log10(2)
    #log_energy_upper = fit_log_energy+np.log10(2)

    #init_params = [fit_arrival,fit_impact,fit_log_energy]
    #stepsize = [1e-4,1e-1,1e-4]
    #bounds = [(arrival_lower,arrival_upper),(impact_lower,impact_upper),(log_energy_lower,log_energy_upper)]
    #solution = minimize(
    #    sqaure_difference_between_1d_images_spline,
    #    x0=init_params,
    #    args=(input_movie_1d,movie_lookup_table_spline,movie_eigen_vectors),
    #    bounds=bounds,
    #    method='L-BFGS-B',
    #    jac=None,
    #    options={'eps':stepsize,'ftol':0.0001},
    #)
    #fit_params = solution['x']
    #fit_arrival = fit_params[0]
    #fit_impact = fit_params[1]
    #fit_log_energy = fit_params[2]
    #fit_chi2 = 0.
    #print (f'final fit_arrival = {fit_arrival:0.3f}, fit_impact = {fit_impact:0.1f}, fit_log_energy = {fit_log_energy:0.3f}')
    #exit()

    return fit_arrival+0.005, fit_impact+0.005, fit_log_energy+0.05, pow(cov_arrival,0.5), pow(cov_impact,0.5), pow(cov_log_energy,0.5), fit_chi2


def single_image_reconstruction(input_image_1d,image_lookup_table,image_lookup_table_spline,image_eigen_vectors,input_time_1d,time_lookup_table,time_lookup_table_spline,time_eigen_vectors,use_spline=False):

    image_weight = 1./np.sum(np.array(input_image_1d)*np.array(input_image_1d))
    time_weight = time_weight_ratio/np.sum(np.array(input_time_1d)*np.array(input_time_1d))

    fit_arrival = 0.15
    fit_impact = 100.
    fit_log_energy = 0.
    init_params = [fit_arrival,fit_impact,fit_log_energy]
    fit_chi2_image = image_weight*sqaure_difference_between_1d_images(init_params,input_image_1d,image_lookup_table,image_eigen_vectors)
    fit_chi2_time = time_weight*sqaure_difference_between_1d_images(init_params,input_time_1d,time_lookup_table,time_eigen_vectors)
    fit_chi2 = fit_chi2_image + fit_chi2_time

    n_bins_arrival = len(image_lookup_table[0].xaxis)
    n_bins_impact = len(image_lookup_table[0].yaxis)
    n_bins_energy = len(image_lookup_table[0].zaxis)

    cov_arrival = 0.
    cov_impact = 0.
    cov_log_energy = 0.

    if use_spline:

        arrival_lower = 0.
        arrival_upper = 0.4
        impact_lower = 0.
        impact_upper = 800.
        log_energy_lower = -1.
        log_energy_upper = 2.

        print (f'initial fit_arrival = {fit_arrival:0.3f}, fit_impact = {fit_impact:0.1f}, fit_log_energy = {fit_log_energy:0.3f}')
        init_params = [fit_arrival,fit_impact,fit_log_energy]
        stepsize = [1e-4,1e-1,1e-4]
        bounds = [(arrival_lower,arrival_upper),(impact_lower,impact_upper),(log_energy_lower,log_energy_upper)]
        solution = minimize(
            combined_sqaure_difference_between_1d_images_spline,
            x0=init_params,
            args=(input_image_1d,image_lookup_table_spline,image_eigen_vectors,input_time_1d,time_lookup_table_spline,time_eigen_vectors),
            #args=(input_image_1d,image_lookup_table,image_eigen_vectors,input_time_1d,time_lookup_table,time_eigen_vectors),
            bounds=bounds,
            method='L-BFGS-B',
            jac=None,
            options={'eps':stepsize,'ftol':0.0001},
        )
        fit_params = solution['x']
        fit_arrival = fit_params[0]
        fit_impact = fit_params[1]
        fit_log_energy = fit_params[2]
        fit_params_err = solution['hess_inv'].todense()
        cov_arrival = fit_params_err[0][0]
        cov_impact = fit_params_err[1][1]
        cov_log_energy = fit_params_err[2][2]

    else:

        fit_coord = []
        fit_idx_x = 0
        fit_idx_y = 0
        fit_idx_z = 0
        for idx_x  in range(0,n_bins_arrival):
            for idx_y  in range(0,n_bins_impact):
                for idx_z  in range(0,n_bins_energy):

                    try_arrival = image_lookup_table[0].xaxis[idx_x]
                    try_impact = image_lookup_table[0].yaxis[idx_y]
                    try_log_energy = image_lookup_table[0].zaxis[idx_z]
                    init_params = [try_arrival,try_impact,try_log_energy]

                    try_chi2_image = image_weight*sqaure_difference_between_1d_images(init_params,input_image_1d,image_lookup_table,image_eigen_vectors)
                    try_chi2_time = time_weight*sqaure_difference_between_1d_images(init_params,input_time_1d,time_lookup_table,time_eigen_vectors)
                    try_chi2 = try_chi2_image + try_chi2_time

                    fit_coord += [(try_chi2,try_arrival,try_impact,try_log_energy)]

                    if try_chi2<fit_chi2:
                        fit_chi2 = try_chi2
                        fit_arrival = try_arrival
                        fit_impact = try_impact
                        fit_log_energy = try_log_energy
                        fit_idx_x = idx_x
                        fit_idx_y = idx_y
                        fit_idx_z = idx_z

        cov_arrival = 0.
        cov_impact = 0.
        cov_log_energy = 0.

        #n_sampling = 3
        #fit_coord.sort(key=sortFirst)
        #print (f'fit_chi2 = {fit_chi2}')
        #print (f'fit_coord[0] = {fit_coord[0]}')
        #print (f'fit_coord[1] = {fit_coord[1]}')
        #print (f'fit_coord[2] = {fit_coord[2]}')
        #for entry in range(1,n_sampling+1):
        #    dchi2 = fit_coord[entry][0] - fit_coord[0][0]
        #    dx2 = pow(fit_coord[entry][1]-fit_coord[0][1],2)
        #    dy2 = pow(fit_coord[entry][2]-fit_coord[0][2],2)
        #    dz2 = pow(fit_coord[entry][3]-fit_coord[0][3],2)
        #    if dchi2==0.:
        #        cov_arrival = 1e10
        #        cov_impact = 1e10
        #        cov_log_energy = 1e10
        #    else:
        #        cov_arrival += dx2/dchi2
        #        cov_impact = dy2/dchi2
        #        cov_log_energy = dz2/dchi2
        #cov_arrival = cov_arrival/float(n_sampling)
        #cov_impact = cov_impact/float(n_sampling)
        #cov_log_energy = cov_log_energy/float(n_sampling)

        #hessian_xx = 0.
        #hessian_yy = 0.
        #hessian_zz = 0.
        #delta_x = time_lookup_table[0].xaxis[1]-time_lookup_table[0].xaxis[0]
        #delta_y = time_lookup_table[0].yaxis[1]-time_lookup_table[0].yaxis[0]
        #delta_z = time_lookup_table[0].zaxis[1]-time_lookup_table[0].zaxis[0]
        #for idx_x  in range(0,n_bins_arrival):
        #    for idx_y  in range(0,n_bins_impact):
        #        for idx_z  in range(0,n_bins_energy):
        #            try_arrival = time_lookup_table[0].xaxis[idx_x]
        #            try_impact = time_lookup_table[0].yaxis[idx_y]
        #            try_log_energy = time_lookup_table[0].zaxis[idx_z]
        #            init_params = [try_arrival,try_impact,try_log_energy]
        #            if idx_x==fit_idx_x:
        #                hessian_xx += -2.*image_weight*sqaure_difference_between_1d_images(init_params,input_image_1d,image_lookup_table,image_eigen_vectors,full_table=True)
        #                hessian_xx += -2.*time_weight*sqaure_difference_between_1d_images(init_params,input_time_1d,time_lookup_table,time_eigen_vectors,full_table=True)
        #            if idx_x==fit_idx_x-1 or idx_x==fit_idx_x+1:
        #                hessian_xx += 1.*image_weight*sqaure_difference_between_1d_images(init_params,input_image_1d,image_lookup_table,image_eigen_vectors,full_table=True)
        #                hessian_xx += 1.*time_weight*sqaure_difference_between_1d_images(init_params,input_time_1d,time_lookup_table,time_eigen_vectors,full_table=True)
        #            if idx_y==fit_idx_y:
        #                hessian_yy += -2.*image_weight*sqaure_difference_between_1d_images(init_params,input_image_1d,image_lookup_table,image_eigen_vectors,full_table=True)
        #                hessian_yy += -2.*time_weight*sqaure_difference_between_1d_images(init_params,input_time_1d,time_lookup_table,time_eigen_vectors,full_table=True)
        #            if idx_y==fit_idx_y-1 or idx_y==fit_idx_y+1:
        #                hessian_yy += 1.*image_weight*sqaure_difference_between_1d_images(init_params,input_image_1d,image_lookup_table,image_eigen_vectors,full_table=True)
        #                hessian_yy += 1.*time_weight*sqaure_difference_between_1d_images(init_params,input_time_1d,time_lookup_table,time_eigen_vectors,full_table=True)
        #            if idx_z==fit_idx_z:
        #                hessian_zz += -2.*image_weight*sqaure_difference_between_1d_images(init_params,input_image_1d,image_lookup_table,image_eigen_vectors,full_table=True)
        #                hessian_zz += -2.*time_weight*sqaure_difference_between_1d_images(init_params,input_time_1d,time_lookup_table,time_eigen_vectors,full_table=True)
        #            if idx_z==fit_idx_z-1 or idx_z==fit_idx_z+1:
        #                hessian_zz += 1.*image_weight*sqaure_difference_between_1d_images(init_params,input_image_1d,image_lookup_table,image_eigen_vectors,full_table=True)
        #                hessian_zz += 1.*time_weight*sqaure_difference_between_1d_images(init_params,input_time_1d,time_lookup_table,time_eigen_vectors,full_table=True)
        #cov_arrival = delta_x*delta_x/hessian_xx*fit_chi2
        #cov_impact = delta_y*delta_y/hessian_yy*fit_chi2
        #cov_log_energy = delta_z*delta_z/hessian_zz*fit_chi2
        #print (f'cov_arrival = {cov_arrival}, cov_impact = {cov_impact}, cov_log_energy = {cov_log_energy}')

    return fit_arrival+0.005, fit_impact+0.005, fit_log_energy+0.05, pow(cov_arrival,0.5), pow(cov_impact,0.5), pow(cov_log_energy,0.5)


def run_monotel_analysis(training_sample_path, min_energy=0.1, max_energy=1000., max_evt=1e10):

    analysis_result = []

    print ('loading svd pickle data... ')
    output_filename = f'{ctapipe_output}/output_machines/movie_lookup_table.pkl'
    movie_lookup_table_pkl = pickle.load(open(output_filename, "rb"))
    output_filename = f'{ctapipe_output}/output_machines/movie_lookup_table_spline.pkl'
    movie_lookup_table_spline = pickle.load(open(output_filename, "rb"))
    output_filename = f'{ctapipe_output}/output_machines/movie_eigen_vectors.pkl'
    movie_eigen_vectors_pkl = pickle.load(open(output_filename, "rb"))

    output_filename = f'{ctapipe_output}/output_machines/image_lookup_table.pkl'
    image_lookup_table_pkl = pickle.load(open(output_filename, "rb"))
    output_filename = f'{ctapipe_output}/output_machines/image_lookup_table_spline.pkl'
    image_lookup_table_spline = pickle.load(open(output_filename, "rb"))
    output_filename = f'{ctapipe_output}/output_machines/image_eigen_vectors.pkl'
    image_eigen_vectors_pkl = pickle.load(open(output_filename, "rb"))
    
    output_filename = f'{ctapipe_output}/output_machines/time_lookup_table.pkl'
    time_lookup_table_pkl = pickle.load(open(output_filename, "rb"))
    output_filename = f'{ctapipe_output}/output_machines/time_lookup_table_spline.pkl'
    time_lookup_table_spline = pickle.load(open(output_filename, "rb"))
    output_filename = f'{ctapipe_output}/output_machines/time_eigen_vectors.pkl'
    time_eigen_vectors_pkl = pickle.load(open(output_filename, "rb"))

    print (f'loading file: {training_sample_path}')
    source = SimTelEventSource(training_sample_path, focal_length_choice='EQUIVALENT')
    
    # Explore the instrument description
    subarray = source.subarray
    print (subarray.to_table())
    
    ob_keys = source.observation_blocks.keys()
    run_id = list(ob_keys)[0]
    
    tel_pointing_alt = float(source.observation_blocks[run_id].subarray_pointing_lat/u.rad)
    tel_pointing_az = float(source.observation_blocks[run_id].subarray_pointing_lon/u.rad)
    print (f'tel_pointing_alt = {tel_pointing_alt}')
    print (f'tel_pointing_az = {tel_pointing_az}')
    
    image_processor_config = Config(
            {
                "ImageProcessor": {
                    "image_cleaner_type": "TailcutsImageCleaner",
                    "TailcutsImageCleaner": {
                        "picture_threshold_pe": [
                            ("type", "LST_LST_LSTCam", 7.5),
                            ("type", "MST_MST_FlashCam", 8),
                            ("type", "MST_MST_NectarCam", 8),
                            ("type", "MST_SCT_SCTCam", 8),
                            ("type", "SST_ASTRI_CHEC", 7),
                            ],
                        "boundary_threshold_pe": [
                            ("type", "LST_LST_LSTCam", 5),
                            ("type", "MST_MST_FlashCam", 4),
                            ("type", "MST_MST_NectarCam", 4),
                            ("type", "MST_SCT_SCTCam", 4),
                            ("type", "SST_ASTRI_CHEC", 4),
                            ],
                        },
                    }
                }
            )
    calib = CameraCalibrator(subarray=subarray)
    image_processor = ImageProcessor(subarray=subarray,config=image_processor_config)
    shower_processor = ShowerProcessor(subarray=subarray)

    evt_count = 0
    good_analysis = 0
    avg_analysis = 0
    bad_analysis = 0
    for event in source:
    
        event_id = event.index['event_id']
    
        evt_count += 1
        if (evt_count % training_event_select)==0: continue

        ntel = len(event.r0.tel)
        
        calib(event)  # fills in r1, dl0, and dl1
        #image_processor(event) # Takes DL1/Image data and cleans and parametrizes the images into DL1/parameters. Should be run after CameraCalibrator.
        #shower_processor(event) # Run the stereo event reconstruction on the input events.
    
        for tel_idx in range(0,len(list(event.dl0.tel.keys()))):

            tel_id = list(event.dl0.tel.keys())[tel_idx]

            if select_event_id!=0:
                if event_id!=select_event_id: continue
            #if tel_id!=24: continue

            print ('====================================================================================')
            print (f'event_id = {event_id}, tel_id = {tel_id}')

            truth_info_array = find_image_truth(source, subarray, run_id, tel_id, event)
            truth_energy = float(truth_info_array[0]/u.TeV)
            truth_core_x = truth_info_array[1]
            truth_core_y = truth_info_array[2]
            truth_alt = float(truth_info_array[3]/u.rad)
            truth_az = float(truth_info_array[4]/u.rad)
            truth_height = truth_info_array[5]
            truth_xmax = truth_info_array[6]
            star_cam_x = truth_info_array[7]
            star_cam_y = truth_info_array[8]
            impact_x = truth_info_array[9]
            impact_y = truth_info_array[10]
            focal_length = source.subarray.tel[tel_id].optics.equivalent_focal_length/u.m

            lightcone, image_moment_array, eco_image_1d, eco_time_1d = make_standard_image(fig, subarray, run_id, tel_id, event)
            image_size = image_moment_array[0]
            image_center_x = image_moment_array[1]
            image_center_y = image_moment_array[2]
            angle = image_moment_array[3]
            semi_major = image_moment_array[4]
            semi_minor = image_moment_array[5]
            time_direction = image_moment_array[6]
            image_direction = image_moment_array[7]
            line_a = image_moment_array[8]
            line_b = image_moment_array[9]
            print (f'image_size = {image_size}')

            if image_size<image_size_cut: 
                print ('failed image_size_cut')
                continue
            #if image_size>300.: 
            #    print ('failed image_size_cut')
            #    continue
            #if not pass_lightcone(lightcone,image_direction): continue


            truth_projection = image_moment_array[10]
            print (f'truth_projection = {truth_projection:0.3f}')

            image_fit_arrival = 0.
            image_fit_impact = 0.
            image_fit_log_energy = 0.
            image_fit_arrival_err = 0.
            image_fit_impact_err = 0.
            image_fit_log_energy_err = 0.
            image_fit_chi2 = 1e10

            use_movie = True
            if 'movie' in ana_tag:
                use_movie = True
            else:
                use_movie = False

            lightcone, image_moment_array, eco_movie_1d = make_a_movie(fig, subarray, run_id, tel_id, event, make_plots=False)
            image_fit_arrival, image_fit_impact, image_fit_log_energy, image_fit_arrival_err, image_fit_impact_err, image_fit_log_energy_err, image_fit_chi2 = single_movie_reconstruction(eco_image_1d,image_lookup_table_pkl,image_eigen_vectors_pkl,eco_time_1d,time_lookup_table_pkl,time_eigen_vectors_pkl,eco_movie_1d,movie_lookup_table_pkl,movie_eigen_vectors_pkl,use_movie=use_movie)

            image_fit_cam_x = image_center_x + image_fit_arrival*np.cos(angle*u.rad)
            image_fit_cam_y = image_center_y + image_fit_arrival*np.sin(angle*u.rad)
            image_fit_alt, image_fit_az = camxy_to_altaz(source, subarray, run_id, tel_id, image_fit_cam_x, image_fit_cam_y)

            image_method_unc = image_fit_arrival_err/focal_length*180./np.pi
            image_method_error = pow(pow(image_fit_cam_x-star_cam_x,2)+pow(image_fit_cam_y-star_cam_y,2),0.5)/focal_length*180./np.pi

            print (f'focal_length = {focal_length}')
            print (f'lightcone = {lightcone}')
            print (f'star_cam_x = {star_cam_x}')
            print (f'star_cam_y = {star_cam_y}')
            print (f'image_fit_cam_x = {image_fit_cam_x}')
            print (f'image_fit_cam_y = {image_fit_cam_y}')
            print (f'truth_alt = {truth_alt}')
            print (f'truth_az = {truth_az}')
            print (f'image_fit_alt = {image_fit_alt}')
            print (f'image_fit_az = {image_fit_az}')

            print (f'truth_energy     = {truth_energy}')
            print (f'image_fit_energy = {pow(10.,image_fit_log_energy)}')
            print (f'truth_fit_impact = {pow(impact_x*impact_x+impact_y*impact_y,0.5)}')
            print (f'image_fit_impact = {image_fit_impact}')
            print (f'image_method_error = {image_method_error:0.3f} deg')
            print (f'image_method_unc = {image_method_unc:0.3f} deg')

            fit_arrival = image_fit_arrival
            fit_impact = image_fit_impact
            fit_log_energy = image_fit_log_energy
            fit_cam_x = image_fit_cam_x
            fit_cam_y = image_fit_cam_y
            fit_alt = image_fit_alt
            fit_az = image_fit_az

            #if image_size>image_size_cut:
            if image_size>5000. or select_event_id!=0:

                if 'image' in ana_tag:

                    fit_params = [fit_arrival,fit_impact,fit_log_energy]
                    plot_monotel_reconstruction(fig, subarray, run_id, tel_id, event, image_moment_array, fit_cam_x, fit_cam_y, 'image')
                    image_simulation(fig, subarray, run_id, tel_id, event, fit_params, image_lookup_table_pkl, image_eigen_vectors_pkl, time_lookup_table_pkl, time_eigen_vectors_pkl)

                if 'movie' in ana_tag:

                    fit_params = [fit_arrival,fit_impact,fit_log_energy]
                    plot_monotel_reconstruction(fig, subarray, run_id, tel_id, event, image_moment_array, fit_cam_x, fit_cam_y, 'movie')
                    movie_simulation(fig, subarray, run_id, tel_id, event, fit_params, movie_lookup_table_pkl, movie_eigen_vectors_pkl)
                    display_a_movie(fig, subarray, run_id, tel_id, event, len(eco_image_1d), eco_movie_1d)

            evt_header = [training_sample_path,event_id,tel_id]
            evt_geometry = [image_size,lightcone,focal_length,image_direction,time_direction]
            evt_truth = [truth_energy,truth_alt,truth_az,star_cam_x,star_cam_y,truth_projection]
            evt_model = [pow(10.,fit_log_energy),fit_alt,fit_az,fit_cam_x,fit_cam_y,image_fit_chi2]

            single_analysis_result = [evt_header,evt_geometry,evt_truth,evt_model]
            analysis_result += [single_analysis_result]

    if select_event_id!=0: 
        print ('No file saved.')
        return

    output_filename = f'{ctapipe_output}/output_analysis/{ana_tag}_run{run_id}.pkl'
    print (f'writing file to {output_filename}')
    with open(output_filename,"wb") as file:
        pickle.dump(analysis_result, file)

    return

training_sample_path = sys.argv[1]

ctapipe_output = os.environ.get("CTAPIPE_OUTPUT_PATH")
#subprocess.call(['sh', './clean_plots.sh'])

run_monotel_analysis(training_sample_path,min_energy=0.1,max_energy=100.0,max_evt=1e10)
print ('Job completed.')
exit()

