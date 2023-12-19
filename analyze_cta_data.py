
import os, sys
import subprocess
import glob

import time
import math
import numpy as np
from astropy import units as u
from scipy.optimize import least_squares, minimize, brute, dual_annealing
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
from load_cta_data import signle_image_reconstruction

ctapipe_output = os.environ.get("CTAPIPE_OUTPUT_PATH")
#subprocess.call(['sh', './clean_plots.sh'])

font = {'family': 'serif', 'color':  'white', 'weight': 'normal', 'size': 10, 'rotation': 0.,}

fig, ax = plt.subplots()
figsize_x = 8.6
figsize_y = 6.4
fig.set_figheight(figsize_y)
fig.set_figwidth(figsize_x)

image_size_cut = 100.
open_angle_threshold = 0.4

def find_intersection_on_focalplane(image_x,image_y,image_a,image_b,image_w):

    pair_x = []
    pair_y = []
    pair_w = []
    for img1 in range(0,len(image_a)-1):
        for img2 in range(img1+1,len(image_a)):
            a1 = image_a[img1]
            b1 = image_b[img1]
            a2 = image_a[img2]
            b2 = image_b[img2]
            x1 = image_x[img1]
            y1 = image_y[img1]
            x2 = image_x[img2]
            y2 = image_y[img2]
            if a1-a2==0.:
                x = 0.
                y = 0.
                w = 0.
            else:
                x = (b2 - b1) / (a1 - a2)
                y = a1 * ((b2 - b1) / (a1 - a2)) + b1
                w = min(image_w[img1],image_w[img2])*abs(np.arctan(abs((a1-a2)/(1.+a1*a2))))/open_angle_threshold
            pair_x += [x]
            pair_y += [y]
            pair_w += [w]

    avg_x = 0.
    avg_y = 0.
    sum_w = 0.
    for sol in range(0,len(pair_x)):
        sum_w += pair_w[sol]
        avg_x += pair_w[sol]*pair_x[sol]
        avg_y += pair_w[sol]*pair_y[sol]

    if sum_w==0.:
        return 0, 0, 0
    avg_x = avg_x/sum_w
    avg_y = avg_y/sum_w

    return avg_x, avg_y, sum_w

def find_single_pair_intersection_on_ground(xp1, yp1, phi1, xp2, yp2, phi2):
    """
    Perform intersection of two lines. This code is borrowed from read_hess.

    Parameters
    ----------
    xp1: ndarray
        X position of first image
    yp1: ndarray
        Y position of first image
    phi1: ndarray
        Rotation angle of first image
    xp2: ndarray
        X position of second image
    yp2: ndarray
        Y position of second image
    phi2: ndarray
        Rotation angle of second image

    Returns
    -------
    ndarray of x and y crossing points for all pairs
    """
    sin_1 = np.sin(phi1)
    cos_1 = np.cos(phi1)
    a1 = sin_1
    b1 = -1 * cos_1
    c1 = yp1 * cos_1 - xp1 * sin_1

    sin_2 = np.sin(phi2)
    cos_2 = np.cos(phi2)

    a2 = sin_2
    b2 = -1 * cos_2
    c2 = yp2 * cos_2 - xp2 * sin_2

    det_ab = a1 * b2 - a2 * b1
    det_bc = b1 * c2 - b2 * c1
    det_ca = c1 * a2 - c2 * a1

    if  math.fabs(det_ab) < 1e-14 : # /* parallel */
       return 0,0
    xs = det_bc / det_ab
    ys = det_ca / det_ab

    return xs, ys

def find_intersection_on_ground(image_x,image_y,image_a,image_b,image_w):

    shower_x = []
    shower_y = []
    shower_w = []
    for img1 in range(0,len(image_a)-1):
        for img2 in range(img1+1,len(image_a)):
            a1 = image_a[img1]
            b1 = image_b[img1]
            a2 = image_a[img2]
            b2 = image_b[img2]
            x1 = image_x[img1]
            y1 = image_y[img1]
            x2 = image_x[img2]
            y2 = image_y[img2]
            #w = np.arctan(abs((a1-a2)/(1.+a1*a2)))
            w = pow(x1-x2,2)+pow(y1-y2,2)
            phi1 = np.arctan(a1)
            phi2 = np.arctan(a2)
            sx, sy = find_single_pair_intersection_on_ground(x1, y1, phi1, x2, y2, phi2)
            if sx==0 and sy==0: w = 0.
            shower_x += [sx]
            shower_y += [sy]
            shower_w += [w]

    avg_sx = 0.
    avg_sy = 0.
    sum_w = 0.
    for sol in range(0,len(shower_x)):
        sum_w += shower_w[sol]
        avg_sx += shower_w[sol]*shower_x[sol]
        avg_sy += shower_w[sol]*shower_y[sol]
    if sum_w==0.:
        return 0, 0, 0
    avg_sx = avg_sx/sum_w
    avg_sy = avg_sy/sum_w

    return avg_sx, avg_sy, sum_w


def fit_image_to_line(image_input,x_axis,y_axis):

    num_rows, num_cols = image_input.shape
    x = []
    y = []
    w = []
    for x_idx in range(0,num_cols):
        for y_idx in range(0,num_rows):
            x += [x_axis[x_idx]]
            y += [y_axis[y_idx]]
            w += [pow(image_input[y_idx,x_idx],1.0)]
    x = np.array(x)
    y = np.array(y)
    w = np.array(w)

    if np.sum(w)==0.:
        return 0., 0., 0.

    avg_x = np.sum(w * x)/np.sum(w)
    avg_y = np.sum(w * y)/np.sum(w)

    # solve x*A = y using SVD
    x = []
    y = []
    w = []
    for x_idx in range(0,num_cols):
        for y_idx in range(0,num_rows):
            if image_input[y_idx,x_idx]==0.: continue
            x += [[x_axis[x_idx]-avg_x]]
            y += [[y_axis[y_idx]-avg_y]]
            w += [pow(image_input[y_idx,x_idx],1.0)]
    x = np.array(x)
    y = np.array(y)
    w = np.diag(w)

    # Compute the weighted SVD
    U, S, Vt = np.linalg.svd(w @ x, full_matrices=False)
    # Calculate the weighted pseudo-inverse
    S_pseudo_w = np.diag(1 / S)
    x_pseudo_w = Vt.T @ S_pseudo_w @ U.T
    # Compute the weighted least-squares solution
    A_svd = x_pseudo_w @ (w @ y)
    # Compute chi2
    chi2 = np.linalg.norm((w @ x).dot(A_svd)-(w @ y), 2)/np.trace(w)

    a = A_svd[0]
    b = avg_y - a*avg_x

    #return a, b, 1./chi2
    return float(a), float(b), float(np.trace(w)/chi2)

def get_cam_coord_axes(geom,image_2d):

    num_rows, num_cols = image_2d.shape
    max_pix_x = float(max(geom.pix_x)/u.m)
    min_pix_x = float(min(geom.pix_x)/u.m)
    max_pix_y = float(max(geom.pix_y)/u.m)
    min_pix_y = float(min(geom.pix_y)/u.m)
    delta_pix_x = (max_pix_x-min_pix_x)/float(num_cols)
    delta_pix_y = (max_pix_y-min_pix_y)/float(num_rows)
    x_axis = []
    y_axis = []
    for x_idx in range(0,num_cols):
        x_axis += [min_pix_x+x_idx*delta_pix_x]
    for y_idx in range(0,num_rows):
        y_axis += [max_pix_y-y_idx*delta_pix_y]

    return x_axis, y_axis

def convert_array_coord_to_tel_coord(array_coord,tel_info):

    tel_alt = tel_info[0]
    tel_az = tel_info[1]
    tel_x = tel_info[2]
    tel_y = tel_info[3]
    tel_focal_length = tel_info[4]

    shower_alt = array_coord[0]
    shower_az = array_coord[1]
    shower_core_x = array_coord[2]
    shower_core_y = array_coord[3]

    cam_x = (shower_alt-tel_alt)*tel_focal_length
    cam_y = (shower_az-tel_az)*tel_focal_length
    impact_x = shower_core_x - tel_x
    impact_y = shower_core_y - tel_y
    return [cam_x, cam_y, impact_x, impact_y]

def convert_tel_coord_to_array_coord(tel_coord,tel_info):

    tel_alt = tel_info[0]
    tel_az = tel_info[1]
    tel_x = tel_info[2]
    tel_y = tel_info[3]
    tel_focal_length = tel_info[4]

    cam_x = tel_coord[0]
    cam_y = tel_coord[1]

    shower_alt = cam_x/tel_focal_length + tel_alt
    shower_az = cam_y/tel_focal_length + tel_az
    return [shower_alt, shower_az]

def get_image_center(evt_image_2d,x_axis,y_axis):

    num_rows, num_cols = evt_image_2d.shape
    avg_x = 0.
    avg_y = 0.
    sum_w = 0.
    for x_idx in range(0,num_cols):
        for y_idx in range(0,num_rows):
            weight = evt_image_2d[y_idx,x_idx]
            avg_x += x_axis[x_idx]*weight
            avg_y += y_axis[y_idx]*weight
            sum_w += weight
    avg_x = avg_x/sum_w
    avg_y = avg_y/sum_w
    return avg_x, avg_y, sum_w

def create_svd_image(shower_params,tel_pos,cam_axes,geom,lookup_table,lookup_table_impact,eigen_vectors):

    shower_cam_x = float(shower_params[0])
    shower_cam_y = float(shower_params[1])
    shower_core_x = float(shower_params[2])
    shower_core_y = float(shower_params[3])
    shower_log_energy = float(shower_params[4])
    #print (f'shower_log_energy = {shower_log_energy}')

    tel_x = tel_pos[0]/1000.
    tel_y = tel_pos[1]/1000.
    x_axis = cam_axes[0]
    y_axis = cam_axes[1]
    shower_impact_x = shower_core_x - tel_x
    shower_impact_y = shower_core_y - tel_y

    shower_impact = pow(pow(shower_impact_x,2)+pow(shower_impact_y,2),0.5)
    #print (f'shower_impact = {shower_impact}')

    latent_space = []
    for r in range(0,len(lookup_table)):
        latent_space += [lookup_table[r].get_bin_content(shower_impact,shower_log_energy)]
    latent_space = np.array(latent_space)
    #print (f'latent_space = {latent_space}')

    arrival = lookup_table_impact.get_bin_content(shower_impact,shower_log_energy)
    #print (f'arrival = {arrival}')
    
    evt_image = eigen_vectors.T @ latent_space
    evt_image_square = geom.image_to_cartesian_representation(evt_image)

    num_rows, num_cols = evt_image_square.shape

    for x_idx in range(0,num_cols):
        for y_idx in range(0,num_rows):
            if math.isnan(evt_image_square[y_idx,x_idx]): evt_image_square[y_idx,x_idx] = 0.
            if evt_image_square[y_idx,x_idx]<0.: evt_image_square[y_idx,x_idx] = 0.

    evt_image_shift = np.zeros_like(evt_image_square)
    shift_x = -arrival
    shift_y = 0.
    evt_image_shift = image_translation(evt_image_square, cam_axes[0], cam_axes[1], shift_x, shift_y)

    evt_image_derotate = np.zeros_like(evt_image_square)
    angle_rad = np.arctan2(shower_impact_y,shower_impact_x)
    #print (f'angle_rad = {angle_rad}')
    if math.isnan(angle_rad): angle_rad = 0.
    evt_image_derotate = image_rotation(evt_image_shift, cam_axes[0], cam_axes[1], angle_rad)

    evt_image_decenter = np.zeros_like(evt_image_square)
    shift_x = shower_cam_x
    shift_y = shower_cam_y
    evt_image_decenter = image_translation(evt_image_derotate, cam_axes[0], cam_axes[1], shift_x, shift_y)

    #print ('===============================================')
    return evt_image_decenter

    #analysis_image_1d = geom.image_from_cartesian_representation(evt_image_decenter)
    #return analysis_image_1d


def shower_image_chi2(data_image_matrix,sim_image_matrix):

    n_rows, n_cols = np.array(data_image_matrix).shape
    chi2 = 0.
    sum_data_cnt = 1.
    sum_sim_cnt = 1.
    #sum_data_cnt = 0.
    #sum_sim_cnt = 0.
    #for row in range(0,n_rows):
    #    for col in range(0,n_cols):
    #        data_cnt = data_image_matrix[row][col]
    #        sim_cnt = sim_image_matrix[row][col]
    #        sum_data_cnt += data_cnt
    #        sum_sim_cnt += sim_cnt
    for row in range(0,n_rows):
        for col in range(0,n_cols):
            data_cnt = data_image_matrix[row][col]/sum_data_cnt
            sim_cnt = sim_image_matrix[row][col]/sum_sim_cnt
            chi2 += pow((data_cnt-sim_cnt),2)
    return chi2

def single_shower_image_chi2(data_image_matrix,sim_image_matrix):

    n_rows = len(data_image_matrix)
    chi2 = 0.
    sum_data_cnt = 0.
    sum_sim_cnt = 0.
    for row in range(0,n_rows):
        data_cnt = data_image_matrix[row]
        sim_cnt = sim_image_matrix[row]
        chi2 += pow((data_cnt-sim_cnt),2)
        sum_data_cnt += data_cnt
        sum_sim_cnt += sim_cnt
    return chi2

def shower_image_correlation(data_image_matrix,sim_image_matrix):

    n_rows, n_cols = np.array(data_image_matrix).shape
    correlation = 0.
    sum_data_cnt = 0.
    sum_sim_cnt = 0.
    for row in range(0,n_rows):
        for col in range(0,n_cols):
            data_cnt = data_image_matrix[row][col]
            sim_cnt = sim_image_matrix[row][col]
            sum_data_cnt += data_cnt
            sum_sim_cnt += sim_cnt
            correlation += data_cnt*sim_cnt
    correlation = correlation/pow(sum_data_cnt*sum_data_cnt+sum_sim_cnt*sum_sim_cnt,0.5)
    return correlation

def single_shower_image_correlation(data_image_matrix,sim_image_matrix):

    n_rows = len(data_image_matrix)
    correlation = 0.
    sum_data_cnt = 0.
    sum_sim_cnt = 0.
    for row in range(0,n_rows):
        data_cnt = data_image_matrix[row]
        sim_cnt = sim_image_matrix[row]
        sum_data_cnt += data_cnt
        sum_sim_cnt += sim_cnt
        correlation += data_cnt*sim_cnt
    correlation = correlation/pow(sum_data_cnt*sum_data_cnt+sum_sim_cnt*sum_sim_cnt,0.5)
    return correlation

def weighted_distance_to_line(image_2d,a,b,cam_geom):

    n_rows, n_cols = np.array(image_2d).shape
    max_pix_x = max(cam_geom.pix_x)/u.m
    min_pix_x = min(cam_geom.pix_x)/u.m
    max_pix_y = max(cam_geom.pix_y)/u.m
    min_pix_y = min(cam_geom.pix_y)/u.m
    delta_pix_x = (max_pix_x-min_pix_x)/float(n_cols)
    delta_pix_y = (max_pix_y-min_pix_y)/float(n_rows)
    x_axis = []
    y_axis = []
    for x_idx in range(0,n_cols):
        x_axis += [min_pix_x+x_idx*delta_pix_x]
    for y_idx in range(0,n_rows):
        y_axis += [min_pix_y+y_idx*delta_pix_y]
    sum_dist_sq = 0.
    for row in range(0,n_rows):
        for col in range(0,n_cols):
            if math.isnan(image_2d[row,col]): continue
            if image_2d[row,col]==0.: continue
            y0 = y_axis[row]
            x0 = x_axis[col]
            dist_sq = pow((a*x0-y0+b),2)/(a*a+1)*image_2d[row,col]
            sum_dist_sq += dist_sq*image_2d[row,col]
    return sum_dist_sq

def weighted_distance_between_two_images(image_a, image_b, cam_geom, cam_axes):

    n_rows, n_cols = np.array(image_a).shape
    x_axis = cam_axes[0]
    y_axis = cam_axes[1]
    sum_a = 0.
    sum_b = 0.
    avg_a_x = 0.
    avg_a_y = 0.
    avg_b_x = 0.
    avg_b_y = 0.
    for row in range(0,n_rows):
        for col in range(0,n_cols):
            if math.isnan(image_a[row,col]): continue
            if math.isnan(image_b[row,col]): continue
            cnt_a = image_a[row,col]
            cnt_b = image_b[row,col]
            y0 = y_axis[row]
            x0 = x_axis[col]
            sum_a += cnt_a
            sum_b += cnt_b
            avg_a_x += x0*cnt_a
            avg_a_y += y0*cnt_a
            avg_b_x += x0*cnt_b
            avg_b_y += y0*cnt_b
    avg_a_x = avg_a_x/sum_a
    avg_a_y = avg_a_y/sum_a
    avg_b_x = avg_b_x/sum_b
    avg_b_y = avg_b_y/sum_b
    dist_sq = pow(avg_a_x-avg_b_x,2)+pow(avg_a_y-avg_b_y,2)
    return dist_sq

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sum_square_difference_between_images_1d(try_params,image_1d_matrix,time_1d_matrix,image_center,cam_axes,geom,lookup_table,eigenvectors,lookup_table_time,eigenvectors_time,lookup_table_impact):

    try_shower_cam_x = try_params[0]
    try_shower_cam_y = try_params[1]
    try_log_energy = try_params[2]


    tic = time.perf_counter()
    chi2 = 0.
    for tel in range(0,len(image_1d_matrix)):
        img_x = image_center[tel][0]
        img_y = image_center[tel][1]

        image_data_1d = image_1d_matrix[tel]
        time_data_1d = time_1d_matrix[tel]

        #image_weight = 1./np.sum(np.array(image_data_1d)*np.array(image_data_1d))
        #time_weight = 1./np.sum(np.array(time_data_1d)*np.array(time_data_1d))
        image_weight = 1.
        time_weight = 1.
        #time_weight = 1.*np.sum(np.array(image_data_1d)*np.array(image_data_1d))/np.sum(np.array(time_data_1d)*np.array(time_data_1d))

        try_arrival = pow(pow(img_x-try_shower_cam_x,2)+pow(img_y-try_shower_cam_y,2),0.5)
        init_params = [try_log_energy,try_arrival]
        image_chi2 = image_weight*sqaure_difference_between_1d_images(init_params,cam_axes,geom,image_data_1d,lookup_table,eigenvectors)
        time_chi2 = time_weight*sqaure_difference_between_1d_images(init_params,cam_axes,geom,time_data_1d,lookup_table_time,eigenvectors_time)
        #print (f'image_chi2 = {image_chi2}')
        #print (f'time_chi2 = {time_chi2}')

        chi2 += image_chi2
        #chi2 += time_chi2

    toc = time.perf_counter()
    #print (f'SVD image trial completed in {toc - tic:0.4f} seconds')
    #exit()
    #print (f'chi2 = {chi2}')

    return chi2

def sum_square_difference_between_images_2d(try_params,image_2d_matrix,time_2d_matrix,tel_pos_matrix,cam_axes,geom,lookup_table,eigenvectors,lookup_table_time,eigenvectors_time,lookup_table_impact,fit_shower_core_x,fit_shower_core_y,fit_log_energy):

    try_shower_cam_x = try_params[0]
    try_shower_cam_y = try_params[1]

    tic = time.perf_counter()
    chi2 = 0.
    for tel in range(0,len(image_2d_matrix)):
        tel_x = tel_pos_matrix[tel][2]
        tel_y = tel_pos_matrix[tel][3]
        tel_pos = [tel_x,tel_y]

        evt_param = [try_shower_cam_x,try_shower_cam_y,fit_shower_core_x,fit_shower_core_y,fit_log_energy]
        evt_param = np.array(evt_param)
        svd_image_2d = create_svd_image(evt_param,tel_pos,cam_axes[tel],geom,lookup_table,lookup_table_impact,eigenvectors)

        svd_image_2d = np.array(svd_image_2d)
        data_image_2d = np.array(image_2d_matrix[tel])

        chi2_image_2d = pow(data_image_2d-svd_image_2d,2)
        chi2 += np.sum(chi2_image_2d)


    toc = time.perf_counter()
    #print (f'SVD image trial completed in {toc - tic:0.4f} seconds')
    #exit()

    return chi2

def sum_square_distance_to_image_lines_and_points(core_xy,list_tel_x,list_tel_y,list_line_a,list_line_w,list_core_x,list_core_y,list_point_w):

    core_x = core_xy[0]
    core_y = core_xy[1]

    #open_angle_sq = 0.
    #for t1 in range(0,len(list_line_a)-1):
    #    for t2 in range(t1+1,len(list_line_a)):
    #        a1 = float(list_line_a[t1])
    #        a2 = float(list_line_a[t2])
    #        open_angle_sq += pow(np.arctan(abs((a1-a2)/(1.+a1*a2))),2)

    sum_dist_sq_line = 0.
    for tel in range(0,len(list_line_w)):
        tel_x = list_tel_x[tel]
        tel_y = list_tel_y[tel]
        a = list_line_a[tel]
        b = tel_y-tel_x*a
        dist_sq = distance_square_point_to_line(core_x,core_y,a,b)
        sum_dist_sq_line += dist_sq*pow(list_line_w[tel],2)
        #sum_dist_sq_line += dist_sq

    sum_dist_sq_point = 0.
    for tel in range(0,len(list_point_w)):
        dist_sq = pow(core_x-list_core_x[tel],2) + pow(core_y-list_core_y[tel],2)
        sum_dist_sq_point += dist_sq*pow(list_point_w[tel],2)
        #sum_dist_sq_point += dist_sq

    #print (f'open_angle_sq = {open_angle_sq}')
    #print (f'sum_dist_sq_line = {sum_dist_sq_line}')
    #print (f'sum_dist_sq_point = {sum_dist_sq_point}')
    #exit()

    line_weight = 1.
    point_weight = 1.
    sum_dist_sq = line_weight*sum_dist_sq_line + point_weight*sum_dist_sq_point
    return sum_dist_sq

def fit_templates_to_all_images(image_matrix,time_matrix,tel_pos_matrix,geom,cam_axes,lookup_table,eigen_vectors,lookup_table_time,eigen_vectors_time,lookup_table_impact,lookup_table_impact_rms):

    fit_line_cam_x, fit_line_cam_y, fit_line_core_x, fit_line_core_y, open_angle = fit_lines_to_all_images(image_matrix,tel_pos_matrix,geom,cam_axes)
    list_img_a, list_img_b, list_img_w, list_line_cam_x, list_line_cam_y, list_line_core_x, list_line_core_y, list_line_weight = fit_intersections_all_images(image_matrix,tel_pos_matrix,geom,cam_axes)
    print (f'open_angle = {open_angle}')
    open_angle_cut = 0.1

    tic = time.perf_counter()

    image_2d_matrix = []
    time_2d_matrix = []
    for tel in range(0,len(image_matrix)):

        image_data_1d = image_matrix[tel]
        time_data_1d = time_matrix[tel]
        image_data_2d = geom.image_to_cartesian_representation(image_data_1d)
        time_data_2d = geom.image_to_cartesian_representation(time_data_1d)
        num_rows, num_cols = image_data_2d.shape
        for x_idx in range(0,num_cols):
            for y_idx in range(0,num_rows):
                if math.isnan(image_data_2d[y_idx,x_idx]): 
                    image_data_2d[y_idx,x_idx] = 0.
                    time_data_2d[y_idx,x_idx] = 0.
        image_2d_matrix += [image_data_2d]
        time_2d_matrix += [time_data_2d]

    list_svd_cam_x = []
    list_svd_cam_y = []
    list_svd_core_x = []
    list_svd_core_y = []
    list_svd_weight = []
    list_img_center_x = []
    list_img_center_y = []
    list_img_foci_x = []
    list_img_foci_y = []
    list_tel_x = []
    list_tel_y = []
    list_image_2d = []
    avg_energy = 0.
    sum_weight = 0.
    for tel in range(0,len(image_matrix)):

        tel_x = tel_pos_matrix[tel][2]/1000.
        tel_y = tel_pos_matrix[tel][3]/1000.

        image_data_2d = image_2d_matrix[tel]
        time_data_2d = time_2d_matrix[tel]
        image_size = np.sum(image_data_2d)

        if image_size<image_size_cut: continue

        guided=True
        arrival_x=fit_line_cam_x
        arrival_y=fit_line_cam_y
        if open_angle<open_angle_cut and tel==0:
            guided=False

        image_center_x, image_center_y, image_foci_x1, image_foci_y1, image_foci_x2, image_foci_y2, center_time, delta_foci_time, semi_major_sq, semi_minor_sq = find_image_moments_guided(image_data_2d, time_data_2d, cam_axes[tel][0], cam_axes[tel][1], guided=guided, arrival_x=arrival_x, arrival_y=arrival_y)
        image_foci_x = image_foci_x1
        image_foci_y = image_foci_y1

        list_img_center_x += [image_center_x]
        list_img_center_y += [image_center_y]
        list_img_foci_x += [image_foci_x]
        list_img_foci_y += [image_foci_y]
        list_tel_x += [tel_x]
        list_tel_y += [tel_y]
        list_image_2d += [image_data_2d]

        fit_arrival, fit_impact, fit_log_energy = signle_image_reconstruction(image_matrix[tel],time_matrix[tel],geom,cam_axes[tel],lookup_table,eigen_vectors,lookup_table_time,eigen_vectors_time,lookup_table_impact,lookup_table_impact_rms, guided=guided, arrival_x=arrival_x, arrival_y=arrival_y)

        angle_rad = np.arctan2(image_foci_y-image_center_y,image_foci_x-image_center_x)
        delta_x = fit_arrival*np.cos(-angle_rad)
        delta_y = fit_arrival*np.sin(-angle_rad)
        cam_x = image_center_x + delta_x
        cam_y = image_center_y - delta_y
        line_a = delta_y/delta_x
        line_b_sky = image_center_y-line_a*image_center_x

        if open_angle<open_angle_cut and tel==0:
            arrival_x=cam_x
            arrival_y=cam_y

        line_b_grd = tel_y-line_a*tel_x
        delta_x = fit_impact*np.cos(-angle_rad)
        delta_y = fit_impact*np.sin(-angle_rad)
        core_x = tel_x + delta_x
        core_y = tel_y - delta_y

        sum_weight += image_size
        avg_energy += pow(10.,fit_log_energy)*image_size
        list_svd_cam_x += [cam_x]
        list_svd_cam_y += [cam_y]
        list_svd_core_x += [core_x]
        list_svd_core_y += [core_y]
        list_svd_weight += [image_size]


    list_cam_x = list_svd_cam_x + list_line_cam_x
    list_cam_y = list_svd_cam_y + list_line_cam_y
    list_core_x = list_svd_core_x + list_line_core_x
    list_core_y = list_svd_core_y + list_line_core_y
    list_weight = list_svd_weight + list_line_weight
    #list_cam_x = list_svd_cam_x
    #list_cam_y = list_svd_cam_y
    #list_core_x = list_svd_core_x
    #list_core_y = list_svd_core_y
    #list_weight = list_svd_weight

    avg_energy = avg_energy/sum_weight

    #try_params = [fit_line_core_x,fit_line_core_y]
    #solution = minimize(
    #    sum_square_distance_to_points,
    #    x0=try_params,
    #    args=(list_core_x,list_core_y,list_weight),
    #    method='L-BFGS-B',
    #)
    #avg_core_x = solution['x'][0]
    #avg_core_y = solution['x'][1]

    #try_params = [fit_line_cam_x,fit_line_cam_y]
    #solution = minimize(
    #    sum_square_distance_to_points,
    #    x0=try_params,
    #    args=(list_cam_x,list_cam_y,list_weight),
    #    method='L-BFGS-B',
    #)
    #avg_cam_x = solution['x'][0]
    #avg_cam_y = solution['x'][1]

    total_weight = 0.
    avg_cam_x = 0.
    avg_cam_y = 0.
    avg_core_x = 0.
    avg_core_y = 0.
    max_weight = max(list_weight)
    for entry in range(0,len(list_cam_x)):
        if list_weight[entry]<0.5*max_weight: continue
        total_weight += list_weight[entry]
        avg_cam_x += list_cam_x[entry]*list_weight[entry]
        avg_cam_y += list_cam_y[entry]*list_weight[entry]
        avg_core_x += list_core_x[entry]*list_weight[entry]
        avg_core_y += list_core_y[entry]*list_weight[entry]
    avg_cam_x = avg_cam_x/total_weight
    avg_cam_y = avg_cam_y/total_weight
    avg_core_x = avg_core_x/total_weight
    avg_core_y = avg_core_y/total_weight


    #if len(image_matrix)>10:
    #    return avg_energy, avg_cam_x, avg_cam_y, avg_core_x, avg_core_y
    return avg_energy, avg_cam_x, avg_cam_y, avg_core_x, avg_core_y


    image_repos_1d_matrix = []
    time_repos_1d_matrix = []
    list_img_center = []
    for tel in range(0,len(image_matrix)):

        image_data_2d = image_2d_matrix[tel]
        time_data_2d = time_2d_matrix[tel]

        guided=True
        arrival_x=avg_cam_x
        arrival_y=avg_cam_y

        image_center_x, image_center_y, image_foci_x1, image_foci_y1, image_foci_x2, image_foci_y2, center_time, delta_foci_time, semi_major_sq, semi_minor_sq = find_image_moments_guided(image_data_2d, time_data_2d, cam_axes[tel][0], cam_axes[tel][1], guided=guided, arrival_x=arrival_x, arrival_y=arrival_y)
        image_foci_x = image_foci_x1
        image_foci_y = image_foci_y1

        list_img_center += [[image_center_x,image_center_y]]

        image_recenter = np.zeros_like(image_data_2d)
        time_recenter = np.zeros_like(time_data_2d)
        shift_x = -image_center_x
        shift_y = -image_center_y
        image_recenter = image_translation(image_data_2d, cam_axes[tel][0], cam_axes[tel][1], shift_x, shift_y)
        time_recenter = image_translation(time_data_2d, cam_axes[tel][0], cam_axes[tel][1], shift_x, shift_y)

        image_rotate = np.zeros_like(image_data_2d)
        time_rotate = np.zeros_like(time_data_2d)
        angle_rad = -np.arctan2(image_foci_y-image_center_y,image_foci_x-image_center_x)
        image_rotate = image_rotation(image_recenter, cam_axes[tel][0], cam_axes[tel][1], angle_rad)
        time_rotate = image_rotation(time_recenter, cam_axes[tel][0], cam_axes[tel][1], angle_rad)
        
        image_rotate_1d = geom.image_from_cartesian_representation(image_rotate)
        time_rotate_1d = geom.image_from_cartesian_representation(time_rotate)
        image_repos_1d_matrix += [image_rotate_1d]
        time_repos_1d_matrix += [time_rotate_1d]




    fit_shower_energy = avg_energy

    fit_shower_cam_x = fit_line_cam_x
    fit_shower_cam_y = fit_line_cam_y
    fit_shower_core_x = fit_line_core_x
    fit_shower_core_y = fit_line_core_y
    #fit_shower_cam_x = avg_cam_x
    #fit_shower_cam_y = avg_cam_y
    #fit_shower_core_x = avg_core_x
    #fit_shower_core_y = avg_core_y

    core_error = 0.1
    stepsize = [0.01,0.01,0.1]
    try_params = [fit_shower_cam_x,fit_shower_cam_y,math.log10(fit_shower_energy)]
    bounds = [(fit_shower_cam_x-0.05,fit_shower_cam_x+0.05),(fit_shower_cam_y-0.05,fit_shower_cam_y+0.05),(math.log10(fit_shower_energy)-0.3,math.log10(fit_shower_energy)+0.3)]
    solution = minimize(
        sum_square_difference_between_images_1d,
        x0=try_params,
        args=(image_repos_1d_matrix,time_repos_1d_matrix,list_img_center,cam_axes,geom,lookup_table,eigen_vectors,lookup_table_time,eigen_vectors_time,lookup_table_impact),
        bounds=bounds,
        method='L-BFGS-B',
        jac=None,
        options={'eps':stepsize,'ftol':0.01},
    )
    fit_shower_cam_x = solution['x'][0]
    fit_shower_cam_y = solution['x'][1]
    fit_shower_log_energy = solution['x'][2]
    fit_shower_energy = pow(10.,solution['x'][2])
    #exit()

    #stepsize = [0.005,0.005]
    #try_params = [fit_shower_cam_x,fit_shower_cam_y]
    #solution = minimize(
    #    sum_square_difference_between_images_2d,
    #    x0=try_params,
    #    args=(image_2d_matrix,time_2d_matrix,tel_pos_matrix,cam_axes,geom,lookup_table,eigen_vectors,lookup_table_time,eigen_vectors_time,lookup_table_impact,fit_shower_core_x,fit_shower_core_y,fit_shower_log_energy),
    #    method='L-BFGS-B',
    #    jac=None,
    #    options={'eps':stepsize,'ftol':0.01},
    #)
    #fit_shower_cam_x = solution['x'][0]
    #fit_shower_cam_y = solution['x'][1]
    #print (f'fit_line_cam_x = {fit_line_cam_x}')
    #print (f'fit_line_cam_y = {fit_line_cam_y}')
    #print (f'fit_shower_cam_x = {fit_shower_cam_x}')
    #print (f'fit_shower_cam_y = {fit_shower_cam_y}')

    #fit_shower_cam_x = 0.
    #fit_shower_cam_y = 0.
    #total_weight = 0.
    #for tel in range(0,len(list_tel_x)):

    #    tel_x = list_tel_x[tel]
    #    tel_y = list_tel_y[tel]

    #    image_size = list_svd_weight[tel]

    #    fit_impact = pow(pow(tel_x-fit_shower_core_x,2)+pow(tel_y-fit_shower_core_y,2),0.5)

    #    image_center_x = list_img_center_x[tel]
    #    image_center_y = list_img_center_y[tel]
    #    image_foci_x = list_img_foci_x[tel]
    #    image_foci_y = list_img_foci_y[tel]

    #    fit_arrival = lookup_table_impact.get_bin_content(fit_impact,np.log10(fit_shower_energy))

    #    angle_rad = np.arctan2(image_foci_y-image_center_y,image_foci_x-image_center_x)
    #    delta_x = fit_arrival*np.cos(-angle_rad)
    #    delta_y = fit_arrival*np.sin(-angle_rad)
    #    cam_x = image_center_x + delta_x
    #    cam_y = image_center_y - delta_y

    #    fit_shower_cam_x += cam_x*image_size
    #    fit_shower_cam_y += cam_y*image_size
    #    total_weight += image_size

    #fit_shower_cam_x = fit_shower_cam_x/total_weight
    #fit_shower_cam_y = fit_shower_cam_y/total_weight

    #sky_xy_log_likelihood = np.zeros_like(image_2d_matrix[0])
    #num_rows, num_cols = sky_xy_log_likelihood.shape
    #for x_idx in range(0,num_cols):
    #    for y_idx in range(0,num_rows):
    #        sky_xy_log_likelihood[y_idx,x_idx] = 0.
    #sky_likelihood_xaxis = cam_axes[0][0]
    #sky_likelihood_yaxis = cam_axes[0][1]

    #for tel in range(0,len(list_tel_x)):
    #    tel_x = list_tel_x[tel]
    #    tel_y = list_tel_y[tel]

    #    image_size = list_svd_weight[tel]

    #    fit_impact = pow(pow(tel_x-fit_shower_core_x,2)+pow(tel_y-fit_shower_core_y,2),0.5)

    #    image_center_x = list_img_center_x[tel]
    #    image_center_y = list_img_center_y[tel]
    #    image_foci_x = list_img_foci_x[tel]
    #    image_foci_y = list_img_foci_y[tel]

    #    fit_arrival = lookup_table_impact.get_bin_content(fit_impact,np.log10(fit_shower_energy))

    #    angle_rad = np.arctan2(image_foci_y-image_center_y,image_foci_x-image_center_x)
    #    delta_x = fit_arrival*np.cos(-angle_rad)
    #    delta_y = fit_arrival*np.sin(-angle_rad)
    #    cam_x = image_center_x + delta_x
    #    cam_y = image_center_y - delta_y

    #    longi_error = 0.02
    #    trans_error = 0.002

    #    a, b, w = fit_image_to_line(list_image_2d[tel],cam_axes[tel][0],cam_axes[tel][1])
    #    aT, bT, wT = fit_image_to_line(np.array(list_image_2d[tel]).transpose(),cam_axes[tel][1],cam_axes[tel][0])
    #    if w<wT:
    #        a = 1./aT
    #        b = -bT/aT
    #        w = wT
    #    for x_idx in range(0,num_cols):
    #        for y_idx in range(0,num_rows):
    #            pix_x = sky_likelihood_xaxis[x_idx]
    #            pix_y = sky_likelihood_yaxis[y_idx]
    #            dist2center_sq = pow(pix_x-cam_x,2) + pow(pix_y-cam_y,2)
    #            dist2line_sq = pow((a*pix_x-pix_y+b),2)/(1.+a*a)
    #            new_likelihood = ((-dist2center_sq/(2.*longi_error)) + (-dist2line_sq/(2.*trans_error)))
    #            sky_xy_log_likelihood[y_idx,x_idx] += new_likelihood

    #max_index = np.argmax(sky_xy_log_likelihood)
    #max_row, max_col = np.unravel_index(max_index, sky_xy_log_likelihood.shape)
    #max_log_likelihood = sky_xy_log_likelihood[max_row,max_col]
    #fit_shower_cam_x = sky_likelihood_xaxis[max_col]
    #fit_shower_cam_y = sky_likelihood_yaxis[max_row]
    
    print (f'fit_line_cam_x = {fit_line_cam_x}')
    print (f'fit_line_cam_y = {fit_line_cam_y}')
    print (f'avg_cam_x = {avg_cam_x}')
    print (f'avg_cam_y = {avg_cam_y}')
    print (f'fit_shower_cam_x = {fit_shower_cam_x}')
    print (f'fit_shower_cam_y = {fit_shower_cam_y}')

    toc = time.perf_counter()
    print (f'SVD image fit completed in {toc - tic:0.4f} seconds')
    #exit()

    return fit_shower_energy, fit_shower_cam_x, fit_shower_cam_y, fit_shower_core_x, fit_shower_core_y


def fit_intersections_all_images(image_matrix,tel_pos_matrix,geom,cam_axes):

    list_all_img_a = []
    list_all_img_b = []
    list_all_img_w = []
    for tel in range(0,len(image_matrix)):
        rad2deg = 180./np.pi
        tel_x = tel_pos_matrix[tel][2]/1000.
        tel_y = tel_pos_matrix[tel][3]/1000.

        cam_x_axis = np.array(cam_axes[tel][0])
        cam_y_axis = np.array(cam_axes[tel][1])

        tel_image_1d = image_matrix[tel]
        tel_image_2d = geom.image_to_cartesian_representation(tel_image_1d)

        num_rows, num_cols = tel_image_2d.shape
        for x_idx in range(0,num_cols):
            for y_idx in range(0,num_rows):
                if math.isnan(tel_image_2d[y_idx,x_idx]): tel_image_2d[y_idx,x_idx] = 0.

        a, b, w = fit_image_to_line(tel_image_2d,cam_x_axis,cam_y_axis)
        aT, bT, wT = fit_image_to_line(np.array(tel_image_2d).transpose(),cam_y_axis,cam_x_axis)
        if w<wT:
            a = 1./aT
            b = -bT/aT
            w = wT

        list_all_img_a += [a]
        list_all_img_b += [b]
        list_all_img_w += [np.sum(tel_image_1d)]


    avg_ls_evt_cam_x = 0.
    avg_ls_evt_cam_y = 0.
    avg_ls_evt_core_x = 0.
    avg_ls_evt_core_y = 0.
    ls_evt_weight = 0.
    list_cam_x = []
    list_cam_y = []
    list_core_x = []
    list_core_y = []
    list_weight = []
    for tel1 in range(0,len(image_matrix)-1):
        for tel2 in range(tel1+1,len(image_matrix)):

            rad2deg = 180./np.pi
            tel1_x = tel_pos_matrix[tel1][2]/1000.
            tel1_y = tel_pos_matrix[tel1][3]/1000.
            tel2_x = tel_pos_matrix[tel2][2]/1000.
            tel2_y = tel_pos_matrix[tel2][3]/1000.

            tel1_image_1d = image_matrix[tel1]
            tel2_image_1d = image_matrix[tel2]
            tel1_image_2d = geom.image_to_cartesian_representation(tel1_image_1d)
            tel2_image_2d = geom.image_to_cartesian_representation(tel2_image_1d)

            num_rows, num_cols = tel1_image_2d.shape
            for x_idx in range(0,num_cols):
                for y_idx in range(0,num_rows):
                    if math.isnan(tel1_image_2d[y_idx,x_idx]): tel1_image_2d[y_idx,x_idx] = 0.
                    if math.isnan(tel2_image_2d[y_idx,x_idx]): tel2_image_2d[y_idx,x_idx] = 0.


            list_tel_x = []
            list_tel_x += [tel1_x]
            list_tel_x += [tel2_x]
            list_tel_y = []
            list_tel_y += [tel1_y]
            list_tel_y += [tel2_y]
            list_img_a = []
            list_img_b = []
            list_img_w = []

            a1 = list_all_img_a[tel1]
            b1 = list_all_img_b[tel1]
            w1 = np.sum(tel1_image_1d)
            list_img_a += [a1]
            list_img_b += [b1]
            list_img_w += [w1]

            a2 = list_all_img_a[tel2]
            b2 = list_all_img_b[tel2]
            w2 = np.sum(tel2_image_1d)
            list_img_a += [a2]
            list_img_b += [b2]
            list_img_w += [w2]

            ls_cam_x, ls_cam_y, ls_sky_weight = find_intersection_on_focalplane(list_tel_x,list_tel_y,list_img_a,list_img_b,list_img_w)
            ls_core_x, ls_core_y, ls_core_weight = find_intersection_on_ground(list_tel_x,list_tel_y,list_img_a,list_img_b,list_img_w)

            list_cam_x += [float(ls_cam_x)]
            list_cam_y += [float(ls_cam_y)]
            list_core_x += [float(ls_core_x)]
            list_core_y += [float(ls_core_y)]
            list_weight += [float(ls_sky_weight)]

    return list_all_img_a, list_all_img_b, list_all_img_w, list_cam_x, list_cam_y, list_core_x, list_core_y, list_weight

def sum_square_distance_to_image_lines(core_xy,list_tel_x,list_tel_y,list_img_a,list_img_w):

    core_x = core_xy[0]
    core_y = core_xy[1]
    sum_dist_sq = 0.
    for tel in range(0,len(list_img_a)):
        tel_x = list_tel_x[tel]
        tel_y = list_tel_y[tel]
        a = list_img_a[tel]
        b = tel_y-tel_x*a
        dist_sq = distance_square_point_to_line(core_x,core_y,a,b)
        sum_dist_sq += dist_sq*pow(list_img_w[tel],2)
    return sum_dist_sq

def sum_square_distance_to_points(core_xy,list_point_x,list_point_y,list_point_w):

    core_x = core_xy[0]
    core_y = core_xy[1]

    sum_dist_sq_point = 0.
    for tel in range(0,len(list_point_w)):
        dist_sq = pow(core_x-list_point_x[tel],2) + pow(core_y-list_point_y[tel],2)
        sum_dist_sq_point += dist_sq*pow(list_point_w[tel],2)

    return sum_dist_sq_point


def fit_lines_to_all_images(image_matrix,tel_pos_matrix,geom,cam_axes):

    init_cam_x = 0.
    init_cam_y = 0.
    init_core_x = 0.
    init_core_y = 0.
    open_angle = 0.

    if len(image_matrix)==1:
        return init_cam_x, init_cam_y, init_core_x, init_core_y, open_angle

    list_img_a, list_img_b, list_img_w, list_cam_x, list_cam_y, list_core_x, list_core_y, list_weight = fit_intersections_all_images(image_matrix,tel_pos_matrix,geom,cam_axes)

    #try_params = [init_core_x,init_core_y]
    #solution = minimize(
    #    sum_square_distance_to_points,
    #    x0=try_params,
    #    args=(list_core_x,list_core_y,list_weight),
    #    method='L-BFGS-B',
    #)
    #fit_core_x = solution['x'][0]
    #fit_core_y = solution['x'][1]

    #try_params = [init_cam_x,init_cam_y]
    #solution = minimize(
    #    sum_square_distance_to_points,
    #    x0=try_params,
    #    args=(list_cam_x,list_cam_y,list_weight),
    #    method='L-BFGS-B',
    #)
    #fit_cam_x = solution['x'][0]
    #fit_cam_y = solution['x'][1]

    total_weight = 0.
    fit_cam_x = 0.
    fit_cam_y = 0.
    fit_core_x = 0.
    fit_core_y = 0.
    max_weight = max(list_weight)
    for entry in range(0,len(list_cam_x)):
        if list_weight[entry]<0.5*max_weight: continue
        total_weight += list_weight[entry]
        fit_cam_x += list_cam_x[entry]*list_weight[entry]
        fit_cam_y += list_cam_y[entry]*list_weight[entry]
        fit_core_x += list_core_x[entry]*list_weight[entry]
        fit_core_y += list_core_y[entry]*list_weight[entry]
    fit_cam_x = fit_cam_x/total_weight
    fit_cam_y = fit_cam_y/total_weight
    fit_core_x = fit_core_x/total_weight
    fit_core_y = fit_core_y/total_weight

    open_angle = 0.
    for t1 in range(0,len(list_img_a)-1):
        for t2 in range(t1+1,len(list_img_a)):
            a1 = list_img_a[t1]
            a2 = list_img_a[t2]
            open_angle += min(list_img_w[t1],list_img_w[t2])*abs(np.arctan(abs((a1-a2)/(1.+a1*a2))))/open_angle_threshold
            #open_angle += abs(np.arctan(abs((a1-a2)/(1.+a1*a2))))

    return fit_cam_x, fit_cam_y, fit_core_x, fit_core_y, open_angle


def construct_line_on_focalplane(cam_x,cam_y,impact_x,impact_y,transpose=False):

    if transpose:
        if impact_y==0.:
            return 0, 0
    else:
        if impact_x==0.:
            return 0, 0

    a = impact_y/impact_x
    b = cam_y - a * cam_x
    if transpose:
        a = impact_x/impact_y
        b = cam_x - a * cam_y

    return a, b

def distance_square_point_to_line(x0,y0,a,b):

    d = pow(a*x0 - y0 + b,2)/(1 + a*a)
    return float(d)


testing_sample_path = sys.argv[1]


print ('loading svd pickle data... ')
output_filename = f'{ctapipe_output}/output_machines/lookup_table.pkl'
lookup_table_pkl = pickle.load(open(output_filename, "rb"))

output_filename = f'{ctapipe_output}/output_machines/lookup_table_time.pkl'
lookup_table_time_pkl = pickle.load(open(output_filename, "rb"))

output_filename = f'{ctapipe_output}/output_machines/eigen_vectors.pkl'
eigen_vectors_pkl = pickle.load(open(output_filename, "rb"))

output_filename = f'{ctapipe_output}/output_machines/eigen_vectors_time.pkl'
eigen_vectors_time_pkl = pickle.load(open(output_filename, "rb"))

output_filename = f'{ctapipe_output}/output_machines/lookup_table_impact.pkl'
lookup_table_impact_pkl = pickle.load(open(output_filename, "rb"))

output_filename = f'{ctapipe_output}/output_machines/lookup_table_impact_rms.pkl'
lookup_table_impact_rms_pkl = pickle.load(open(output_filename, "rb"))



print ('analyzing test data... ')
collected_all_images = False
current_run = 0
current_event = 0
plot_count = 0
testing_image_matrix = []
testing_time_matrix = []
testing_param_matrix = []
testing_moment_matrix = []
truth_shower_position_matrix = []
hillas_shower_position_matrix = []
all_cam_axes = []
telesc_position_matrix = []

all_runs = []
all_events = []
all_truth_energy = []
all_hillas_valid = []
temp_fit_energy = []
hillas_sky_err = []
line_fit_sky_err = []
temp_fit_sky_err = []
all_open_angle = []
all_image_size = []
line_impact = []
temp_impact = []
line_fit_cam_x_err = []
temp_fit_cam_x_err = []
line_fit_cam_y_err = []
temp_fit_cam_y_err = []

testing_id_list = []
big_telesc_position_matrix = []
big_truth_shower_position_matrix = []
big_hillas_shower_position_matrix = []
test_cam_axes = []
big_testing_image_matrix = []
big_testing_time_matrix = []
big_testing_param_matrix = []
big_testing_moment_matrix = []

source = SimTelEventSource(testing_sample_path, focal_length_choice='EQUIVALENT')
subarray = source.subarray
ob_keys = source.observation_blocks.keys()
run_id = list(ob_keys)[0]
output_filename = f'{ctapipe_output}/output_samples/testing_sample_noisy_clean_origin_run{run_id}.pkl'
print (f'loading pickle trainging sample data: {output_filename}')
if not os.path.exists(output_filename):
    print (f'file does not exist.')
    exit()
training_sample = pickle.load(open(output_filename, "rb"))

testing_id_list += training_sample[0]
big_telesc_position_matrix += training_sample[1]
big_truth_shower_position_matrix += training_sample[2]
test_cam_axes += training_sample[3]
big_testing_image_matrix += training_sample[4]
big_testing_time_matrix += training_sample[5]
big_testing_param_matrix += training_sample[6]
big_testing_moment_matrix += training_sample[7]
big_hillas_shower_position_matrix += training_sample[8]

big_testing_image_matrix = np.array(big_testing_image_matrix)
big_testing_time_matrix = np.array(big_testing_time_matrix)


for img in range(0,len(testing_id_list)):

    current_run = testing_id_list[img][0]
    current_event = testing_id_list[img][1]
    current_tel_id = testing_id_list[img][2]
    subarray = testing_id_list[img][3]
    geom = subarray.tel[current_tel_id].camera.geometry

    if img+1 == len(testing_id_list):
        collected_all_images = True
    elif current_run != testing_id_list[img+1][0] or current_event != testing_id_list[img+1][1]:
        collected_all_images = True

    if not collected_all_images:
        testing_image_matrix += [big_testing_image_matrix[img]]
        testing_time_matrix += [big_testing_time_matrix[img]]
        testing_param_matrix += [big_testing_param_matrix[img]]
        testing_moment_matrix += [big_testing_moment_matrix[img]]
        truth_shower_position_matrix += [big_truth_shower_position_matrix[img]]
        hillas_shower_position_matrix += [big_hillas_shower_position_matrix[img]]
        all_cam_axes += [test_cam_axes[img]]
        telesc_position_matrix += [big_telesc_position_matrix[img]]
    else:
        collected_all_images = False

        print ('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print (f'collected all images.')
        print (f'current_run = {current_run}')
        print (f'current_event = {current_event}')
        print (f'len(testing_image_matrix) = {len(testing_image_matrix)}')

        skip_the_event = False

        if len(testing_image_matrix)==0: 
            skip_the_event = True
            continue

        max_image_size = 0.
        sum_image_size = 0.
        for img in range(0,len(testing_image_matrix)):
            current_image_size = np.sum(testing_image_matrix[img])
            sum_image_size += current_image_size
            if max_image_size<current_image_size:
                max_image_size = current_image_size
        print (f'max_image_size = {max_image_size}')

        if max_image_size<image_size_cut: 
            skip_the_event = True
            print ('event rejected because of size cut.')
        if sum_image_size<2.*image_size_cut: 
            skip_the_event = True
            print ('event rejected because of size cut.')

        deg2rad = np.pi/180.
        rad2deg = 180./np.pi

        tel_alt = telesc_position_matrix[0][0]
        tel_az = telesc_position_matrix[0][1]
        tel_focal_length = telesc_position_matrix[0][4]
        tel_info = [tel_alt,tel_az,0.,0.,tel_focal_length]

        truth_shower_alt = truth_shower_position_matrix[0][0]
        truth_shower_az = truth_shower_position_matrix[0][1]
        truth_shower_core_x = truth_shower_position_matrix[0][2]
        truth_shower_core_y = truth_shower_position_matrix[0][3]
        truth_shower_energy = pow(10.,truth_shower_position_matrix[0][4])
        truth_shower_height = 20000.

        hillas_shower_alt = hillas_shower_position_matrix[0][0]
        hillas_shower_az = hillas_shower_position_matrix[0][1]
        hillas_shower_core_x = hillas_shower_position_matrix[0][2]
        hillas_shower_core_y = hillas_shower_position_matrix[0][3]
        hillas_valid = hillas_shower_position_matrix[0][4]

        truth_array_coord = [truth_shower_alt,truth_shower_az,truth_shower_core_x,truth_shower_core_y]
        truth_tel_coord = convert_array_coord_to_tel_coord(truth_array_coord,tel_info)
        truth_cam_x = truth_tel_coord[0]
        truth_cam_y = truth_tel_coord[1]

        #if not hillas_valid:
        #    skip_the_event = True
        #    print ('event rejected because of hillas reconstruction.')

        min_ntel = 2
        max_ntel = 1e10
        if len(testing_image_matrix)<min_ntel or len(testing_image_matrix)>max_ntel: 
            skip_the_event = True
            print ('event rejected because of n tel cut.')

        min_energy = 0.1
        max_energy = 100.0
        if truth_shower_energy<min_energy or truth_shower_energy>max_energy:
            skip_the_event = True
            print ('event rejected because of energy cut.')

        #this_event = 82501
        #if current_event != this_event: 
        #    skip_the_event = True
        #    if current_event > this_event: 
        #        exit()

        if skip_the_event:
            testing_image_matrix = []
            testing_time_matrix = []
            testing_param_matrix = []
            testing_moment_matrix = []
            truth_shower_position_matrix = []
            hillas_shower_position_matrix = []
            all_cam_axes = []
            telesc_position_matrix = []
            continue

        all_runs += [current_run]
        all_events += [current_event]

        all_truth_energy += [truth_shower_energy]
        if hillas_valid:
            all_hillas_valid += [1]
        else:
            all_hillas_valid += [0]

        fit_line_cam_x, fit_line_cam_y, fit_line_core_x, fit_line_core_y, open_angle = fit_lines_to_all_images(testing_image_matrix,telesc_position_matrix,geom,all_cam_axes)

        list_img_a, list_img_b, list_img_w, list_cam_x, list_cam_y, list_core_x, list_core_y, list_weight = fit_intersections_all_images(testing_image_matrix,telesc_position_matrix,geom,all_cam_axes)

        print ('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print ('individual line fit result:')
        print (f'current_event = {current_event}')
        print (f'truth_cam_x      = {truth_cam_x:0.3f}')
        print (f'truth_cam_y      = {truth_cam_y:0.3f}')
        print (f'fit_line_cam_x = {fit_line_cam_x:0.3f}')
        print (f'fit_line_cam_y = {fit_line_cam_y:0.3f}')
        print (f'truth_shower_core_x = {truth_shower_core_x:0.3f}')
        print (f'truth_shower_core_y = {truth_shower_core_y:0.3f}')
        print (f'fit_line_core_x = {fit_line_core_x*1000.:0.3f}')
        print (f'fit_line_core_y = {fit_line_core_y*1000.:0.3f}')
        print ('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

        
        fit_indiv_tel_coord = [fit_line_cam_x,fit_line_cam_y]
        fit_indiv_array_coord = convert_tel_coord_to_array_coord(fit_indiv_tel_coord,tel_info)
        fit_line_alt = fit_indiv_array_coord[0]
        fit_line_az = fit_indiv_array_coord[1]

        line_fit_cam_x_err += [fit_line_cam_x-0.]
        line_fit_cam_y_err += [fit_line_cam_y-0.]

        # Sort based on the values of elements in the first list
        #combined_lists = list(zip(list_pair_weight, list_pair_images, list_pair_telpos))
        #sorted_lists = sorted(combined_lists, key=lambda x: x[0], reverse=True)
        #list_pair_weight, list_pair_images, list_pair_telpos = zip(*sorted_lists)

        all_open_angle += [open_angle]
        all_image_size += [sum_image_size]



        fit_temp_energy = truth_shower_energy
        fit_temp_cam_x = truth_cam_x
        fit_temp_cam_y = truth_cam_y
        fit_temp_core_x = truth_shower_core_x/1000.
        fit_temp_core_y = truth_shower_core_y/1000.
        fit_temp_energy, fit_temp_cam_x, fit_temp_cam_y, fit_temp_core_x, fit_temp_core_y = fit_templates_to_all_images(testing_image_matrix,testing_time_matrix,telesc_position_matrix,geom,all_cam_axes,lookup_table_pkl,eigen_vectors_pkl,lookup_table_time_pkl,eigen_vectors_time_pkl,lookup_table_impact_pkl,lookup_table_impact_rms_pkl)
        print ('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print ('all template fit result:')
        print (f'current_event = {current_event}')
        print (f'truth_cam_x      = {truth_cam_x:0.3f}')
        print (f'truth_cam_y      = {truth_cam_y:0.3f}')
        print (f'fit_temp_cam_x = {fit_temp_cam_x:0.3f}')
        print (f'fit_temp_cam_y = {fit_temp_cam_y:0.3f}')
        print (f'truth_shower_core_x = {truth_shower_core_x:0.3f}')
        print (f'truth_shower_core_y = {truth_shower_core_y:0.3f}')
        print (f'fit_temp_core_x = {fit_temp_core_x*1000.:0.3f}')
        print (f'fit_temp_core_y = {fit_temp_core_y*1000.:0.3f}')
        print (f'truth_shower_energy = {truth_shower_energy:0.3f}')
        print (f'fit_temp_energy = {fit_temp_energy:0.3f}')
        print ('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

        avg_line_impact = 0.
        avg_temp_impact = 0.
        sum_weight = 0.
        for i in range(0,len(telesc_position_matrix)):
            tel_x = telesc_position_matrix[i][2]/1000.
            tel_y = telesc_position_matrix[i][3]/1000.
            avg_line_impact += (pow(fit_line_core_x-tel_x,2) + pow(fit_line_core_y-tel_y,2))
            avg_temp_impact += (pow(fit_temp_core_x-tel_x,2) + pow(fit_temp_core_y-tel_y,2))
            sum_weight += 1.
        avg_line_impact = pow(avg_line_impact/sum_weight,0.5)
        avg_temp_impact = pow(avg_temp_impact/sum_weight,0.5)
        line_impact += [avg_line_impact]
        temp_impact += [avg_temp_impact]


        fit_temp_tel_coord = [fit_temp_cam_x,fit_temp_cam_y]
        fit_temp_array_coord = convert_tel_coord_to_array_coord(fit_temp_tel_coord,tel_info)
        fit_temp_evt_alt = fit_temp_array_coord[0]
        fit_temp_evt_az = fit_temp_array_coord[1]

        temp_fit_cam_x_err += [fit_temp_cam_x-0.]
        temp_fit_cam_y_err += [fit_temp_cam_y-0.]


        hillas_err = 180./math.pi*pow(pow(hillas_shower_alt-truth_shower_alt,2)+pow(hillas_shower_az-truth_shower_az,2),0.5)
        hillas_sky_err += [hillas_err]
        line_err = 180./math.pi*pow(pow(fit_line_alt-truth_shower_alt,2)+pow(fit_line_az-truth_shower_az,2),0.5)
        line_fit_sky_err += [line_err]
        temp_err = 180./math.pi*pow(pow(fit_temp_evt_alt-truth_shower_alt,2)+pow(fit_temp_evt_az-truth_shower_az,2),0.5)
        temp_fit_sky_err += [temp_err]
        temp_fit_energy += [fit_temp_energy]
        print (f'hillas_err = {hillas_err} deg')
        print (f'line_err = {line_err} deg')
        print (f'temp_err = {temp_err} deg')

        image_sum_truth = None
        time_sum_truth = None
        image_sum_svd_line = None
        image_sum_svd_temp = None
        image_sum_svd_truth_temp = None
        baseline_chi2 = 0.
        temp_chi2 = 0.
        line_chi2 = 0.
        for i in range(0,len(testing_param_matrix)):

            tel_alt = telesc_position_matrix[i][0]
            tel_az = telesc_position_matrix[i][1]
            tel_x = telesc_position_matrix[i][2]
            tel_y = telesc_position_matrix[i][3]
            tel_pos = [tel_x,tel_y]

            delta_foci_time = np.log10(max(1e-3,testing_moment_matrix[i][7]))

            analysis_image_1d = testing_image_matrix[i]
            analysis_image_square = geom.image_to_cartesian_representation(analysis_image_1d)
            analysis_time_1d = testing_time_matrix[i]
            analysis_time_square = geom.image_to_cartesian_representation(analysis_time_1d)
            num_rows, num_cols = analysis_image_square.shape
            for row in range(0,num_rows):
                for col in range(0,num_cols):
                    if math.isnan(analysis_image_square[row,col]): 
                        analysis_image_square[row,col] = 0.
                        analysis_time_square[row,col] = 0.

            fit_shower_energy = truth_shower_energy
            fit_shower_cam_x = truth_cam_x
            fit_shower_cam_y = truth_cam_y
            fit_shower_core_x = truth_shower_core_x/1000.
            fit_shower_core_y = truth_shower_core_y/1000.
            evt_param = [fit_shower_cam_x,fit_shower_cam_y,fit_shower_core_x,fit_shower_core_y,math.log10(fit_shower_energy)]
            evt_param = np.array(evt_param)
            svd_image_2d_truth_temp = create_svd_image(evt_param,tel_pos,all_cam_axes[i],geom,lookup_table_pkl,lookup_table_impact_pkl,eigen_vectors_pkl)

            fit_shower_energy = fit_temp_energy
            fit_shower_cam_x = fit_line_cam_x
            fit_shower_cam_y = fit_line_cam_y
            fit_shower_core_x = fit_line_core_x
            fit_shower_core_y = fit_line_core_y
            evt_param = [fit_shower_cam_x,fit_shower_cam_y,fit_shower_core_x,fit_shower_core_y,math.log10(fit_shower_energy)]
            evt_param = np.array(evt_param)
            svd_image_2d_line = create_svd_image(evt_param,tel_pos,all_cam_axes[i],geom,lookup_table_pkl,lookup_table_impact_pkl,eigen_vectors_pkl)

            fit_shower_energy = fit_temp_energy
            fit_shower_cam_x = fit_temp_cam_x
            fit_shower_cam_y = fit_temp_cam_y
            fit_shower_core_x = fit_temp_core_x
            fit_shower_core_y = fit_temp_core_y
            evt_param = [fit_shower_cam_x,fit_shower_cam_y,fit_shower_core_x,fit_shower_core_y,math.log10(fit_shower_energy)]
            evt_param = np.array(evt_param)
            svd_image_2d_temp = create_svd_image(evt_param,tel_pos,all_cam_axes[i],geom,lookup_table_pkl,lookup_table_impact_pkl,eigen_vectors_pkl)

            for row in range(0,num_rows):
                for col in range(0,num_cols):
                    baseline_chi2 += pow(analysis_image_square[row,col],2)
                    line_chi2 += pow(analysis_image_square[row,col]-svd_image_2d_line[row,col],2)
                    temp_chi2 += pow(analysis_image_square[row,col]-svd_image_2d_temp[row,col],2)

            if i==0:
                image_sum_truth = analysis_image_square
                time_sum_truth = analysis_time_square
                image_sum_svd_line = svd_image_2d_line
                image_sum_svd_temp = svd_image_2d_temp
                image_sum_svd_truth_temp = svd_image_2d_truth_temp
            else:
                image_sum_truth += analysis_image_square
                time_sum_truth += analysis_time_square
                image_sum_svd_line += svd_image_2d_line
                image_sum_svd_temp += svd_image_2d_temp
                image_sum_svd_truth_temp += svd_image_2d_truth_temp



        xmax = max(geom.pix_x)/u.m
        xmin = min(geom.pix_x)/u.m
        ymax = max(geom.pix_y)/u.m
        ymin = min(geom.pix_y)/u.m
        num_rows, num_cols = image_sum_truth.shape

        core_xmax = 1.
        core_xmin = -1.
        core_ymax = 1.
        core_ymin = -1.

        temp_pointing_error = 180./math.pi*pow(pow(fit_temp_evt_alt-truth_shower_alt,2)+pow(fit_temp_evt_az-truth_shower_az,2),0.5)
        line_pointing_error = 180./math.pi*pow(pow(fit_line_alt-truth_shower_alt,2)+pow(fit_line_az-truth_shower_az,2),0.5)

        #if temp_pointing_error>0.2 and temp_pointing_error>line_pointing_error: 
        if temp_pointing_error>0.3: 

            fig.clf()
            axbig = fig.add_subplot()
            label_x = 'X'
            label_y = 'Y'
            axbig.set_xlabel(label_x)
            axbig.set_ylabel(label_y)
            im = axbig.imshow(image_sum_truth,origin='lower',extent=(xmin,xmax,ymin,ymax))
            line_x = np.linspace(xmin, xmax, 100)
            for tel in range(0,len(list_img_a)):
                a = list_img_a[tel]
                b = list_img_b[tel]
                line_y = -(a*line_x + b)
                axbig.plot(line_x,line_y,color='w',alpha=0.3,linestyle='dashed')
            axbig.set_xlim(xmin,xmax)
            axbig.set_ylim(ymin,ymax)
            axbig.scatter(truth_cam_x, -truth_cam_y, s=90, c='r', marker='+')
            axbig.scatter(fit_line_cam_x, -fit_line_cam_y, s=90, c='g', marker='+')
            cbar = fig.colorbar(im)
            cbar.set_label('PE')
            txt = axbig.text(-0.35, 0.35, 'open angle = %0.2e'%(open_angle), fontdict=font)
            #txt.set_path_effects([patheffects.withStroke(linewidth=2, foreground='w')])
            fig.savefig(f'{ctapipe_output}/output_plots/sum_image_run{current_run}_evt{current_event}_lines.png',bbox_inches='tight')
            axbig.remove()

            fig.clf()
            axbig = fig.add_subplot()
            label_x = 'X'
            label_y = 'Y'
            axbig.set_xlabel(label_x)
            axbig.set_ylabel(label_y)
            im = axbig.imshow(image_sum_svd_truth_temp,origin='lower',extent=(xmin,xmax,ymin,ymax))
            line_x = np.linspace(xmin, xmax, 100)
            for tel in range(0,len(list_img_a)):
                a = list_img_a[tel]
                b = list_img_b[tel]
                line_y = -(a*line_x + b)
                axbig.plot(line_x,line_y,color='w',alpha=0.3,linestyle='dashed')
            axbig.set_xlim(xmin,xmax)
            axbig.set_ylim(ymin,ymax)
            axbig.scatter(truth_cam_x, -truth_cam_y, s=90, c='r', marker='+')
            axbig.scatter(fit_temp_cam_x, -fit_temp_cam_y, s=90, c='g', marker='+')
            cbar = fig.colorbar(im)
            cbar.set_label('PE')
            fig.savefig(f'{ctapipe_output}/output_plots/sum_image_run{current_run}_evt{current_event}_svd_truth_temp.png',bbox_inches='tight')
            axbig.remove()

            fig.clf()
            axbig = fig.add_subplot()
            label_x = 'X'
            label_y = 'Y'
            axbig.set_xlabel(label_x)
            axbig.set_ylabel(label_y)
            im = axbig.imshow(image_sum_svd_temp,origin='lower',extent=(xmin,xmax,ymin,ymax))
            line_x = np.linspace(xmin, xmax, 100)
            for tel in range(0,len(list_img_a)):
                a = list_img_a[tel]
                b = list_img_b[tel]
                line_y = -(a*line_x + b)
                axbig.plot(line_x,line_y,color='w',alpha=0.3,linestyle='dashed')
            axbig.set_xlim(xmin,xmax)
            axbig.set_ylim(ymin,ymax)
            axbig.scatter(truth_cam_x, -truth_cam_y, s=90, c='r', marker='+')
            axbig.scatter(fit_temp_cam_x, -fit_temp_cam_y, s=90, c='g', marker='+')
            cbar = fig.colorbar(im)
            cbar.set_label('PE')
            fig.savefig(f'{ctapipe_output}/output_plots/sum_image_run{current_run}_evt{current_event}_svd_temp.png',bbox_inches='tight')
            axbig.remove()

            plot_count += 1
            #exit()


        testing_image_matrix = []
        testing_time_matrix = []
        testing_param_matrix = []
        testing_moment_matrix = []
        truth_shower_position_matrix = []
        hillas_shower_position_matrix = []
        all_cam_axes = []
        telesc_position_matrix = []

        #if plot_count==10: exit()

cta_array_ana_output = []
cta_array_ana_output += [all_truth_energy]
cta_array_ana_output += [all_hillas_valid]
cta_array_ana_output += [temp_fit_energy]
cta_array_ana_output += [hillas_sky_err]
cta_array_ana_output += [line_fit_sky_err]
cta_array_ana_output += [temp_fit_sky_err]
cta_array_ana_output += [all_open_angle]
cta_array_ana_output += [line_impact]
cta_array_ana_output += [temp_impact]
cta_array_ana_output += [all_image_size]
cta_array_ana_output += [line_fit_cam_x_err]
cta_array_ana_output += [line_fit_cam_y_err]
cta_array_ana_output += [temp_fit_cam_x_err]
cta_array_ana_output += [temp_fit_cam_y_err]
cta_array_ana_output += [all_runs]
cta_array_ana_output += [all_events]

output_filename = f'{ctapipe_output}/output_analysis/cta_array_ana_output_run{run_id}.pkl'
with open(output_filename,"wb") as file:
    pickle.dump(cta_array_ana_output, file)

#exit()




