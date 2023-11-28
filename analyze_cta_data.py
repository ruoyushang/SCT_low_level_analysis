
import os
import subprocess
import glob

import time
import math
import numpy as np
from astropy import units as u
from scipy.optimize import least_squares, minimize, brute, dual_annealing
from matplotlib import pyplot as plt
from matplotlib import colors
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

ctapipe_output = os.environ.get("CTAPIPE_OUTPUT_PATH")
subprocess.call(['sh', './clean_plots.sh'])


fig, ax = plt.subplots()
figsize_x = 8.6
figsize_y = 6.4
fig.set_figheight(figsize_y)
fig.set_figwidth(figsize_x)

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
                w = (image_w[img1]*image_w[img2])*np.arctan(abs((a1-a2)/(1.+a1*a2)))
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
    return a, b, np.trace(w)/chi2

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

def create_svd_image(shower_params,tel_pos,cam_axes,geom,lookup_table,lookup_table_arrival,eigen_vectors):

    shower_cam_x = float(shower_params[0])
    shower_cam_y = float(shower_params[1])
    shower_core_x = float(shower_params[2])
    shower_core_y = float(shower_params[3])
    shower_log_energy = float(shower_params[4])
    delta_foci_time = float(shower_params[5])
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
        latent_space += [lookup_table[r].get_bin_content(shower_impact,shower_log_energy,delta_foci_time)]
    latent_space = np.array(latent_space)
    #print (f'latent_space = {latent_space}')

    arrival = lookup_table_arrival.get_bin_content(shower_impact,shower_log_energy,delta_foci_time)
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

def sum_square_difference_between_images_v_cam_xy(try_params,shower_core_x,shower_core_y,delta_foci_time,image_2d_matrix,telesc_position_matrix,all_cam_axes,geom,lookup_table,lookup_table_arrival,eigenvectors):

    try_shower_cam_x = try_params[0]
    try_shower_cam_y = try_params[1]
    try_log_energy = try_params[2]

    tic = time.perf_counter()
    chi2 = 1.
    for tel in range(0,len(image_2d_matrix)):
        tel_x = telesc_position_matrix[tel][2]
        tel_y = telesc_position_matrix[tel][3]
        tel_pos = [tel_x,tel_y]

        image_data_2d = image_2d_matrix[tel]
    
        evt_param = [try_shower_cam_x,try_shower_cam_y,shower_core_x,shower_core_y,try_log_energy,delta_foci_time]
        evt_param = np.array(evt_param)
        image_svd_2d = create_svd_image(evt_param,tel_pos,all_cam_axes[tel],geom,lookup_table,lookup_table_arrival,eigenvectors)

        num_rows, num_cols = image_data_2d.shape
        for row in range(0,num_rows):
            for col in range(0,num_cols):
                data = image_data_2d[row,col]
                svd = image_svd_2d[row,col]
                chi2 += pow(data-svd,2)
                #chi2 += data*svd

    toc = time.perf_counter()
    #print (f'chi2 = {chi2}')
    #print (f'SVD image trial completed in {toc - tic:0.4f} seconds')
    #exit()

    return chi2
    #return 1./chi2

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

    sum_dist_sq_point = 0.
    for tel in range(0,len(list_point_w)):
        dist_sq = pow(core_x-list_core_x[tel],2) + pow(core_y-list_core_y[tel],2)
        sum_dist_sq_point += dist_sq*pow(list_point_w[tel],2)

    #print (f'open_angle_sq = {open_angle_sq}')
    #print (f'sum_dist_sq_line = {sum_dist_sq_line}')
    #print (f'sum_dist_sq_point = {sum_dist_sq_point}')
    #exit()

    alpha = 50.
    #alpha = 50./open_angle_sq
    sum_dist_sq = sum_dist_sq_line + alpha*sum_dist_sq_point
    return sum_dist_sq

def fit_templates_to_all_images(image_matrix,time_matrix,telesc_position_matrix,geom,all_cam_axes,lookup_table,lookup_table_arrival,lookup_table_arrival_rms,eigenvectors):

    fit_line_cam_x, fit_line_cam_y, fit_line_core_x, fit_line_core_y, open_angle = fit_lines_to_all_images(image_matrix,telesc_position_matrix,geom,all_cam_axes)

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

    list_img_x = []
    list_img_y = []
    list_tel_x = []
    list_tel_y = []
    list_line_a = []
    list_line_b = []
    list_line_w = []
    for tel in range(0,len(image_matrix)):

        image_data_1d = image_matrix[tel]
        image_data_2d = image_2d_matrix[tel]

        image_size = np.sum(image_data_1d)
        tel_x = telesc_position_matrix[tel][2]/1000.
        tel_y = telesc_position_matrix[tel][3]/1000.
        list_tel_x += [tel_x]
        list_tel_y += [tel_y]

        avg_x, avg_y, sum_w = get_image_center(image_data_2d, all_cam_axes[tel][0], all_cam_axes[tel][1])
        list_img_x += [avg_x]
        list_img_y += [avg_y]

        a, b, w = fit_image_to_line(image_data_2d, all_cam_axes[tel][0], all_cam_axes[tel][1])
        aT, bT, wT = fit_image_to_line(np.array(image_data_2d).transpose(), all_cam_axes[tel][1], all_cam_axes[tel][0])
        if w<wT:
            a = 1./aT
            b = -bT/aT
            w = wT

        list_line_a += [a]
        list_line_b += [b]
        list_line_w += [w]


    list_cam_x = []
    list_cam_y = []
    list_core_x = []
    list_core_y = []
    list_point_w = []
    avg_energy = 0.
    sum_weight = 0.
    for tel in range(0,len(image_matrix)):

        image_data_2d = image_2d_matrix[tel]
        time_data_2d = time_2d_matrix[tel]

        image_size = np.sum(image_data_1d)

        tel_x = telesc_position_matrix[tel][2]/1000.
        tel_y = telesc_position_matrix[tel][3]/1000.

        image_center_x, image_center_y, image_foci_x1, image_foci_y1, image_foci_x2, image_foci_y2, center_time, delta_foci_time, semi_major_sq, semi_minor_sq = find_image_moments_guided(image_data_2d, time_data_2d, all_cam_axes[tel][0], all_cam_axes[tel][1])
        if open_angle>0.1:
            image_center_x, image_center_y, image_foci_x1, image_foci_y1, image_foci_x2, image_foci_y2, center_time, delta_foci_time, semi_major_sq, semi_minor_sq = find_image_moments_guided(image_data_2d, time_data_2d, all_cam_axes[tel][0], all_cam_axes[tel][1], guided=True, arrival_x=fit_line_cam_x, arrival_y=fit_line_cam_y)
        delta_foci_time = np.log10(max(1e-3,delta_foci_time))

        image_foci_x = image_foci_x1
        image_foci_y = image_foci_y1
        image_foci_r1 = pow(image_foci_x1*image_foci_x1+image_foci_y1*image_foci_y1,0.5)
        image_foci_r2 = pow(image_foci_x2*image_foci_x2+image_foci_y2*image_foci_y2,0.5)

        image_data_recenter = np.zeros_like(image_data_2d)
        time_data_recenter = np.zeros_like(time_data_2d)
        shift_x = -image_foci_x
        shift_y = -image_foci_y
        image_data_recenter = image_translation(image_data_2d, all_cam_axes[tel][0], all_cam_axes[tel][1], shift_x, shift_y)
        time_data_recenter = image_translation(time_data_2d, all_cam_axes[tel][0], all_cam_axes[tel][1], shift_x, shift_y)

        image_data_rotate = np.zeros_like(image_data_2d)
        time_data_rotate = np.zeros_like(time_data_2d)
        angle_rad = -np.arctan2(image_foci_y-image_center_y,image_foci_x-image_center_x)
        image_data_rotate = image_rotation(image_data_recenter, all_cam_axes[tel][0], all_cam_axes[tel][1], angle_rad)
        time_data_rotate = image_rotation(time_data_recenter, all_cam_axes[tel][0], all_cam_axes[tel][1], angle_rad)
    
        image_data_rotate_1d = geom.image_from_cartesian_representation(image_data_rotate)
        time_data_rotate_1d = geom.image_from_cartesian_representation(time_data_rotate)

        fit_log_energy = 0.
        fit_impact = 0.1
        fit_chi2 = 1e10
        init_params = [fit_log_energy,fit_impact]
        n_bins_energy = len(lookup_table[0].yaxis)
        n_bins_impact = len(lookup_table[0].xaxis)
        for idx_x  in range(0,n_bins_impact):
            for idx_y  in range(0,n_bins_energy):
                try_log_energy = lookup_table[0].yaxis[idx_y]
                try_impact = lookup_table[0].xaxis[idx_x]
                init_params = [try_log_energy,try_impact,delta_foci_time]
                try_chi2 = sqaure_difference_between_1d_images(init_params,all_cam_axes[tel],geom,image_data_rotate_1d,lookup_table,eigenvectors)
                if try_chi2<fit_chi2:
                    fit_chi2 = try_chi2
                    fit_log_energy = try_log_energy
                    fit_impact = try_impact

        print (f'fit_log_energy = {fit_log_energy}')
        fit_arrival = lookup_table_arrival.get_bin_content(fit_impact,fit_log_energy,delta_foci_time)
        fit_arrival_error = lookup_table_arrival_rms.get_bin_content(fit_impact,fit_log_energy,delta_foci_time)
        if fit_arrival<0.01:
            fit_arrival = 0.01
        if fit_arrival_error<0.01:
            fit_arrival_error = 0.01

        delta_x = fit_arrival*np.cos(-angle_rad)
        delta_y = fit_arrival*np.sin(-angle_rad)
        cam_x = image_foci_x + delta_x
        cam_y = image_foci_y + delta_y
        line_a = delta_y/delta_x
        line_b_sky = image_foci_y-line_a*image_foci_x

        line_b_grd = tel_y-line_a*tel_x
        delta_x = fit_impact*np.cos(-angle_rad)
        delta_y = fit_impact*np.sin(-angle_rad)
        core_x = tel_x + delta_x
        core_y = tel_y + delta_y

        sum_weight += 1./(fit_arrival_error*fit_arrival_error)
        avg_energy += pow(10.,fit_log_energy)/(fit_arrival_error*fit_arrival_error)
        list_cam_x += [cam_x]
        list_cam_y += [cam_y]
        list_core_x += [core_x]
        list_core_y += [core_y]
        list_point_w += [1./(fit_arrival_error*fit_arrival_error)]


    avg_energy = avg_energy/sum_weight
    sum_error = pow(1./sum_weight,0.5)
    print (f'sum_error = {sum_error}')

    try_params = [fit_line_core_x,fit_line_core_y]
    solution = minimize(
        sum_square_distance_to_image_lines_and_points,
        x0=try_params,
        args=(list_tel_x,list_tel_y,list_line_a,list_line_w,list_core_x,list_core_y,list_point_w),
        method='L-BFGS-B',
    )
    avg_core_x = solution['x'][0]
    avg_core_y = solution['x'][1]

    try_params = [fit_line_cam_x,fit_line_cam_y]
    solution = minimize(
        sum_square_distance_to_image_lines_and_points,
        x0=try_params,
        args=(list_img_x,list_img_y,list_line_a,list_line_w,list_cam_x,list_cam_y,list_point_w),
        method='L-BFGS-B',
    )
    avg_cam_x = solution['x'][0]
    avg_cam_y = solution['x'][1]


    return avg_energy, avg_cam_x, avg_cam_y, avg_core_x, avg_core_y
    #diff_line_vs_temp = pow(pow(avg_cam_x-fit_line_cam_x,2)+pow(avg_cam_y-fit_line_cam_y,2),0.5)
    #if diff_line_vs_temp<0.02:
    #    return avg_energy, avg_cam_x, avg_cam_y, avg_core_x, avg_core_y

    min_chi2 = 1e10
    fit_shower_energy = avg_energy
    #fit_shower_cam_x = fit_line_cam_x
    #fit_shower_cam_y = fit_line_cam_y
    #fit_shower_core_x = fit_line_core_x
    #fit_shower_core_y = fit_line_core_y
    fit_shower_cam_x = avg_cam_x
    fit_shower_cam_y = avg_cam_y
    fit_shower_core_x = avg_core_x
    fit_shower_core_y = avg_core_y

    stepsize = [0.005,0.005,0.1]
    try_params = [fit_shower_cam_x,fit_shower_cam_y,math.log10(fit_shower_energy)]
    bounds = [(fit_shower_cam_x-2.*sum_error,fit_shower_cam_x+2.*sum_error),(fit_shower_cam_y-2.*sum_error,fit_shower_cam_y+2.*sum_error),(math.log10(fit_shower_energy)-0.3,math.log10(fit_shower_energy)+0.3)]
    solution = minimize(
        sum_square_difference_between_images_v_cam_xy,
        x0=try_params,
        args=(fit_line_core_x,fit_line_core_y,delta_foci_time,image_2d_matrix,telesc_position_matrix,all_cam_axes,geom,lookup_table,lookup_table_arrival,eigenvectors),
        bounds=bounds,
        method='L-BFGS-B',
        jac=None,
        options={'eps':stepsize,'ftol':0.01},
    )
    fit_shower_cam_x = solution['x'][0]
    fit_shower_cam_y = solution['x'][1]
    fit_shower_energy = pow(10.,solution['x'][2])
    
    toc = time.perf_counter()
    print (f'SVD image fit completed in {toc - tic:0.4f} seconds')

    return fit_shower_energy, fit_shower_cam_x, fit_shower_cam_y, fit_shower_core_x, fit_shower_core_y

def fit_templates_to_individual_images(image_matrix,time_matrix,telesc_position_matrix,geom,all_cam_axes,lookup_table_arrival,lookup_table_arrival_error,lookup_table,eigenvectors,init_cam_x,init_cam_y,init_core_x,init_core_y,open_angle):

    image_data_1d = image_matrix[0]
    image_data_2d = geom.image_to_cartesian_representation(image_data_1d)
    sky_xy_log_likelihood = np.zeros_like(image_data_2d)
    num_rows, num_cols = sky_xy_log_likelihood.shape
    for x_idx in range(0,num_cols):
        for y_idx in range(0,num_rows):
            sky_xy_log_likelihood[y_idx,x_idx] = 0.
    sky_likelihood_xaxis = all_cam_axes[0][0]
    sky_likelihood_yaxis = all_cam_axes[0][1]

    n_bins_core = 200
    grd_map_size = 2.
    grd_xy_log_likelihood = np.zeros((n_bins_core,n_bins_core))
    grd_likelihood_xaxis = []
    grd_likelihood_yaxis = []
    for x_idx  in range(0,n_bins_core):
        grd_likelihood_xaxis += [-0.5*grd_map_size+x_idx*grd_map_size/float(n_bins_core)]
    for y_idx  in range(0,n_bins_core):
        grd_likelihood_yaxis += [-0.5*grd_map_size+y_idx*grd_map_size/float(n_bins_core)]
    for x_idx  in range(0,n_bins_core):
        for y_idx  in range(0,n_bins_core):
            grd_xy_log_likelihood[y_idx,x_idx] = 0.

    sum_image_size = 0.
    for tel in range(0,len(image_matrix)):
        image_data_1d = image_matrix[tel]
        sum_image_size += np.sum(image_data_1d)

    avg_energy = 0.
    avg_cam_x = 0.
    avg_cam_y = 0.
    sum_weight = 0.
    for tel in range(0,len(image_matrix)):

        image_data_1d = image_matrix[tel]
        image_data_2d = geom.image_to_cartesian_representation(image_data_1d)
        time_data_1d = time_matrix[tel]
        time_data_2d = geom.image_to_cartesian_representation(time_data_1d)
        num_rows, num_cols = image_data_2d.shape
        for x_idx in range(0,num_cols):
            for y_idx in range(0,num_rows):
                if math.isnan(image_data_2d[y_idx,x_idx]): 
                    image_data_2d[y_idx,x_idx] = 0.
                    time_data_2d[y_idx,x_idx] = 0.

        image_size = np.sum(image_data_1d)

        tel_x = telesc_position_matrix[tel][2]/1000.
        tel_y = telesc_position_matrix[tel][3]/1000.
        init_impact = pow(pow(init_core_x-tel_x,2) + pow(init_core_y-tel_y,2),0.5)

        open_angle_cut = 0.05
        image_center_x, image_center_y, image_foci_x1, image_foci_y1, image_foci_x2, image_foci_y2, center_time, delta_foci_time, semi_major_sq, semi_minor_sq = find_image_moments_guided(image_data_2d, time_data_2d, all_cam_axes[tel][0], all_cam_axes[tel][1], guided=True, arrival_x=init_cam_x, arrival_y=init_cam_y)
        if tel==0 and open_angle<open_angle_cut:
            image_center_x, image_center_y, image_foci_x1, image_foci_y1, image_foci_x2, image_foci_y2, center_time, delta_foci_time, semi_major_sq, semi_minor_sq = find_image_moments_guided(image_data_2d, time_data_2d, all_cam_axes[tel][0], all_cam_axes[tel][1])
        #image_center_x, image_center_y, image_foci_x1, image_foci_y1, image_foci_x2, image_foci_y2, center_time, delta_foci_time, semi_major_sq, semi_minor_sq = find_image_moments_guided(image_data_2d, time_data_2d, all_cam_axes[tel][0], all_cam_axes[tel][1])

        image_foci_x = image_foci_x1
        image_foci_y = image_foci_y1
        image_foci_r1 = pow(image_foci_x1*image_foci_x1+image_foci_y1*image_foci_y1,0.5)
        image_foci_r2 = pow(image_foci_x2*image_foci_x2+image_foci_y2*image_foci_y2,0.5)
        print (f'image_foci_r1 = {image_foci_r1}')
        print (f'image_foci_r2 = {image_foci_r2}')
        print (f'delta_foci_time = {delta_foci_time}')
        delta_foci_time = np.log10(max(1e-3,delta_foci_time))

        image_data_recenter = np.zeros_like(image_data_2d)
        time_data_recenter = np.zeros_like(time_data_2d)
        shift_x = -image_foci_x
        shift_y = -image_foci_y
        image_data_recenter = image_translation(image_data_2d, all_cam_axes[tel][0], all_cam_axes[tel][1], shift_x, shift_y)
        time_data_recenter = image_translation(time_data_2d, all_cam_axes[tel][0], all_cam_axes[tel][1], shift_x, shift_y)

        image_data_rotate = np.zeros_like(image_data_2d)
        time_data_rotate = np.zeros_like(time_data_2d)
        angle_rad = -np.arctan2(image_foci_y-image_center_y,image_foci_x-image_center_x)
        image_data_rotate = image_rotation(image_data_recenter, all_cam_axes[tel][0], all_cam_axes[tel][1], angle_rad)
        time_data_rotate = image_rotation(time_data_recenter, all_cam_axes[tel][0], all_cam_axes[tel][1], angle_rad)
    
        image_data_rotate_1d = geom.image_from_cartesian_representation(image_data_rotate)
        time_data_rotate_1d = geom.image_from_cartesian_representation(time_data_rotate)

        fit_log_energy = 0.
        fit_impact = 0.1
        fit_chi2 = 1e10
        init_params = [fit_log_energy,fit_impact]
        n_bins_energy = len(lookup_table[0].yaxis)
        n_bins_impact = len(lookup_table[0].xaxis)
        for idx_x  in range(0,n_bins_impact):
            for idx_y  in range(0,n_bins_energy):
                try_log_energy = lookup_table[0].yaxis[idx_y]
                try_impact = lookup_table[0].xaxis[idx_x]
                init_params = [try_log_energy,try_impact,delta_foci_time]
                try_chi2 = sqaure_difference_between_1d_images(init_params,all_cam_axes[tel],geom,image_data_rotate_1d,lookup_table,eigenvectors)
                if try_chi2<fit_chi2:
                    fit_chi2 = try_chi2
                    fit_log_energy = try_log_energy
                    fit_impact = try_impact


        image_fit_arrival = lookup_table_arrival.get_bin_content(fit_impact,fit_log_energy,delta_foci_time)
        image_fit_arrival_error = lookup_table_arrival_error.get_bin_content(fit_impact,fit_log_energy,delta_foci_time)
        print (f'fit_log_energy = {fit_log_energy}')
        print (f'fit_impact = {fit_impact}')
        print (f'image_fit_arrival = {image_fit_arrival}')
        print (f'image_fit_arrival_error = {image_fit_arrival_error}')
        if image_fit_arrival_error==0.:
            image_fit_arrival_error = 1e10

        fit_arrival = image_fit_arrival
        fit_arrival_error = image_fit_arrival

        delta_x = fit_arrival*np.cos(-angle_rad)
        delta_y = fit_arrival*np.sin(-angle_rad)
        cam_x = image_foci_x + delta_x
        cam_y = image_foci_y + delta_y
        line_a = delta_y/delta_x
        line_b_sky = image_foci_y-line_a*image_foci_x

        if tel==0 and open_angle<open_angle_cut:
            init_cam_x = cam_x
            init_cam_y = cam_y

        line_b_grd = tel_y-line_a*tel_x
        delta_x = fit_impact*np.cos(-angle_rad)
        delta_y = fit_impact*np.sin(-angle_rad)
        core_x = tel_x + delta_x
        core_y = tel_y + delta_y

        reco_uncertainty = fit_arrival_error

        longi_error = reco_uncertainty*reco_uncertainty
        trans_error = longi_error*semi_minor_sq/semi_major_sq

        for x_idx in range(0,num_cols):
            for y_idx in range(0,num_rows):
                pix_x = sky_likelihood_xaxis[x_idx]
                pix_y = sky_likelihood_yaxis[y_idx]
                dist2center_sq = pow(pix_x-cam_x,2) + pow(pix_y-cam_y,2)
                dist2line_sq = pow((line_a*pix_x-pix_y+line_b_sky),2)/(1.+line_a*line_a)
                new_likelihood = ((-dist2center_sq/(2.*longi_error)) + (-dist2line_sq/(2.*trans_error)))
                sky_xy_log_likelihood[y_idx,x_idx] += new_likelihood

        for x_idx in range(0,n_bins_core):
            for y_idx in range(0,n_bins_core):
                pix_x = grd_likelihood_xaxis[x_idx]
                pix_y = grd_likelihood_yaxis[y_idx]
                dist2center_sq = pow(pix_x-core_x,2) + pow(pix_y-core_y,2)
                dist2line_sq = pow((line_a*pix_x-pix_y+line_b_grd),2)/(1.+line_a*line_a)
                new_likelihood = ((-dist2center_sq/(2.*longi_error)) + (-dist2line_sq/(2.*trans_error)))
                grd_xy_log_likelihood[y_idx,x_idx] += new_likelihood

        avg_energy += pow(10.,fit_log_energy)*1./longi_error
        sum_weight += 1./longi_error

    avg_energy = avg_energy/sum_weight

    max_index = np.argmax(sky_xy_log_likelihood)
    max_row, max_col = np.unravel_index(max_index, sky_xy_log_likelihood.shape)
    max_log_likelihood = sky_xy_log_likelihood[max_row,max_col]
    avg_cam_x = sky_likelihood_xaxis[max_col]
    avg_cam_y = sky_likelihood_yaxis[max_row]

    max_index = np.argmax(grd_xy_log_likelihood)
    max_row, max_col = np.unravel_index(max_index, grd_xy_log_likelihood.shape)
    max_log_likelihood = grd_xy_log_likelihood[max_row,max_col]
    avg_core_x = grd_likelihood_xaxis[max_col]
    avg_core_y = grd_likelihood_yaxis[max_row]

    return avg_energy, avg_cam_x, avg_cam_y, sky_xy_log_likelihood, avg_core_x, avg_core_y, grd_xy_log_likelihood, sum_weight

def fit_lines_to_individual_images(image_matrix,telesc_position_matrix,geom,all_cam_axes):

    list_all_img_a = []
    list_all_img_b = []
    list_all_img_w = []
    for tel in range(0,len(image_matrix)):
        rad2deg = 180./np.pi
        tel_x = telesc_position_matrix[tel][2]
        tel_y = telesc_position_matrix[tel][3]

        tel_x_axis = np.array(all_cam_axes[tel][0])
        tel_y_axis = np.array(all_cam_axes[tel][1])

        tel_image_1d = image_matrix[tel]
        tel_image_2d = geom.image_to_cartesian_representation(tel_image_1d)

        num_rows, num_cols = tel_image_2d.shape
        for x_idx in range(0,num_cols):
            for y_idx in range(0,num_rows):
                if math.isnan(tel_image_2d[y_idx,x_idx]): tel_image_2d[y_idx,x_idx] = 0.

        a, b, w = fit_image_to_line(tel_image_2d,tel_x_axis,tel_y_axis)
        aT, bT, wT = fit_image_to_line(np.array(tel_image_2d).transpose(),tel_y_axis,tel_x_axis)
        if w<wT:
            a = 1./aT
            b = -bT/aT
            w = wT

        list_all_img_a += [a]
        list_all_img_b += [b]
        list_all_img_w += [w]


    avg_ls_evt_cam_x = 0.
    avg_ls_evt_cam_y = 0.
    avg_ls_evt_core_x = 0.
    avg_ls_evt_core_y = 0.
    ls_evt_weight = 0.
    list_pair_images = []
    list_pair_telpos = []
    list_pair_weight = []
    list_pair_a = []
    for tel1 in range(0,len(image_matrix)-1):
        for tel2 in range(tel1+1,len(image_matrix)):

            rad2deg = 180./np.pi
            tel1_x = telesc_position_matrix[tel1][2]
            tel1_y = telesc_position_matrix[tel1][3]
            tel2_x = telesc_position_matrix[tel2][2]
            tel2_y = telesc_position_matrix[tel2][3]

            tel1_x_axis = np.array(all_cam_axes[tel1][0])
            tel1_y_axis = np.array(all_cam_axes[tel1][1])
            tel2_x_axis = np.array(all_cam_axes[tel2][0])
            tel2_y_axis = np.array(all_cam_axes[tel2][1])

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
            w1 = list_all_img_w[tel1]
            list_img_a += [a1]
            list_img_b += [b1]
            list_img_w += [w1]

            a2 = list_all_img_a[tel2]
            b2 = list_all_img_b[tel2]
            w2 = list_all_img_w[tel2]
            list_img_a += [a2]
            list_img_b += [b2]
            list_img_w += [w2]

            ls_cam_x, ls_cam_y, ls_sky_weight = find_intersection_on_focalplane(list_tel_x,list_tel_y,list_img_a,list_img_b,list_img_w)
            ls_core_x, ls_core_y, ls_core_weight = find_intersection_on_ground(list_tel_x,list_tel_y,list_img_a,list_img_b,list_img_w)

            ls_pair_weight = ls_sky_weight
            avg_ls_evt_cam_x += ls_cam_x*ls_pair_weight
            avg_ls_evt_cam_y += ls_cam_y*ls_pair_weight
            avg_ls_evt_core_x += ls_core_x*ls_pair_weight
            avg_ls_evt_core_y += ls_core_y*ls_pair_weight
            ls_evt_weight += ls_pair_weight

            list_pair_images += [[tel1_image_1d,tel2_image_1d]]
            list_pair_a += [[a1,a2]]
            list_pair_weight += [ls_core_weight]
            list_pair_telpos += [[telesc_position_matrix[tel1],telesc_position_matrix[tel2]]]

    if ls_evt_weight!=0.:
        avg_ls_evt_cam_x = avg_ls_evt_cam_x/ls_evt_weight
        avg_ls_evt_cam_y = avg_ls_evt_cam_y/ls_evt_weight
        avg_ls_evt_core_x = avg_ls_evt_core_x/ls_evt_weight
        avg_ls_evt_core_y = avg_ls_evt_core_y/ls_evt_weight

    return [avg_ls_evt_cam_x, avg_ls_evt_cam_y, avg_ls_evt_core_x/1000., avg_ls_evt_core_y/1000.], list_all_img_a, list_all_img_b, list_all_img_w, list_pair_images, list_pair_a, list_pair_weight, list_pair_telpos

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

def fit_lines_to_all_images(image_matrix,telesc_position_matrix,geom,all_cam_axes):

    init_cam_x = 0.
    init_cam_y = 0.
    init_core_x = 0.
    init_core_y = 0.

    list_img_x = []
    list_img_y = []
    list_tel_x = []
    list_tel_y = []
    list_img_a = []
    list_img_b = []
    list_img_w = []
    for tel in range(0,len(image_matrix)):

        image_data_1d = image_matrix[tel]
        image_data_2d = geom.image_to_cartesian_representation(image_data_1d)
        num_rows, num_cols = image_data_2d.shape
        for x_idx in range(0,num_cols):
            for y_idx in range(0,num_rows):
                if math.isnan(image_data_2d[y_idx,x_idx]): 
                    image_data_2d[y_idx,x_idx] = 0.

        image_size = np.sum(image_data_1d)
        tel_x = telesc_position_matrix[tel][2]/1000.
        tel_y = telesc_position_matrix[tel][3]/1000.
        list_tel_x += [tel_x]
        list_tel_y += [tel_y]

        avg_x, avg_y, sum_w = get_image_center(image_data_2d, all_cam_axes[tel][0], all_cam_axes[tel][1])
        list_img_x += [avg_x]
        list_img_y += [avg_y]

        a, b, w = fit_image_to_line(image_data_2d, all_cam_axes[tel][0], all_cam_axes[tel][1])
        aT, bT, wT = fit_image_to_line(np.array(image_data_2d).transpose(), all_cam_axes[tel][1], all_cam_axes[tel][0])
        if w<wT:
            a = 1./aT
            b = -bT/aT
            w = wT

        list_img_a += [a]
        list_img_b += [b]
        list_img_w += [w]

    try_params = [init_core_x,init_core_y]
    solution = minimize(
        sum_square_distance_to_image_lines,
        x0=try_params,
        args=(list_tel_x,list_tel_y,list_img_a,list_img_w),
        method='L-BFGS-B',
    )
    fit_core_x = solution['x'][0]
    fit_core_y = solution['x'][1]

    try_params = [init_cam_x,init_cam_y]
    solution = minimize(
        sum_square_distance_to_image_lines,
        x0=try_params,
        args=(list_img_x,list_img_y,list_img_a,list_img_w),
        method='L-BFGS-B',
    )
    fit_cam_x = solution['x'][0]
    fit_cam_y = solution['x'][1]

    open_angle = 0.
    for t1 in range(0,len(list_img_a)-1):
        for t2 in range(t1+1,len(list_img_a)):
            a1 = list_img_a[t1]
            a2 = list_img_a[t2]
            open_angle += pow(np.arctan(abs((a1-a2)/(1.+a1*a2))),2)

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



testing_sample_path = []
#testing_sample_path = [get_dataset_path('gamma_20deg_0deg_run835___cta-prod3-sct_desert-2150m-Paranal-SCT.simtel.gz')]
with open('test_sim_files.txt', 'r') as file:
    for line in file:
        testing_sample_path += [get_dataset_path(line.strip('\n'))]

print ('loading svd pickle data... ')
output_filename = f'{ctapipe_output}/output_machines/lookup_table.pkl'
lookup_table_pkl = pickle.load(open(output_filename, "rb"))

output_filename = f'{ctapipe_output}/output_machines/eigen_vectors.pkl'
eigen_vectors_pkl = pickle.load(open(output_filename, "rb"))

output_filename = f'{ctapipe_output}/output_machines/lookup_table_arrival.pkl'
lookup_table_arrival_pkl = pickle.load(open(output_filename, "rb"))

output_filename = f'{ctapipe_output}/output_machines/lookup_table_arrival_rms.pkl'
lookup_table_arrival_rms_pkl = pickle.load(open(output_filename, "rb"))


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
all_cam_axes = []
telesc_position_matrix = []

all_truth_energy = []
indiv_temp_fit_energy = []
simul_temp_fit_energy = []
indiv_line_fit_sky_err = []
indiv_temp_fit_sky_err = []
simul_temp_fit_sky_err = []
all_open_angle = []
all_temp_weight = []
indiv_line_impact = []
indiv_temp_impact = []
simul_temp_impact = []
all_temp_chi2 = []
all_line_chi2 = []

for path in range(0,len(testing_sample_path)):

    testing_id_list = []
    big_telesc_position_matrix = []
    big_truth_shower_position_matrix = []
    test_cam_axes = []
    big_testing_image_matrix = []
    big_testing_time_matrix = []
    big_testing_param_matrix = []
    big_testing_moment_matrix = []

    source = SimTelEventSource(testing_sample_path[path], focal_length_choice='EQUIVALENT')
    subarray = source.subarray
    ob_keys = source.observation_blocks.keys()
    run_id = list(ob_keys)[0]
    output_filename = f'{ctapipe_output}/output_samples/testing_sample_noisy_clean_origin_run{run_id}.pkl'
    #output_filename = f'{ctapipe_output}/output_samples/testing_sample_truth_dirty_origin_run{run_id}.pkl'
    print (f'loading pickle trainging sample data: {output_filename}')
    if not os.path.exists(output_filename):
        print (f'file does not exist.')
        continue
    training_sample = pickle.load(open(output_filename, "rb"))

    testing_id_list += training_sample[0]
    big_telesc_position_matrix += training_sample[1]
    big_truth_shower_position_matrix += training_sample[2]
    test_cam_axes += training_sample[3]
    big_testing_image_matrix += training_sample[4]
    big_testing_time_matrix += training_sample[5]
    big_testing_param_matrix += training_sample[6]
    big_testing_moment_matrix += training_sample[7]

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
            all_cam_axes += [test_cam_axes[img]]
            telesc_position_matrix += [big_telesc_position_matrix[img]]
        else:
            collected_all_images = False
    
            print ('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
            print (f'collected all images.')
            print (f'current_run = {current_run}')
            print (f'current_event = {current_event}')
            print (f'len(testing_image_matrix) = {len(testing_image_matrix)}')
    
            if len(testing_image_matrix)==0: 
                testing_image_matrix = []
                testing_time_matrix = []
                testing_param_matrix = []
                testing_moment_matrix = []
                truth_shower_position_matrix = []
                all_cam_axes = []
                telesc_position_matrix = []
                continue
    
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
    
            truth_array_coord = [truth_shower_alt,truth_shower_az,truth_shower_core_x,truth_shower_core_y]
            truth_tel_coord = convert_array_coord_to_tel_coord(truth_array_coord,tel_info)
            truth_cam_x = truth_tel_coord[0]
            truth_cam_y = truth_tel_coord[1]
    
            min_ntel = 3
            max_ntel = 1e10
            if len(testing_image_matrix)<min_ntel or len(testing_image_matrix)>max_ntel: 
                testing_image_matrix = []
                testing_time_matrix = []
                testing_param_matrix = []
                testing_moment_matrix = []
                truth_shower_position_matrix = []
                all_cam_axes = []
                telesc_position_matrix = []
                continue
    
            min_energy = 0.1
            max_energy = 10.0
            if truth_shower_energy<min_energy or truth_shower_energy>max_energy:
                testing_image_matrix = []
                testing_time_matrix = []
                testing_param_matrix = []
                testing_moment_matrix = []
                truth_shower_position_matrix = []
                all_cam_axes = []
                telesc_position_matrix = []
                continue

            #if current_event != 130705: 
            #if current_event != 24905: 
            #    testing_image_matrix = []
            #    testing_time_matrix = []
            #    testing_param_matrix = []
            #    testing_moment_matrix = []
            #    truth_shower_position_matrix = []
            #    all_cam_axes = []
            #    telesc_position_matrix = []
            #    continue
    
            fit_all_cam_x, fit_all_cam_y, fit_all_core_x, fit_all_core_y, open_angle = fit_lines_to_all_images(testing_image_matrix,telesc_position_matrix,geom,all_cam_axes)
    
            ls_evt_line_vec, list_img_a, list_img_b, list_img_w, list_pair_images, list_pair_a, list_pair_weight, list_pair_telpos = fit_lines_to_individual_images(testing_image_matrix,telesc_position_matrix,geom,all_cam_axes)
            fit_line_cam_x = float(ls_evt_line_vec[0])
            fit_line_cam_y = float(ls_evt_line_vec[1])
            fit_line_core_x = float(ls_evt_line_vec[2])
            fit_line_core_y = float(ls_evt_line_vec[3])

            print ('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
            print ('individual line fit result:')
            print (f'current_event = {current_event}')
            print (f'truth_cam_x      = {truth_cam_x:0.3f}')
            print (f'truth_cam_y      = {truth_cam_y:0.3f}')
            print (f'fit_line_cam_x = {fit_line_cam_x:0.3f}')
            print (f'fit_line_cam_y = {fit_line_cam_y:0.3f}')
            print (f'fit_all_cam_x = {fit_all_cam_x:0.3f}')
            print (f'fit_all_cam_y = {fit_all_cam_y:0.3f}')
            print (f'truth_shower_core_x = {truth_shower_core_x:0.3f}')
            print (f'truth_shower_core_y = {truth_shower_core_y:0.3f}')
            print (f'fit_line_core_x = {fit_line_core_x*1000.:0.3f}')
            print (f'fit_line_core_y = {fit_line_core_y*1000.:0.3f}')
            print (f'fit_all_core_x = {fit_all_core_x*1000.:0.3f}')
            print (f'fit_all_core_y = {fit_all_core_y*1000.:0.3f}')
            print ('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

            
            fit_indiv_tel_coord = [fit_line_cam_x,fit_line_cam_y]
            fit_indiv_array_coord = convert_tel_coord_to_array_coord(fit_indiv_tel_coord,tel_info)
            fit_line_alt = fit_indiv_array_coord[0]
            fit_line_az = fit_indiv_array_coord[1]
    
            # Sort based on the values of elements in the first list
            combined_lists = list(zip(list_pair_weight, list_pair_images, list_pair_telpos))
            sorted_lists = sorted(combined_lists, key=lambda x: x[0], reverse=True)
            list_pair_weight, list_pair_images, list_pair_telpos = zip(*sorted_lists)

            #open_angle = 0.
            #for tp in range(0,len(list_pair_a)):
            #    a1 = list_pair_a[tp][0]
            #    a2 = list_pair_a[tp][1]
            #    open_angle += pow(np.arctan(abs((a1-a2)/(1.+a1*a2))),2)
            all_open_angle += [open_angle]

    
    
            fit_temp_energy = truth_shower_energy
            fit_temp_cam_x = truth_cam_x
            fit_temp_cam_y = truth_cam_y
            fit_temp_core_x = truth_shower_core_x/1000.
            fit_temp_core_y = truth_shower_core_y/1000.
            #fit_temp_energy, fit_temp_cam_x, fit_temp_cam_y, temp_weight_skymap, fit_temp_core_x, fit_temp_core_y, temp_weight_grdmap, temp_weight = fit_templates_to_individual_images(testing_image_matrix,testing_time_matrix,telesc_position_matrix,geom,all_cam_axes,lookup_table_arrival_pkl,lookup_table_arrival_error_pkl,lookup_table_pkl,eigen_vectors_pkl, fit_line_cam_x, fit_line_cam_y,fit_line_core_x,fit_line_core_y,open_angle)
            #print ('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
            #print ('individual template fit result:')
            #print (f'current_event = {current_event}')
            #print (f'truth_cam_x      = {truth_cam_x:0.3f}')
            #print (f'truth_cam_y      = {truth_cam_y:0.3f}')
            #print (f'fit_temp_cam_x = {fit_temp_cam_x:0.3f}')
            #print (f'fit_temp_cam_y = {fit_temp_cam_y:0.3f}')
            #print (f'truth_shower_core_x = {truth_shower_core_x:0.3f}')
            #print (f'truth_shower_core_y = {truth_shower_core_y:0.3f}')
            #print (f'fit_temp_core_x = {fit_temp_core_x*1000.:0.3f}')
            #print (f'fit_temp_core_y = {fit_temp_core_y*1000.:0.3f}')
            #print (f'truth_shower_energy = {truth_shower_energy:0.3f}')
            #print (f'fit_temp_energy = {fit_temp_energy:0.3f}')
            #print ('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    
            #if temp_weight<2.*1e5: continue
    
            fit_temp_tel_coord = [fit_temp_cam_x,fit_temp_cam_y]
            fit_temp_array_coord = convert_tel_coord_to_array_coord(fit_temp_tel_coord,tel_info)
            fit_temp_evt_alt = fit_temp_array_coord[0]
            fit_temp_evt_az = fit_temp_array_coord[1]
    
    
            all_temp_weight += [1.]
            #all_temp_weight += [temp_weight]

            indiv_fit_sky_error = pow(pow(fit_temp_cam_x-fit_line_cam_x,2)+pow(fit_temp_cam_y-fit_line_cam_y,2),0.5)
            indiv_fit_grd_error = pow(pow(fit_temp_core_x-fit_line_core_x,2)+pow(fit_temp_core_y-fit_line_core_y,2),0.5)
            print (f'indiv_fit_sky_error = {indiv_fit_sky_error}')
            print (f'indiv_fit_grd_error = {indiv_fit_grd_error}')
          
            fit_simul_temp_energy = fit_temp_energy
            fit_simul_temp_cam_x = fit_temp_cam_x
            fit_simul_temp_cam_y = fit_temp_cam_y
            fit_simul_temp_core_x = fit_temp_core_x
            fit_simul_temp_core_y = fit_temp_core_y
            fit_simul_temp_energy, fit_simul_temp_cam_x, fit_simul_temp_cam_y, fit_simul_temp_core_x, fit_simul_temp_core_y = fit_templates_to_all_images(testing_image_matrix,testing_time_matrix,telesc_position_matrix,geom,all_cam_axes,lookup_table_pkl,lookup_table_arrival_pkl,lookup_table_arrival_rms_pkl,eigen_vectors_pkl)
            print ('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
            print ('all template fit result:')
            print (f'current_event = {current_event}')
            print (f'truth_cam_x      = {truth_cam_x:0.3f}')
            print (f'truth_cam_y      = {truth_cam_y:0.3f}')
            print (f'fit_simul_temp_cam_x = {fit_simul_temp_cam_x:0.3f}')
            print (f'fit_simul_temp_cam_y = {fit_simul_temp_cam_y:0.3f}')
            print (f'truth_shower_core_x = {truth_shower_core_x:0.3f}')
            print (f'truth_shower_core_y = {truth_shower_core_y:0.3f}')
            print (f'fit_simul_temp_core_x = {fit_simul_temp_core_x*1000.:0.3f}')
            print (f'fit_simul_temp_core_y = {fit_simul_temp_core_y*1000.:0.3f}')
            print (f'truth_shower_energy = {truth_shower_energy:0.3f}')
            print (f'fit_simul_temp_energy = {fit_simul_temp_energy:0.3f}')
            print ('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    
            avg_indiv_line_impact = 0.
            avg_indiv_temp_impact = 0.
            avg_simul_temp_impact = 0.
            sum_weight = 0.
            for i in range(0,len(telesc_position_matrix)):
                tel_x = telesc_position_matrix[i][2]/1000.
                tel_y = telesc_position_matrix[i][3]/1000.
                avg_indiv_line_impact += (pow(fit_line_core_x-tel_x,2) + pow(fit_line_core_y-tel_y,2))
                avg_indiv_temp_impact += (pow(fit_temp_core_x-tel_x,2) + pow(fit_temp_core_y-tel_y,2))
                avg_simul_temp_impact += (pow(fit_simul_temp_core_x-tel_x,2) + pow(fit_simul_temp_core_y-tel_y,2))
                sum_weight += 1.
            avg_indiv_line_impact = pow(avg_indiv_line_impact/sum_weight,0.5)
            avg_indiv_temp_impact = pow(avg_indiv_temp_impact/sum_weight,0.5)
            avg_simul_temp_impact = pow(avg_simul_temp_impact/sum_weight,0.5)
            indiv_line_impact += [avg_indiv_line_impact]
            indiv_temp_impact += [avg_indiv_temp_impact]
            simul_temp_impact += [avg_simul_temp_impact]
    

            fit_simul_temp_tel_coord = [fit_simul_temp_cam_x,fit_simul_temp_cam_y]
            fit_simul_temp_array_coord = convert_tel_coord_to_array_coord(fit_simul_temp_tel_coord,tel_info)
            fit_simul_temp_evt_alt = fit_simul_temp_array_coord[0]
            fit_simul_temp_evt_az = fit_simul_temp_array_coord[1]
    
    
            indiv_line_fit_sky_err += [180./math.pi*pow(pow(fit_line_alt-truth_shower_alt,2)+pow(fit_line_az-truth_shower_az,2),0.5)]
            indiv_temp_fit_sky_err += [180./math.pi*pow(pow(fit_temp_evt_alt-truth_shower_alt,2)+pow(fit_temp_evt_az-truth_shower_az,2),0.5)]
            simul_temp_fit_sky_err += [180./math.pi*pow(pow(fit_simul_temp_evt_alt-truth_shower_alt,2)+pow(fit_simul_temp_evt_az-truth_shower_az,2),0.5)]
            all_truth_energy += [truth_shower_energy]
            indiv_temp_fit_energy += [fit_temp_energy]
            simul_temp_fit_energy += [fit_simul_temp_energy]
    
            image_sum_truth = None
            time_sum_truth = None
            image_sum_svd_indiv_line = None
            image_sum_svd_indiv_temp = None
            image_sum_svd_simul_temp = None
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
                evt_param = [fit_shower_cam_x,fit_shower_cam_y,fit_shower_core_x,fit_shower_core_y,math.log10(fit_shower_energy),delta_foci_time]
                evt_param = np.array(evt_param)
                svd_image_2d_truth_temp = create_svd_image(evt_param,tel_pos,all_cam_axes[i],geom,lookup_table_pkl,lookup_table_arrival_pkl,eigen_vectors_pkl)

                fit_shower_energy = fit_temp_energy
                fit_shower_cam_x = fit_line_cam_x
                fit_shower_cam_y = fit_line_cam_y
                fit_shower_core_x = fit_line_core_x
                fit_shower_core_y = fit_line_core_y
                evt_param = [fit_shower_cam_x,fit_shower_cam_y,fit_shower_core_x,fit_shower_core_y,math.log10(fit_shower_energy),delta_foci_time]
                evt_param = np.array(evt_param)
                svd_image_2d_indiv_line = create_svd_image(evt_param,tel_pos,all_cam_axes[i],geom,lookup_table_pkl,lookup_table_arrival_pkl,eigen_vectors_pkl)
    
                fit_shower_energy = fit_temp_energy
                fit_shower_cam_x = fit_temp_cam_x
                fit_shower_cam_y = fit_temp_cam_y
                fit_shower_core_x = fit_temp_core_x
                fit_shower_core_y = fit_temp_core_y
                evt_param = [fit_shower_cam_x,fit_shower_cam_y,fit_shower_core_x,fit_shower_core_y,math.log10(fit_shower_energy),delta_foci_time]
                evt_param = np.array(evt_param)
                svd_image_2d_indiv_temp = create_svd_image(evt_param,tel_pos,all_cam_axes[i],geom,lookup_table_pkl,lookup_table_arrival_pkl,eigen_vectors_pkl)
    
                fit_shower_energy = fit_simul_temp_energy
                fit_shower_cam_x = fit_simul_temp_cam_x
                fit_shower_cam_y = fit_simul_temp_cam_y
                fit_shower_core_x = fit_simul_temp_core_x
                fit_shower_core_y = fit_simul_temp_core_y
                evt_param = [fit_shower_cam_x,fit_shower_cam_y,fit_shower_core_x,fit_shower_core_y,math.log10(fit_shower_energy),delta_foci_time]
                evt_param = np.array(evt_param)
                svd_image_2d_simul_temp = create_svd_image(evt_param,tel_pos,all_cam_axes[i],geom,lookup_table_pkl,lookup_table_arrival_pkl,eigen_vectors_pkl)

                for row in range(0,num_rows):
                    for col in range(0,num_cols):
                        baseline_chi2 += pow(analysis_image_square[row,col],2)
                        line_chi2 += pow(analysis_image_square[row,col]-svd_image_2d_indiv_line[row,col],2)
                        temp_chi2 += pow(analysis_image_square[row,col]-svd_image_2d_simul_temp[row,col],2)

                if i==0:
                    image_sum_truth = analysis_image_square
                    time_sum_truth = analysis_time_square
                    image_sum_svd_indiv_line = svd_image_2d_indiv_line
                    image_sum_svd_indiv_temp = svd_image_2d_indiv_temp
                    image_sum_svd_simul_temp = svd_image_2d_simul_temp
                    image_sum_svd_truth_temp = svd_image_2d_truth_temp
                else:
                    image_sum_truth += analysis_image_square
                    time_sum_truth += analysis_time_square
                    image_sum_svd_indiv_line += svd_image_2d_indiv_line
                    image_sum_svd_indiv_temp += svd_image_2d_indiv_temp
                    image_sum_svd_simul_temp += svd_image_2d_simul_temp
                    image_sum_svd_truth_temp += svd_image_2d_truth_temp

            all_line_chi2 += [line_chi2/baseline_chi2]
            all_temp_chi2 += [temp_chi2/baseline_chi2]
            print (f'line_chi2/baseline_chi2 = {line_chi2/baseline_chi2}')
            print (f'temp_chi2/baseline_chi2 = {temp_chi2/baseline_chi2}')
    
    
    
            xmax = max(geom.pix_x)/u.m
            xmin = min(geom.pix_x)/u.m
            ymax = max(geom.pix_y)/u.m
            ymin = min(geom.pix_y)/u.m
            num_rows, num_cols = image_sum_truth.shape

            core_xmax = 1.
            core_xmin = -1.
            core_ymax = 1.
            core_ymin = -1.
    
            pointing_error = 180./math.pi*pow(pow(fit_simul_temp_evt_alt-truth_shower_alt,2)+pow(fit_simul_temp_evt_az-truth_shower_az,2),0.5)
            #pointing_error = 180./math.pi*pow(pow(fit_line_alt-truth_shower_alt,2)+pow(fit_line_az-truth_shower_az,2),0.5)
    
            if pointing_error>0.2: 

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
                fig.savefig(f'{ctapipe_output}/output_plots/sum_image_evt{current_event}_indiv_lines.png',bbox_inches='tight')
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
                fig.savefig(f'{ctapipe_output}/output_plots/sum_image_evt{current_event}_svd_truth_temp.png',bbox_inches='tight')
                axbig.remove()
    
                #fig.clf()
                #axbig = fig.add_subplot()
                #label_x = 'X'
                #label_y = 'Y'
                #axbig.set_xlabel(label_x)
                #axbig.set_ylabel(label_y)
                #im = axbig.imshow(image_sum_svd_indiv_temp,origin='lower',extent=(xmin,xmax,ymin,ymax))
                #line_x = np.linspace(xmin, xmax, 100)
                #for tel in range(0,len(list_img_a)):
                #    a = list_img_a[tel]
                #    b = list_img_b[tel]
                #    line_y = -(a*line_x + b)
                #    axbig.plot(line_x,line_y,color='w',alpha=0.3,linestyle='dashed')
                #axbig.set_xlim(xmin,xmax)
                #axbig.set_ylim(ymin,ymax)
                #axbig.scatter(truth_cam_x, -truth_cam_y, s=90, c='r', marker='+')
                #axbig.scatter(fit_temp_cam_x, -fit_temp_cam_y, s=90, c='g', marker='+')
                #cbar = fig.colorbar(im)
                #cbar.set_label('PE')
                #fig.savefig(f'{ctapipe_output}/output_plots/sum_image_evt{current_event}_svd_indiv_temp.png',bbox_inches='tight')
                #axbig.remove()
    
                fig.clf()
                axbig = fig.add_subplot()
                label_x = 'X'
                label_y = 'Y'
                axbig.set_xlabel(label_x)
                axbig.set_ylabel(label_y)
                im = axbig.imshow(image_sum_svd_simul_temp,origin='lower',extent=(xmin,xmax,ymin,ymax))
                line_x = np.linspace(xmin, xmax, 100)
                for tel in range(0,len(list_img_a)):
                    a = list_img_a[tel]
                    b = list_img_b[tel]
                    line_y = -(a*line_x + b)
                    axbig.plot(line_x,line_y,color='w',alpha=0.3,linestyle='dashed')
                axbig.set_xlim(xmin,xmax)
                axbig.set_ylim(ymin,ymax)
                axbig.scatter(truth_cam_x, -truth_cam_y, s=90, c='r', marker='+')
                axbig.scatter(fit_simul_temp_cam_x, -fit_simul_temp_cam_y, s=90, c='g', marker='+')
                cbar = fig.colorbar(im)
                cbar.set_label('PE')
                fig.savefig(f'{ctapipe_output}/output_plots/sum_image_evt{current_event}_svd_simul_temp.png',bbox_inches='tight')
                axbig.remove()
    
                #fig.clf()
                #axbig = fig.add_subplot()
                #label_x = 'X'
                #label_y = 'Y'
                #axbig.set_xlabel(label_x)
                #axbig.set_ylabel(label_y)
                #im = axbig.imshow(np.exp(temp_weight_skymap),origin='lower',cmap='magma',extent=(xmin,xmax,ymin,ymax))
                #line_x = np.linspace(xmin, xmax, 100)
                #for tel in range(0,len(list_img_a)):
                #    a = list_img_a[tel]
                #    b = list_img_b[tel]
                #    line_y = -(a*line_x + b)
                #    axbig.plot(line_x,line_y,color='w',alpha=0.3,linestyle='dashed')
                #axbig.set_xlim(xmin,xmax)
                #axbig.set_ylim(ymin,ymax)
                #axbig.scatter(truth_cam_x, -truth_cam_y, s=90, c='r', marker='+')
                #axbig.scatter(fit_temp_cam_x, -fit_temp_cam_y, s=90, c='g', marker='+')
                #cbar = fig.colorbar(im)
                #cbar.set_label('PE')
                #fig.savefig(f'{ctapipe_output}/output_plots/sum_image_evt{current_event}_temp_weight_skymap.png',bbox_inches='tight')
                #axbig.remove()
    
                #fig.clf()
                #axbig = fig.add_subplot()
                #label_x = 'X'
                #label_y = 'Y'
                #axbig.set_xlabel(label_x)
                #axbig.set_ylabel(label_y)
                #im = axbig.imshow(np.exp(temp_weight_grdmap),origin='lower',cmap='magma',extent=(core_xmin,core_xmax,core_ymin,core_ymax))
                #axbig.scatter(truth_shower_core_x/1000., truth_shower_core_y/1000., s=90, c='r', marker='+')
                #axbig.scatter(fit_temp_core_x, fit_temp_core_y, s=90, c='g', marker='+')
                #cbar = fig.colorbar(im)
                #cbar.set_label('PE')
                #fig.savefig(f'{ctapipe_output}/output_plots/sum_image_evt{current_event}_temp_weight_grdmap.png',bbox_inches='tight')
                #axbig.remove()
    
                plot_count += 1
                #exit()
    
            log_energy_axis = []
            for x in range(0,4):
                log_energy_axis += [pow(10.,-1+x*0.5)]
            hist_line_sky_err_vs_energy = get_average(indiv_temp_fit_energy,pow(np.array(indiv_line_fit_sky_err),2),log_energy_axis)
            hist_line_sky_err_vs_energy.yaxis = pow(np.array(hist_line_sky_err_vs_energy.yaxis),0.5)
            hist_temp_sky_err_vs_energy = get_average(indiv_temp_fit_energy,pow(np.array(indiv_temp_fit_sky_err),2),log_energy_axis)
            hist_temp_sky_err_vs_energy.yaxis = pow(np.array(hist_temp_sky_err_vs_energy.yaxis),0.5)
            hist_simul_temp_sky_err_vs_energy = get_average(simul_temp_fit_energy,pow(np.array(simul_temp_fit_sky_err),2),log_energy_axis)
            hist_simul_temp_sky_err_vs_energy.yaxis = pow(np.array(hist_simul_temp_sky_err_vs_energy.yaxis),0.5)
            print (f'hist_line_sky_err_vs_energy.yaxis = {hist_line_sky_err_vs_energy.yaxis}')
            print (f'hist_temp_sky_err_vs_energy.yaxis = {hist_temp_sky_err_vs_energy.yaxis}')
            print (f'hist_simul_temp_sky_err_vs_energy.yaxis = {hist_simul_temp_sky_err_vs_energy.yaxis}')
    
            fig.clf()
            axbig = fig.add_subplot()
            label_x = 'Energy'
            label_y = 'Sky coordinate error'
            axbig.set_xlabel(label_x)
            axbig.set_ylabel(label_y)
            axbig.scatter(indiv_temp_fit_energy, indiv_line_fit_sky_err, s=90, c='r', marker='+')
            axbig.scatter(indiv_temp_fit_energy, indiv_temp_fit_sky_err, s=90, c='g', marker='+')
            axbig.scatter(simul_temp_fit_energy, simul_temp_fit_sky_err, s=90, c='b', marker='+')
            axbig.set_xscale('log')
            #axbig.set_yscale('log')
            fig.savefig(f'{ctapipe_output}/output_plots/sky_error_vs_energy.png',bbox_inches='tight')
            axbig.remove()
            
            fig.clf()
            axbig = fig.add_subplot()
            label_x = 'Sum square open angles'
            label_y = 'Sky coordinate error'
            axbig.set_xlabel(label_x)
            axbig.set_ylabel(label_y)
            axbig.scatter(all_open_angle, indiv_line_fit_sky_err, s=90, c='r', marker='+')
            axbig.scatter(all_open_angle, indiv_temp_fit_sky_err, s=90, c='g', marker='+')
            axbig.scatter(all_open_angle, simul_temp_fit_sky_err, s=90, c='b', marker='+')
            axbig.set_xscale('log')
            #axbig.set_yscale('log')
            fig.savefig(f'{ctapipe_output}/output_plots/sky_error_vs_angle.png',bbox_inches='tight')
            axbig.remove()
            
            fig.clf()
            axbig = fig.add_subplot()
            label_x = 'template weight'
            label_y = 'Sky coordinate error'
            axbig.set_xlabel(label_x)
            axbig.set_ylabel(label_y)
            axbig.scatter(all_temp_weight, indiv_line_fit_sky_err, s=90, c='r', marker='+')
            axbig.scatter(all_temp_weight, indiv_temp_fit_sky_err, s=90, c='g', marker='+')
            axbig.scatter(all_temp_weight, simul_temp_fit_sky_err, s=90, c='b', marker='+')
            axbig.set_xscale('log')
            #axbig.set_yscale('log')
            fig.savefig(f'{ctapipe_output}/output_plots/sky_error_vs_temp_weight.png',bbox_inches='tight')
            axbig.remove()

            fig.clf()
            axbig = fig.add_subplot()
            label_x = 'impact'
            label_y = 'Sky coordinate error'
            axbig.set_xlabel(label_x)
            axbig.set_ylabel(label_y)
            axbig.scatter(indiv_line_impact, indiv_line_fit_sky_err, s=90, c='r', marker='+')
            axbig.scatter(indiv_temp_impact, indiv_temp_fit_sky_err, s=90, c='g', marker='+')
            axbig.scatter(simul_temp_impact, simul_temp_fit_sky_err, s=90, c='b', marker='+')
            axbig.set_xscale('log')
            #axbig.set_yscale('log')
            fig.savefig(f'{ctapipe_output}/output_plots/sky_error_vs_temp_impact.png',bbox_inches='tight')
            axbig.remove()
    
            fig.clf()
            axbig = fig.add_subplot()
            label_x = 'template chi2'
            label_y = 'Sky coordinate error'
            axbig.set_xlabel(label_x)
            axbig.set_ylabel(label_y)
            axbig.scatter(all_line_chi2, indiv_line_fit_sky_err, s=90, c='r', marker='+')
            axbig.scatter(all_temp_chi2, indiv_temp_fit_sky_err, s=90, c='g', marker='+')
            axbig.scatter(all_temp_chi2, simul_temp_fit_sky_err, s=90, c='b', marker='+')
            #axbig.set_xscale('log')
            #axbig.set_yscale('log')
            axbig.set_xlim(0.,2.)
            fig.savefig(f'{ctapipe_output}/output_plots/sky_error_vs_temp_chi2.png',bbox_inches='tight')
            axbig.remove()
    
            fig.clf()
            axbig = fig.add_subplot()
            label_x = 'Energy (truth)'
            label_y = 'Energy (reconstruction)'
            axbig.set_xlabel(label_x)
            axbig.set_ylabel(label_y)
            axbig.scatter(all_truth_energy, indiv_temp_fit_energy, s=90, c='r', marker='+')
            axbig.scatter(all_truth_energy, simul_temp_fit_energy, s=90, c='g', marker='+')
            axbig.set_xscale('log')
            axbig.set_yscale('log')
            fig.savefig(f'{ctapipe_output}/output_plots/energy_reconstruction_error.png',bbox_inches='tight')
            axbig.remove()
    
            testing_image_matrix = []
            testing_time_matrix = []
            testing_param_matrix = []
            testing_moment_matrix = []
            truth_shower_position_matrix = []
            all_cam_axes = []
            telesc_position_matrix = []

            #if plot_count==10: exit()




