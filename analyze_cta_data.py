
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
from load_cta_data import smooth_map
from load_cta_data import NeuralNetwork

ctapipe_output = os.environ.get("CTAPIPE_OUTPUT_PATH")
subprocess.call(['sh', './clean.sh'])


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

def create_svd_image(shower_params,tel_pos,cam_axes,geom,lookup_table,eigen_vectors):

    shower_energy = float(shower_params[0])
    shower_height = float(shower_params[1])
    shower_cam_x = float(shower_params[2])
    shower_cam_y = float(shower_params[3])
    shower_core_x = float(shower_params[4])
    shower_core_y = float(shower_params[5])

    tel_x = tel_pos[0]
    tel_y = tel_pos[1]
    x_axis = cam_axes[0]
    y_axis = cam_axes[1]
    shower_impact_x = shower_core_x - tel_x
    shower_impact_y = shower_core_y - tel_y

    shower_impact = pow(pow(shower_impact_x,2)+pow(shower_impact_y,2),0.5)
    shower_impact = pow(shower_impact/1000.,1)

    latent_space = []
    for r in range(0,len(lookup_table)):
        latent_space += [lookup_table[r].get_bin_content(shower_impact,math.log10(shower_energy))]
    latent_space = np.array(latent_space)
    
    evt_image = eigen_vectors.T @ latent_space
    evt_image_square = geom.image_to_cartesian_representation(evt_image)

    num_rows, num_cols = evt_image_square.shape

    for x_idx in range(0,num_cols):
        for y_idx in range(0,num_rows):
            if math.isnan(evt_image_square[y_idx,x_idx]): evt_image_square[y_idx,x_idx] = 0.
            if evt_image_square[y_idx,x_idx]<0.: evt_image_square[y_idx,x_idx] = 0.

    evt_image_derotate = np.zeros_like(evt_image_square)
    angle_rad = np.arctan2(shower_impact_y,shower_impact_x)
    if math.isnan(angle_rad): angle_rad = 0.
    evt_image_derotate = image_rotation(evt_image_square, cam_axes[0], cam_axes[1], angle_rad)

    evt_image_decenter = np.zeros_like(evt_image_square)
    shift_x = shower_cam_x
    shift_y = shower_cam_y
    evt_image_decenter = image_translation(evt_image_derotate, cam_axes[0], cam_axes[1], shift_x, shift_y)

    return evt_image_decenter

    #analysis_image_1d = geom.image_from_cartesian_representation(evt_image_decenter)
    #return analysis_image_1d


def sum_sqaure_difference_between_images(params,fix_params,var_name,tel_pos,all_cam_axes,geom,image_2d_matrix,param2image):

    if var_name=='cam_xy':
        cam_x = params[0]
        cam_y = params[1]
        energy = fix_params[0]
        height = fix_params[1]
        core_x = fix_params[2]
        core_y = fix_params[3]
    elif var_name=='energy':
        energy = params[0]
        height = params[1]
        cam_x = fix_params[0]
        cam_y = fix_params[1]
        core_x = fix_params[2]
        core_y = fix_params[3]
    elif var_name=='core_xy':
        core_x = params[0]
        core_y = params[1]
        energy = fix_params[0]
        height = fix_params[1]
        cam_x = fix_params[2]
        cam_y = fix_params[3]

    sum_chi2 = 0.
    sum_dist_sq = 0.
    sum_correlation = 1.
    for tel in range(0,len(image_2d_matrix)):
        evt_param = [energy,height,cam_x,cam_y,core_x,core_y]
        evt_param = np.array(evt_param)
        svd_image_2d = create_svd_image(evt_param,tel_pos[tel],all_cam_axes[tel],geom,param2image)
        data_image_2d = image_2d_matrix[tel]
        n_rows, n_cols = data_image_2d.shape
        sum_w = 0.
        for row in range(0,n_rows):
            for col in range(0,n_cols):
                diff = svd_image_2d[row,col] - data_image_2d[row,col]
                corr = svd_image_2d[row,col] * data_image_2d[row,col]
                sum_chi2 += diff*diff
                sum_correlation += corr
                sum_w += svd_image_2d[row,col]+data_image_2d[row,col]
        svg_avg_x, svd_avg_y, svd_sum_w = get_image_center(svd_image_2d, all_cam_axes[tel][0], all_cam_axes[tel][1])
        data_avg_x, data_avg_y, data_sum_w = get_image_center(data_image_2d, all_cam_axes[tel][0], all_cam_axes[tel][1])
        sum_dist_sq += sum_w*sum_w*(pow(svg_avg_x-data_avg_x,2)+pow(svd_avg_y-data_avg_y,2))
     
    return 1./sum_correlation
    #return sum_chi2
    #return sum_chi2*sum_dist_sq

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

            #tel1_image_2d = smooth_map(tel1_image_2d,tel1_x_axis,tel1_y_axis,50.)
            #tel2_image_2d = smooth_map(tel2_image_2d,tel2_x_axis,tel2_y_axis,50.)
            #tel1_image_1d = geom.image_from_cartesian_representation(tel1_image_2d)
            #tel2_image_1d = geom.image_from_cartesian_representation(tel2_image_2d)


            list_tel_x = []
            list_tel_x += [tel1_x]
            list_tel_x += [tel2_x]
            list_tel_y = []
            list_tel_y += [tel1_y]
            list_tel_y += [tel2_y]
            list_img_a = []
            list_img_b = []
            list_img_w = []
            a1, b1, w1 = fit_image_to_line(tel1_image_2d,tel1_x_axis,tel1_y_axis)
            a1T, b1T, w1T = fit_image_to_line(np.array(tel1_image_2d).transpose(),tel1_y_axis,tel1_x_axis)
            if w1<w1T:
                a1 = 1./a1T
                b1 = -b1T/a1T
                w1 = w1T
            list_img_a += [a1]
            list_img_b += [b1]
            list_img_w += [w1]
            a2, b2, w2 = fit_image_to_line(tel2_image_2d,tel2_x_axis,tel2_y_axis)
            a2T, b2T, w2T = fit_image_to_line(np.array(tel2_image_2d).transpose(),tel2_y_axis,tel2_x_axis)
            if w2<w2T:
                a2 = 1./a2T
                b2 = -b2T/a2T
                w2 = w2T
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

    return [avg_ls_evt_cam_x, avg_ls_evt_cam_y, avg_ls_evt_core_x, avg_ls_evt_core_y], list_all_img_a, list_all_img_b, list_all_img_w, list_pair_images, list_pair_a, list_pair_weight, list_pair_telpos

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
    return d

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sum_sqaure_distance_images_to_3D_line(params,line_a,tel_pos,all_cam_axes,image_2d_matrix,guiding_line_weight):

    cam_x = params[0]
    cam_y = params[1]
    core_x = params[2]
    core_y = params[3]

    sum_dist_sq_pixel = 0.
    sum_dist_sq_line = 0.
    sigma_cam_xy = 0.01
    for tel in range(0,len(image_2d_matrix)):

        rad2deg = 180./np.pi
        tel_x = tel_pos[tel][0]
        tel_y = tel_pos[tel][1]
        x_axis = all_cam_axes[tel][0]
        y_axis = all_cam_axes[tel][1]
        impact_x = core_x - tel_x
        impact_y = core_y - tel_y

        a, b = construct_line_on_focalplane(cam_x,cam_y,impact_x,impact_y)
        aT, bT = construct_line_on_focalplane(cam_y,cam_x,impact_y,impact_x)
        input_a = line_a[tel]
        diff_ang = np.arctan(abs((a-input_a)/(1.+a*input_a)))

        tel_image_2d = image_2d_matrix[tel]
        num_rows, num_cols = tel_image_2d.shape

        avg_pix_x, avg_pix_y, sum_w = get_image_center(tel_image_2d, x_axis, y_axis)

        vec_pix = np.array([avg_pix_x-cam_x,avg_pix_y-cam_y])
        vec_impact = -np.array([impact_x-cam_x,impact_y-cam_y])
        direction = 1./sigmoid(vec_pix.dot(vec_impact))

        sum_dist_sq_line += diff_ang*diff_ang*direction

        for col in range(0,num_cols):
            for row in range(0,num_rows):
                pix_x = x_axis[col]
                pix_y = y_axis[row]
                pix_w = tel_image_2d[row,col]
                if pix_w==0.: continue
                dist_sq = distance_square_point_to_line(pix_x,pix_y,a,b)
                dist_sq_T = distance_square_point_to_line(pix_y,pix_x,aT,bT)
                dist_sq = min(dist_sq,dist_sq_T)
                sum_dist_sq_pixel += (dist_sq/(sigma_cam_xy*sigma_cam_xy)*pow(pix_w,1))*direction

    return sum_dist_sq_pixel + guiding_line_weight*sum_dist_sq_line


def simultaneously_fit_3D_line_to_all_images(image_1d_matrix,line_a_matrix,init_params,bounds,telesc_position_matrix,geom,all_cam_axes,guiding_line_weight):

    tic = time.perf_counter()
    print ('simultaneously_fit_3D_line_to_all_images...')

    image_2d_matrix = []
    cam_axes = []
    telpos_matrix = []
    for tel in range(0,len(image_1d_matrix)):
        tel_image_1d = image_1d_matrix[tel]
        tel_image_2d = geom.image_to_cartesian_representation(tel_image_1d)
        num_rows, num_cols = tel_image_2d.shape
        for x_idx in range(0,num_cols):
            for y_idx in range(0,num_rows):
                if math.isnan(tel_image_2d[y_idx,x_idx]): tel_image_2d[y_idx,x_idx] = 0.
        tel_image_2d = np.array(tel_image_2d)
        image_2d_matrix += [tel_image_2d]
        cam_axes += [[all_cam_axes[tel][0],all_cam_axes[tel][1]]]
        telpos_matrix += [[telesc_position_matrix[tel][2],telesc_position_matrix[tel][3]]]

    fit_cam_x = init_params[0]
    fit_cam_y = init_params[1]
    fit_core_x = init_params[2]
    fit_core_y = init_params[3]

    stepsize = [0.01,0.01,5.,5.]
    params = [fit_cam_x,fit_cam_y,fit_core_x,fit_core_y]
    solution = minimize(
        sum_sqaure_distance_images_to_3D_line,
        x0=params,
        args=(line_a_matrix,telpos_matrix,cam_axes,image_2d_matrix,guiding_line_weight),
        bounds=bounds,
        method='L-BFGS-B',
        jac=None,
        options={'eps':stepsize},
    )
    fit_cam_x = solution['x'][0]
    fit_cam_y = solution['x'][1]
    fit_core_x = solution['x'][2]
    fit_core_y = solution['x'][3]

    toc = time.perf_counter()
    print (f'simultaneously_fit_3D_line_to_all_images in {toc - tic:0.4f} seconds')

    return [fit_cam_x,fit_cam_y,fit_core_x,fit_core_y]

def simultaneously_fit_3D_template_to_all_images(image_1d_matrix,init_params,bounds,telesc_position_matrix,geom,all_cam_axes,param2image):

    tic = time.perf_counter()
    print ('simultaneously_fit_3D_template_to_all_images...')

    image_2d_matrix = []
    for tel in range(0,len(image_1d_matrix)):
        image_2d = geom.image_to_cartesian_representation(image_1d_matrix[tel])
        num_rows, num_cols = image_2d.shape
        for x_idx in range(0,num_cols):
            for y_idx in range(0,num_rows):
                if math.isnan(image_2d[y_idx,x_idx]): image_2d[y_idx,x_idx] = 0.
        image_2d_matrix += [np.array(image_2d)]

    cam_axes = []
    telpos_matrix = []
    for tel in range(0,len(image_2d_matrix)):
        cam_axes += [[all_cam_axes[tel][0],all_cam_axes[tel][1]]]
        telpos_matrix += [[telesc_position_matrix[tel][2],telesc_position_matrix[tel][3]]]

    fit_energy = init_params[0]
    fit_height = init_params[1]
    fit_cam_x = init_params[2]
    fit_cam_y = init_params[3]
    fit_core_x = init_params[4]
    fit_core_y = init_params[5]
    bound_energy = bounds[0]
    bound_height = bounds[1]
    bound_cam_x = bounds[2]
    bound_cam_y = bounds[3]
    bound_core_x = bounds[4]
    bound_core_y = bounds[5]

    var_name = 'core_xy'
    var_params = [fit_core_x,fit_core_y]
    var_bounds = [bound_core_x,bound_core_y]
    fix_params = [fit_energy,fit_height,fit_cam_x,fit_cam_y]
    stepsize = 5.
    print (f'{var_name} stepsize = {stepsize}')
    solution = minimize(
        sum_sqaure_difference_between_images,
        x0=var_params,
        args=(fix_params,var_name,telpos_matrix,cam_axes,geom,image_2d_matrix,param2image),
        bounds=var_bounds,
        method='L-BFGS-B',
        #jac=None,
        #options={'eps':stepsize},
    )
    fit_core_x = solution['x'][0]
    fit_core_y = solution['x'][1]
    sum_chi2 = sum_sqaure_difference_between_images([fit_core_x,fit_core_y],fix_params,var_name,telpos_matrix,cam_axes,geom,image_2d_matrix,param2image)
    print (f'sum_chi2 = {sum_chi2}')
    print (f'init_core_x = {init_params[4]}')
    print (f'init_core_y = {init_params[5]}')
    print (f'fit_core_x = {fit_core_x}')
    print (f'fit_core_y = {fit_core_y}')

    var_name = 'cam_xy'
    var_params = [fit_cam_x,fit_cam_y]
    var_bounds = [bound_cam_x,bound_cam_y]
    fix_params = [fit_energy,fit_height,fit_core_x,fit_core_y]
    stepsize = (all_cam_axes[0][0][1]-all_cam_axes[0][0][0])*1.01
    print (f'{var_name} stepsize = {stepsize}')
    solution = minimize(
        sum_sqaure_difference_between_images,
        x0=var_params,
        args=(fix_params,var_name,telpos_matrix,cam_axes,geom,image_2d_matrix,param2image),
        bounds=var_bounds,
        method='L-BFGS-B',
        jac=None,
        options={'eps':stepsize},
    )
    fit_cam_x = solution['x'][0]
    fit_cam_y = solution['x'][1]
    sum_chi2 = sum_sqaure_difference_between_images([fit_cam_x,fit_cam_y],fix_params,var_name,telpos_matrix,cam_axes,geom,image_2d_matrix,param2image)
    print (f'sum_chi2 = {sum_chi2}')
    print (f'init_cam_x = {init_params[2]}')
    print (f'init_cam_y = {init_params[3]}')
    print (f'fit_cam_x = {fit_cam_x}')
    print (f'fit_cam_y = {fit_cam_y}')

    toc = time.perf_counter()
    print (f'simultaneously_fit_3D_template_to_all_images in {toc - tic:0.4f} seconds')

    return [fit_energy,fit_height,fit_cam_x,fit_cam_y,fit_core_x,fit_core_y]


testing_sample_path = []
testing_sample_path = [get_dataset_path('gamma_20deg_0deg_run835___cta-prod3-sct_desert-2150m-Paranal-SCT.simtel.gz')]
#with open('testing_sim_files.txt', 'r') as file:
#    for line in file:
#        testing_sample_path += [get_dataset_path(line.strip('\n'))]

testing_id_list = []
big_telesc_position_matrix = []
big_truth_shower_position_matrix = []
test_cam_axes = []
big_testing_image_matrix = []
big_testing_param_matrix = []
big_testing_hillas_matrix = []
for path in range(0,len(testing_sample_path)):
    source = SimTelEventSource(testing_sample_path[path], focal_length_choice='EQUIVALENT')
    subarray = source.subarray
    ob_keys = source.observation_blocks.keys()
    run_id = list(ob_keys)[0]
    print ('loading pickle testing sample data... ')
    output_filename = f'{ctapipe_output}/output_samples/testing_sample_run{run_id}.pkl'
    if not os.path.exists(output_filename):
        continue
    training_sample = pickle.load(open(output_filename, "rb"))

    testing_id_list += training_sample[0]
    big_telesc_position_matrix += training_sample[1]
    big_truth_shower_position_matrix = training_sample[2]
    test_cam_axes += training_sample[3]
    big_testing_image_matrix += training_sample[4]
    big_testing_param_matrix += training_sample[5]
    big_testing_hillas_matrix += training_sample[6]


print ('loading svd pickle data... ')
output_filename = f'{ctapipe_output}/output_machines/lookup_table.pkl'
lookup_table_pkl = pickle.load(open(output_filename, "rb"))

output_filename = f'{ctapipe_output}/output_machines/eigen_vectors.pkl'
eigen_vectors_pkl = pickle.load(open(output_filename, "rb"))


print ('analyzing test data... ')
collected_all_images = False
current_run = 0
current_event = 0
testing_image_matrix = []
testing_param_matrix = []
testing_hillas_matrix = []
truth_shower_position_matrix = []
all_cam_axes = []
telesc_position_matrix = []

indiv_fit_alt_err = []
indiv_fit_az_err = []
simul_fit_alt_err = []
simul_fit_az_err = []
indiv_fit_core_x_err = []
indiv_fit_core_y_err = []
simul_fit_core_x_err = []
simul_fit_core_y_err = []
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
        testing_param_matrix += [big_testing_param_matrix[img]]
        testing_hillas_matrix += [big_testing_hillas_matrix[img]]
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

        if len(testing_image_matrix)<=1: 
            testing_image_matrix = []
            testing_param_matrix = []
            testing_hillas_matrix = []
            truth_shower_position_matrix = []
            all_cam_axes = []
            telesc_position_matrix = []
            continue
        if len(testing_image_matrix)>=10: 
            testing_image_matrix = []
            testing_param_matrix = []
            testing_hillas_matrix = []
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
        truth_shower_energy = truth_shower_position_matrix[0][4]
        truth_shower_height = 20000.

        truth_array_coord = [truth_shower_alt,truth_shower_az,truth_shower_core_x,truth_shower_core_y]
        truth_tel_coord = convert_array_coord_to_tel_coord(truth_array_coord,tel_info)
        truth_cam_x = truth_tel_coord[0]
        truth_cam_y = truth_tel_coord[1]

        ls_evt_line_vec, list_img_a, list_img_b, list_img_w, list_pair_images, list_pair_a, list_pair_weight, list_pair_telpos = fit_lines_to_individual_images(testing_image_matrix,telesc_position_matrix,geom,all_cam_axes)
        fit_indiv_evt_cam_x = float(ls_evt_line_vec[0])
        fit_indiv_evt_cam_y = float(ls_evt_line_vec[1])
        fit_indiv_evt_core_x = float(ls_evt_line_vec[2])
        fit_indiv_evt_core_y = float(ls_evt_line_vec[3])
        fit_indiv_tel_coord = [fit_indiv_evt_cam_x,fit_indiv_evt_cam_y]
        print ('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print ('individual line fit result:')
        print (f'current_event = {current_event}')
        print (f'truth_cam_x      = {truth_cam_x:0.3f}')
        print (f'truth_cam_y      = {truth_cam_y:0.3f}')
        print (f'fit_indiv_evt_cam_x = {fit_indiv_evt_cam_x:0.3f}')
        print (f'fit_indiv_evt_cam_y = {fit_indiv_evt_cam_y:0.3f}')
        print (f'truth_shower_core_x = {truth_shower_core_x:0.3f}')
        print (f'truth_shower_core_y = {truth_shower_core_y:0.3f}')
        print (f'fit_indiv_evt_core_x = {fit_indiv_evt_core_x:0.3f}')
        print (f'fit_indiv_evt_core_y = {fit_indiv_evt_core_y:0.3f}')
        print ('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        
        fit_indiv_array_coord = convert_tel_coord_to_array_coord(fit_indiv_tel_coord,tel_info)
        fit_indiv_evt_alt = fit_indiv_array_coord[0]
        fit_indiv_evt_az = fit_indiv_array_coord[1]

        # Sort based on the values of elements in the first list
        combined_lists = list(zip(list_pair_weight, list_pair_images, list_pair_telpos))
        sorted_lists = sorted(combined_lists, key=lambda x: x[0], reverse=True)
        list_pair_weight, list_pair_images, list_pair_telpos = zip(*sorted_lists)

        init_cam_x = fit_indiv_evt_cam_x
        init_cam_y = fit_indiv_evt_cam_y
        init_core_x = fit_indiv_evt_core_x
        init_core_y = fit_indiv_evt_core_y
        simul_fit_image_matrix = []
        simul_fit_line_a_matrix = []
        simul_fit_tel_pos = []
        simul_fit_cam_axes = []
        max_weight = list_pair_weight[0]
        simult_2d_solution = None
        for tp in range(0,len(list_pair_weight)):
            pair_weight = list_pair_weight[tp]
            if pair_weight<max_weight*0.5: continue
            simul_fit_image_matrix += [np.array(list_pair_images[tp][0])*pair_weight]
            simul_fit_image_matrix += [np.array(list_pair_images[tp][1])*pair_weight]
            simul_fit_line_a_matrix += [list_pair_a[tp][0]]
            simul_fit_line_a_matrix += [list_pair_a[tp][1]]
            simul_fit_tel_pos += [list_pair_telpos[tp][0]]
            simul_fit_tel_pos += [list_pair_telpos[tp][1]]
            simul_fit_cam_axes += [all_cam_axes[0]]
            simul_fit_cam_axes += [all_cam_axes[0]]
        guiding_line_weight = [1.0,0.1,0.0]
        for lw in range(0,len(guiding_line_weight)):
            init_params = [init_cam_x,init_cam_y,init_core_x,init_core_y]
            cam_x_bound = (init_cam_x-0.1,init_cam_x+0.1)
            cam_y_bound = (init_cam_y-0.1,init_cam_y+0.1)
            core_x_bound = (init_core_x-1e3,init_core_x+1e3)
            core_y_bound = (init_core_y-1e3,init_core_y+1e3)
            bounds = [cam_x_bound,cam_y_bound,core_x_bound,core_y_bound]
            simult_2d_solution = simultaneously_fit_3D_line_to_all_images(simul_fit_image_matrix,simul_fit_line_a_matrix,init_params,bounds,simul_fit_tel_pos,geom,simul_fit_cam_axes,guiding_line_weight[lw])
            init_cam_x = float(simult_2d_solution[0])
            init_cam_y = float(simult_2d_solution[1])
            init_core_x = float(simult_2d_solution[2])
            init_core_y = float(simult_2d_solution[3])
            print ('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
            print (f'guiding_line_weight = {guiding_line_weight[lw]}')
            print (f'truth_cam_x     = {truth_cam_x:0.3f}')
            print (f'init_cam_x      = {init_cam_x:0.3f}')
            print (f'truth_cam_y     = {truth_cam_y:0.3f}')
            print (f'init_cam_y      = {init_cam_y:0.3f}')
            print (f'truth_shower_core_x = {truth_shower_core_x:0.3f}')
            print (f'init_core_x         = {init_core_x:0.3f}')
            print (f'truth_shower_core_y = {truth_shower_core_y:0.3f}')
            print (f'init_core_y         = {init_core_y:0.3f}')
            print ('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        fit_all_line_evt_cam_x = float(simult_2d_solution[0])
        fit_all_line_evt_cam_y = float(simult_2d_solution[1])
        fit_all_line_evt_core_x = float(simult_2d_solution[2])
        fit_all_line_evt_core_y = float(simult_2d_solution[3])
        fit_all_tel_coord = [fit_all_line_evt_cam_x,fit_all_line_evt_cam_y]
        fit_all_array_coord = convert_tel_coord_to_array_coord(fit_all_tel_coord,tel_info)
        fit_all_line_evt_alt = fit_all_array_coord[0]
        fit_all_line_evt_az = fit_all_array_coord[1]
        print ('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print ('simultaneous line fit result:')
        print (f'current_event = {current_event}')
        print (f'truth_shower_core_x = {truth_shower_core_x:0.3f}')
        print (f'truth_shower_core_y = {truth_shower_core_y:0.3f}')
        print (f'fit_all_line_evt_core_x = {fit_all_line_evt_core_x:0.3f}')
        print (f'fit_all_line_evt_core_y = {fit_all_line_evt_core_y:0.3f}')
        print ('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')


        indiv_fit_alt_err += [fit_indiv_evt_alt-truth_shower_alt]
        indiv_fit_az_err += [fit_indiv_evt_az-truth_shower_az]
        simul_fit_alt_err += [fit_all_line_evt_alt-truth_shower_alt]
        simul_fit_az_err += [fit_all_line_evt_az-truth_shower_az]
        indiv_fit_core_x_err += [fit_indiv_evt_core_x-truth_shower_core_x]
        indiv_fit_core_y_err += [fit_indiv_evt_core_y-truth_shower_core_y]
        simul_fit_core_x_err += [fit_all_line_evt_core_x-truth_shower_core_x]
        simul_fit_core_y_err += [fit_all_line_evt_core_y-truth_shower_core_y]



        image_sum_truth = None
        image_sum_svd = None
        for i in range(0,len(testing_param_matrix)):

            tel_alt = telesc_position_matrix[i][0]
            tel_az = telesc_position_matrix[i][1]
            tel_x = telesc_position_matrix[i][2]
            tel_y = telesc_position_matrix[i][3]

            fit_shower_energy = truth_shower_energy
            fit_shower_height = truth_shower_height
            #fit_shower_energy = avg_evt_energy
            #fit_shower_height = 15000.

            fit_shower_cam_x = truth_cam_x
            fit_shower_cam_y = truth_cam_y
            fit_shower_core_x = truth_shower_core_x
            fit_shower_core_y = truth_shower_core_y
            #fit_shower_cam_x = fit_all_temp_evt_cam_x
            #fit_shower_cam_y = fit_all_temp_evt_cam_y
            #fit_shower_core_x = fit_all_temp_evt_core_x
            #fit_shower_core_y = fit_all_temp_evt_core_y

            analysis_image_1d = testing_image_matrix[i]
            analysis_image_square = geom.image_to_cartesian_representation(analysis_image_1d)

            tel_pos = [tel_x,tel_y]

            evt_param = [fit_shower_energy,fit_shower_height,fit_shower_cam_x,fit_shower_cam_y,fit_shower_core_x,fit_shower_core_y]
            evt_param = np.array(evt_param)

            print (f'creating SVD image...')
            svd_image_2d = create_svd_image(evt_param,tel_pos,all_cam_axes[i],geom,lookup_table_pkl,eigen_vectors_pkl)

            if i==0:
                image_sum_truth = analysis_image_square
                image_sum_svd = svd_image_2d
            else:
                image_sum_truth += analysis_image_square
                image_sum_svd += svd_image_2d



        xmax = max(geom.pix_x)/u.m
        xmin = min(geom.pix_x)/u.m
        ymax = max(geom.pix_y)/u.m
        ymin = min(geom.pix_y)/u.m

        fig.clf()
        axbig = fig.add_subplot()
        label_x = 'X'
        label_y = 'Y'
        axbig.set_xlabel(label_x)
        axbig.set_ylabel(label_y)
        im = axbig.imshow(image_sum_svd,origin='lower',extent=(xmin,xmax,ymin,ymax))
        cbar = fig.colorbar(im)
        cbar.set_label('PE')
        axbig.scatter(fit_all_line_evt_cam_x, -fit_all_line_evt_cam_y, s=90, c='r', marker='+')
        fig.savefig(f'{ctapipe_output}/output_plots/sum_image_evt{current_event}_svd.png',bbox_inches='tight')
        axbig.remove()

        fig.clf()
        axbig = fig.add_subplot()
        label_x = 'X'
        label_y = 'Y'
        axbig.set_xlabel(label_x)
        axbig.set_ylabel(label_y)
        im = axbig.imshow(image_sum_truth,origin='lower',extent=(xmin,xmax,ymin,ymax))
        line_x = np.linspace(xmin, xmax, 100)
        for tel in range(0,len(testing_image_matrix)):
            tel_x = telesc_position_matrix[tel][2]
            tel_y = telesc_position_matrix[tel][3]
            evt_impact_x = fit_all_line_evt_core_x - tel_x
            evt_impact_y = fit_all_line_evt_core_y - tel_y
            a, b = construct_line_on_focalplane(fit_all_line_evt_cam_x,fit_all_line_evt_cam_y,evt_impact_x,evt_impact_y)
            line_y = -(a*line_x + b)
            axbig.plot(line_x,line_y,color='w',alpha=0.3,linestyle='dashed')
        axbig.set_xlim(xmin,xmax)
        axbig.set_ylim(ymin,ymax)
        axbig.scatter(fit_all_line_evt_cam_x, -fit_all_line_evt_cam_y, s=90, c='r', marker='+')
        cbar = fig.colorbar(im)
        cbar.set_label('PE')
        fig.savefig(f'{ctapipe_output}/output_plots/sum_image_evt{current_event}_simult_lines.png',bbox_inches='tight')
        axbig.remove()

        fig.clf()
        axbig = fig.add_subplot()
        label_x = 'X'
        label_y = 'Y'
        axbig.set_xlabel(label_x)
        axbig.set_ylabel(label_y)
        im = axbig.imshow(image_sum_truth,origin='lower',extent=(xmin,xmax,ymin,ymax))
        line_x = np.linspace(xmin, xmax, 100)
        for tel in range(0,len(testing_image_matrix)):
            tel_x = telesc_position_matrix[tel][2]
            tel_y = telesc_position_matrix[tel][3]
            evt_impact_x = truth_shower_core_x - tel_x
            evt_impact_y = truth_shower_core_y - tel_y
            a, b = construct_line_on_focalplane(truth_cam_x,truth_cam_y,evt_impact_x,evt_impact_y)
            line_y = -(a*line_x + b)
            axbig.plot(line_x,line_y,color='w',alpha=0.3,linestyle='dashed')
        axbig.set_xlim(xmin,xmax)
        axbig.set_ylim(ymin,ymax)
        axbig.scatter(truth_cam_x, -truth_cam_y, s=90, c='r', marker='+')
        cbar = fig.colorbar(im)
        cbar.set_label('PE')
        fig.savefig(f'{ctapipe_output}/output_plots/sum_image_evt{current_event}_truth_lines.png',bbox_inches='tight')
        axbig.remove()

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
        axbig.scatter(fit_indiv_evt_cam_x, -fit_indiv_evt_cam_y, s=90, c='r', marker='+')
        cbar = fig.colorbar(im)
        cbar.set_label('PE')
        fig.savefig(f'{ctapipe_output}/output_plots/sum_image_evt{current_event}_indiv_lines.png',bbox_inches='tight')
        axbig.remove()


        testing_image_matrix = []
        testing_param_matrix = []
        testing_hillas_matrix = []
        truth_shower_position_matrix = []
        all_cam_axes = []
        telesc_position_matrix = []

        #exit()


fig.clf()
axbig = fig.add_subplot()
label_x = 'X'
label_y = 'Y'
axbig.set_xlabel(label_x)
axbig.set_ylabel(label_y)
axbig.scatter(indiv_fit_alt_err, indiv_fit_az_err, s=90, c='r', marker='+')
axbig.scatter(simul_fit_alt_err, simul_fit_az_err, s=90, c='k', marker='+')
fig.savefig(f'{ctapipe_output}/output_plots/sky_reconstruction_error.png',bbox_inches='tight')
axbig.remove()

fig.clf()
axbig = fig.add_subplot()
label_x = 'X'
label_y = 'Y'
axbig.set_xlabel(label_x)
axbig.set_ylabel(label_y)
axbig.scatter(indiv_fit_core_x_err, indiv_fit_core_y_err, s=90, c='r', marker='+')
axbig.scatter(simul_fit_core_x_err, simul_fit_core_y_err, s=90, c='k', marker='+')
fig.savefig(f'{ctapipe_output}/output_plots/core_reconstruction_error.png',bbox_inches='tight')
axbig.remove()



