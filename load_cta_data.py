import os
import subprocess
import glob

import time
import math
import numpy as np
from astropy import units as u
import pickle
from matplotlib import pyplot as plt

from ctapipe import utils
from ctapipe.utils.datasets import get_dataset_path
from ctapipe.calib import CameraCalibrator
from ctapipe.image import hillas_parameters, tailcuts_clean, ImageProcessor
from ctapipe.reco import ShowerProcessor
from ctapipe.io import EventSource, SimTelEventSource, HDF5TableWriter
from ctapipe.visualization import CameraDisplay
from traitlets.config import Config

ctapipe_output = os.environ.get("CTAPIPE_OUTPUT_PATH")


def find_brightest_pixels(tel_array):
    brightest_tel_array = []
    for tel in tel_array.keys():
        #print (f'tel = {tel}')
        brightest_pixel = np.argmax(tel_array[tel].waveform[0].sum(axis=1))
        #print (f'brightest_pixel = {brightest_pixel}')
        brightest_tel_array += [[tel,brightest_pixel]]
    return brightest_tel_array

def rank_brightest_telescope(tel_array):
    brightest_tel_array = find_brightest_pixels(tel_array)
    brightest_sum_waveform = 0.
    brightest_tel_key = 0
    for tel1 in range(0,len(brightest_tel_array)):
        tel1_key = brightest_tel_array[tel1][0]
        tel1_pix = brightest_tel_array[tel1][1]
        sum_waveform_tel1 = tel_array[tel1_key].waveform[0,tel1_pix].sum()
        for tel2 in range(tel1+1,len(brightest_tel_array)):
            tel2_key = brightest_tel_array[tel2][0]
            tel2_pix = brightest_tel_array[tel2][1]
            sum_waveform_tel2 = tel_array[tel2_key].waveform[0,tel2_pix].sum()
            if sum_waveform_tel1<sum_waveform_tel2:
                tmp_key = tel1_key
                tmp_pix = tel1_pix
                tel1_key = tel2_key
                tel1_pix = tel2_pix
                tel2_key = tmp_key
                tel2_pix = tmp_pix
                brightest_tel_array[tel1][0] = tel1_key
                brightest_tel_array[tel1][1] = tel1_pix
                brightest_tel_array[tel2][0] = tel2_key
                brightest_tel_array[tel2][1] = tel2_pix
    return brightest_tel_array

def smooth_image(image_data,xaxis,yaxis,mode):

    image_smooth = np.zeros_like(image_data)

    bin_size = xaxis[1]-xaxis[0]
    max_x = max(xaxis)
    min_x = min(xaxis)

    kernel_radius = (max_x-min_x)/mode

    kernel_pix_size = int(kernel_radius/bin_size)
    gaus_norm = 2.*np.pi*kernel_radius*kernel_radius
    image_kernel = np.zeros_like(image_data)
    central_bin_x = int(len(xaxis)/2)
    central_bin_y = int(len(yaxis)/2)
    for idx_x in range(0,len(xaxis)):
        for idx_y in range(0,len(yaxis)):
            pix_x = xaxis[idx_x]
            pix_y = yaxis[idx_y]
            distance = pow(pix_x*pix_x+pix_y*pix_y,0.5)
            pix_content = np.exp(-(distance*distance)/(2.*kernel_radius*kernel_radius))
            image_kernel[idx_y,idx_x] = pix_content/gaus_norm

    #fft_mtx_kernel = np.fft.fft2(image_kernel)
    #fft_mtx_data = np.fft.fft2(image_data)
    #result_fft = fft_mtx_kernel * fft_mtx_data
    #image_smooth = np.fft.ifft2(result_fft).real

    kernel_norm = np.sum(image_kernel)
    for idx_x1 in range(0,len(xaxis)):
        for idx_y1 in range(0,len(yaxis)):
            image_smooth[idx_y1,idx_x1] = 0.
            for idx_x2 in range(idx_x1-2*kernel_pix_size,idx_x1+2*kernel_pix_size):
                for idx_y2 in range(idx_y1-2*kernel_pix_size,idx_y1+2*kernel_pix_size):
                    if idx_x2<0: continue
                    if idx_y2<0: continue
                    if idx_x2>=len(xaxis): continue
                    if idx_y2>=len(yaxis): continue
                    old_content = image_data[idx_y2,idx_x2]
                    scale = image_kernel[central_bin_y+idx_y2-idx_y1,central_bin_x+idx_x2-idx_x1]
                    image_smooth[idx_y1,idx_x1] += old_content*scale/kernel_norm

    return image_smooth

def smooth_time_image(time_data,image_data,xaxis,yaxis,mode):

    time_smooth = np.zeros_like(time_data)

    bin_size = xaxis[1]-xaxis[0]
    max_x = max(xaxis)
    min_x = min(xaxis)

    kernel_radius = (max_x-min_x)/mode
    kernel_pix_size = int(kernel_radius/bin_size)

    for idx_x1 in range(0,len(xaxis)):
        for idx_y1 in range(0,len(yaxis)):
            time_smooth[idx_y1,idx_x1] = 0.
            if image_data[idx_y1,idx_x1]==0.: continue
            sum_weight = 0.
            for idx_x2 in range(idx_x1-2*kernel_pix_size,idx_x1+2*kernel_pix_size):
                for idx_y2 in range(idx_y1-2*kernel_pix_size,idx_y1+2*kernel_pix_size):
                    if idx_x2<0: continue
                    if idx_y2<0: continue
                    if idx_x2>=len(xaxis): continue
                    if idx_y2>=len(yaxis): continue
                    time = time_data[idx_y2,idx_x2]
                    weight = image_data[idx_y2,idx_x2]
                    time_smooth[idx_y1,idx_x1] += time*weight
                    sum_weight += weight
            time_smooth[idx_y1,idx_x1] = time_smooth[idx_y1,idx_x1]/sum_weight

    return time_smooth

def renormalize_background(image_data,image_mask):

    num_rows, num_cols = image_data.shape

    pix_mean = 0.
    pix_cnt = 0.
    for x_idx in range(0,num_cols):
        for y_idx in range(0,num_rows):
            if image_data[y_idx,x_idx]==0.: continue
            if image_mask[y_idx,x_idx]==1: continue
            pix_mean += image_data[y_idx,x_idx]
            pix_cnt += 1.
    pix_mean = pix_mean/pix_cnt

    for x_idx in range(0,num_cols):
        for y_idx in range(0,num_rows):
            if image_data[y_idx,x_idx]==0.: continue
            image_data[y_idx,x_idx] += -1.*pix_mean

def my_clean_image(image_data,image_mask):

    num_rows, num_cols = image_data.shape
    image_clean = np.zeros_like(image_data)

    for x_idx in range(0,num_cols):
        for y_idx in range(0,num_rows):
            if image_data[y_idx,x_idx]==0.: continue
            if image_mask[y_idx,x_idx]!=1: 
                image_clean[y_idx,x_idx] = 0.
            else:
                image_clean[y_idx,x_idx] = image_data[y_idx,x_idx]
            if image_data[y_idx,x_idx]<0.:
                image_clean[y_idx,x_idx] = 0.

    return image_clean

def find_mask(image_data):

    image_mask = np.zeros_like(image_data)
    mask_threshold = 5.
    #mask_threshold = 4.5

    num_rows, num_cols = image_data.shape

    pix_mean = 0.
    pix_cnt = 0.
    for x_idx in range(0,num_cols):
        for y_idx in range(0,num_rows):
            if image_data[y_idx,x_idx]==0.: continue
            if image_mask[y_idx,x_idx]==1: continue
            pix_mean += image_data[y_idx,x_idx]
            pix_cnt += 1.
    pix_mean = pix_mean/pix_cnt

    pix_rms = 0.
    for x_idx in range(0,num_cols):
        for y_idx in range(0,num_rows):
            if image_data[y_idx,x_idx]==0.: continue
            if image_mask[y_idx,x_idx]==1: continue
            pix_rms += pow(image_data[y_idx,x_idx]-pix_mean,2)
    pix_rms = pow(pix_rms/pix_cnt,0.5)

    for x_idx in range(0,num_cols):
        for y_idx in range(0,num_rows):
            if image_data[y_idx,x_idx]==0.: continue
            if image_mask[y_idx,x_idx]==1: continue
            significance = (image_data[y_idx,x_idx]-pix_mean)/pix_rms
            if significance>mask_threshold:
                image_mask[y_idx,x_idx] = 1

    return image_mask

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
        return 0., 0., np.inf

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
    if chi2==0.:
        return 0., 0., np.inf
    else:
        return a, b, np.trace(w)/chi2

def find_image_moments_guided(input_image_2d, input_time_2d, x_axis, y_axis, guided=False, arrival_x=0., arrival_y=0.):

    num_rows, num_cols = input_image_2d.shape

    image_center_x = 0.
    image_center_y = 0.
    image_size = 0.
    for x_idx in range(0,num_cols):
        for y_idx in range(0,num_rows):
            image_size += input_image_2d[y_idx,x_idx]
            image_center_x += x_axis[x_idx]*input_image_2d[y_idx,x_idx]
            image_center_y += y_axis[y_idx]*input_image_2d[y_idx,x_idx]
    image_center_x = image_center_x/image_size
    image_center_y = image_center_y/image_size
    #print (f'image_size = {image_size}')
    #print (f'image_center_x = {image_center_x}')
    #print (f'image_center_y = {image_center_y}')

    a, b, w = fit_image_to_line(input_image_2d,x_axis,y_axis)
    aT, bT, wT = fit_image_to_line(np.array(input_image_2d).transpose(),y_axis,x_axis)
    if w<wT:
        a = 1./aT
        b = -bT/aT
        w = wT

    cov_xx = 0.
    cov_xy = 0.
    cov_yx = 0.
    cov_yy = 0.
    for x_idx in range(0,num_cols):
        for y_idx in range(0,num_rows):
            diff_x = x_axis[x_idx]-image_center_x
            diff_y = y_axis[y_idx]-image_center_y
            weight = input_image_2d[y_idx,x_idx]
            cov_xx += diff_x*diff_x*weight
            cov_xy += diff_x*diff_y*weight
            cov_yx += diff_y*diff_x*weight
            cov_yy += diff_y*diff_y*weight
    cov_xx = cov_xx/image_size
    cov_xy = cov_xy/image_size
    cov_yx = cov_yx/image_size
    cov_yy = cov_yy/image_size

    covariance_matrix = np.array([[cov_xx, cov_xy], [cov_yx, cov_yy]])
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    semi_major_sq = eigenvalues[0]
    semi_minor_sq = eigenvalues[1]
    if semi_minor_sq>semi_major_sq:
        x = semi_minor_sq
        semi_minor_sq = semi_major_sq
        semi_major_sq = x
    dist_foci = pow(semi_major_sq-semi_minor_sq,0.5) 

    angle = np.arctan(a)
    delta_x = dist_foci*np.cos(angle)
    delta_y = dist_foci*np.sin(angle)
    foci_1_x = image_center_x + delta_x
    foci_1_y = image_center_y + delta_y
    foci_2_x = image_center_x - delta_x
    foci_2_y = image_center_y - delta_y

    foci_1_time = 0.
    foci_2_time = 0.
    center_time = 0.
    sum_weight_1 = 0.
    sum_weight_2 = 0.
    sum_weight_c = 0.
    for x_idx in range(0,num_cols):
        for y_idx in range(0,num_rows):
            diff_1_x = x_axis[x_idx]-foci_1_x
            diff_1_y = y_axis[y_idx]-foci_1_y
            diff_1_r = pow(diff_1_x*diff_1_x + diff_1_y*diff_1_y,0.5)
            diff_2_x = x_axis[x_idx]-foci_2_x
            diff_2_y = y_axis[y_idx]-foci_2_y
            diff_2_r = pow(diff_2_x*diff_2_x + diff_2_y*diff_2_y,0.5)
            diff_c_x = x_axis[x_idx]-image_center_x
            diff_c_y = y_axis[y_idx]-image_center_y
            diff_c_r = pow(diff_c_x*diff_c_x + diff_c_y*diff_c_y,0.5)
            weight = input_image_2d[y_idx,x_idx]
            time = input_time_2d[y_idx,x_idx]
            if diff_1_r<0.03:
                foci_1_time += time*weight
                sum_weight_1 += weight
            if diff_2_r<0.03:
                foci_2_time += time*weight
                sum_weight_2 += weight
            if diff_c_r<0.03:
                center_time += time*weight
                sum_weight_c += weight
    if sum_weight_1>0.:
        foci_1_time = foci_1_time/sum_weight_1
    if sum_weight_2>0.:
        foci_2_time = foci_2_time/sum_weight_2
    if sum_weight_c>0.:
        center_time = center_time/sum_weight_c

    image_foci_x1 = 0.
    image_foci_y1 = 0.
    image_foci_x2 = 0.
    image_foci_y2 = 0.
    delta_foci_time = 0.
    if guided:
        diff_1_x = arrival_x-foci_1_x
        diff_1_y = arrival_y-foci_1_y
        diff_1_r = pow(diff_1_x*diff_1_x + diff_1_y*diff_1_y,0.5)
        diff_2_x = arrival_x-foci_2_x
        diff_2_y = arrival_y-foci_2_y
        diff_2_r = pow(diff_2_x*diff_2_x + diff_2_y*diff_2_y,0.5)
        if diff_1_r<diff_2_r:
            center_time = foci_2_time
            image_foci_x1 = float(foci_1_x)
            image_foci_y1 = float(foci_1_y)
            image_foci_x2 = float(foci_2_x)
            image_foci_y2 = float(foci_2_y)
            delta_foci_time = foci_2_time-foci_1_time
        else:
            center_time = foci_1_time
            image_foci_x1 = float(foci_2_x)
            image_foci_y1 = float(foci_2_y)
            image_foci_x2 = float(foci_1_x)
            image_foci_y2 = float(foci_1_y)
            delta_foci_time = foci_1_time-foci_2_time
    else:
        if foci_1_time<foci_2_time:
            center_time = foci_2_time
            image_foci_x1 = float(foci_1_x)
            image_foci_y1 = float(foci_1_y)
            image_foci_x2 = float(foci_2_x)
            image_foci_y2 = float(foci_2_y)
            delta_foci_time = foci_2_time-foci_1_time
        else:
            center_time = foci_1_time
            image_foci_x1 = float(foci_2_x)
            image_foci_y1 = float(foci_2_y)
            image_foci_x2 = float(foci_1_x)
            image_foci_y2 = float(foci_1_y)
            delta_foci_time = foci_1_time-foci_2_time


    return image_center_x, image_center_y, image_foci_x1, image_foci_y1, image_foci_x2, image_foci_y2, center_time, delta_foci_time, semi_major_sq, semi_minor_sq

def image_translation(input_image_2d, x_axis, y_axis, shift_x, shift_y):

    num_rows, num_cols = input_image_2d.shape
    min_pix_x = x_axis[0]
    max_pix_x = x_axis[len(x_axis)-1]
    min_pix_y = y_axis[0]
    max_pix_y = y_axis[len(y_axis)-1]
    delta_pix_x = (max_pix_x-min_pix_x)/float(num_cols)
    delta_pix_y = (max_pix_y-min_pix_y)/float(num_rows)

    image_shift = np.zeros_like(input_image_2d)
    shift_row = round(shift_y/delta_pix_y) 
    shift_col = round(shift_x/delta_pix_x)
    for x_idx in range(0,num_cols):
        for y_idx in range(0,num_rows):
            if y_idx+shift_row<0: continue
            if x_idx+shift_col<0: continue
            if y_idx+shift_row>=num_rows: continue
            if x_idx+shift_col>=num_cols: continue
            image_shift[y_idx+shift_row,x_idx+shift_col] = input_image_2d[y_idx,x_idx]

    return image_shift

def image_rotation(input_image_2d, x_axis, y_axis, angle_rad):

    num_rows, num_cols = input_image_2d.shape
    min_pix_x = x_axis[0]
    max_pix_x = x_axis[len(x_axis)-1]
    min_pix_y = y_axis[0]
    max_pix_y = y_axis[len(y_axis)-1]
    delta_pix_x = (max_pix_x-min_pix_x)/float(num_cols)
    delta_pix_y = (max_pix_y-min_pix_y)/float(num_rows)

    image_rotate = np.zeros_like(input_image_2d)
    rotation_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],[np.sin(angle_rad), np.cos(angle_rad)]])
    center_col = float(num_cols)/2.
    center_row = float(num_rows)/2.
    for x_idx in range(0,num_cols):
        for y_idx in range(0,num_rows):
            delta_col = float(x_idx - center_col)
            delta_row = -float(y_idx - center_row)
            delta_coord = np.array([delta_col,delta_row])
            rot_coord = rotation_matrix @ delta_coord
            rot_row = round(-rot_coord[1]+center_row)
            rot_col = round(rot_coord[0]+center_col)
            if rot_row<0: continue
            if rot_row>=num_rows: continue
            if rot_col<0: continue
            if rot_col>=num_cols: continue
            #image_rotate[rot_row,rot_col] += input_image_2d[y_idx,x_idx]
            image_rotate[rot_row,rot_col] = input_image_2d[y_idx,x_idx]

    return image_rotate



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
        x_axis += [min_pix_x+x_idx*delta_pix_x+0.5*delta_pix_x]
    for y_idx in range(0,num_rows):
        y_axis += [max_pix_y-y_idx*delta_pix_y-0.5*delta_pix_y]

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

def load_training_samples(training_sample_path, is_gamma=True, is_training=False, use_truth=False, do_cleaning=True, do_reposition=False, min_energy=0.1, max_energy=1000., max_evt=1e10):

    fig, ax = plt.subplots()
    figsize_x = 8.6
    figsize_y = 6.4
    fig.set_figheight(figsize_y)
    fig.set_figwidth(figsize_x)

    ana_tag = 'testing_sample'
    if is_training:
        ana_tag = 'training_sample'

    if use_truth:
        ana_tag += '_truth'
    else:
        ana_tag += '_noisy'
    if do_cleaning:
        ana_tag += '_clean'
    else:
        ana_tag += '_dirty'
    if do_reposition:
        ana_tag += '_repose'
    else:
        ana_tag += '_origin'

    
    id_list = []
    truth_shower_position_matrix = []
    hillas_shower_position_matrix = []
    cam_axes = []
    telesc_position_matrix = []
    big_image_matrix = []
    big_time_matrix = []
    big_param_matrix = []
    big_moment_matrix = []

    print (f'loading file: {training_sample_path}')
    source = SimTelEventSource(training_sample_path, focal_length_choice='EQUIVALENT')
    
    # Explore the instrument description
    subarray = source.subarray
    print (subarray.to_table())
    
    ob_keys = source.observation_blocks.keys()
    run_id = list(ob_keys)[0]
    print (f'run_id = {run_id}')
    
    tel_pointing_alt = float(source.observation_blocks[run_id].subarray_pointing_lat/u.rad)
    tel_pointing_az = float(source.observation_blocks[run_id].subarray_pointing_lon/u.rad)
    if tel_pointing_alt>1.*math.pi:
        tel_pointing_alt = tel_pointing_alt-2.*math.pi
    if tel_pointing_az>1.*math.pi:
        tel_pointing_az = tel_pointing_az-2.*math.pi

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
    
    # Apply some calibration and trace integration
    calib = CameraCalibrator(subarray=subarray)
    image_processor = ImageProcessor(subarray=subarray,config=image_processor_config)
    shower_processor = ShowerProcessor(subarray=subarray)

    
    evt_idx = -1
    for event in source:
    
        evt_idx += 1

        if max_evt==evt_idx: continue
        if not is_training:
            if (evt_idx % 5)==0: continue

        event_id = event.index['event_id']
    
        ntel = len(event.r0.tel)
    
        shower_energy = float(event.simulation.shower.energy/u.TeV)
        shower_core_x = float(event.simulation.shower.core_x/u.m)
        shower_core_y = float(event.simulation.shower.core_y/u.m)
        shower_alt = float(event.simulation.shower.alt/u.rad)
        shower_az = float(event.simulation.shower.az/u.rad)
        shower_height = float(event.simulation.shower.h_first_int/u.m)
        shower_x_max = float(event.simulation.shower.x_max/(u.g/(u.cm*u.cm)))
        if shower_alt>1.*math.pi:
            shower_alt = shower_alt-2.*math.pi
        if shower_az>1.*math.pi:
            shower_az = shower_az-2.*math.pi
    
        if shower_energy<min_energy: continue
        if shower_energy>max_energy: continue
    
        calib(event)  # fills in r1, dl0, and dl1
        image_processor(event) # Takes DL1/Image data and cleans and parametrizes the images into DL1/parameters. Should be run after CameraCalibrator.
        shower_processor(event)
        ranked_tel_key_array = rank_brightest_telescope(event.r0.tel)

        for tel_idx in range(0,len(ranked_tel_key_array)):
    
            tel_key = ranked_tel_key_array[tel_idx][0]
            geom = subarray.tel[tel_key].camera.geometry

            dl1tel = event.dl1.tel[tel_key]
            clean_image = dl1tel.image

            image_size = np.sum(clean_image)
            if image_size<=0.: continue

            # image cleaning
            if do_cleaning:

                #analysis_image_2d = geom.image_to_cartesian_representation(dl1tel.image)
                #x_axis, y_axis = get_cam_coord_axes(geom,analysis_image_2d)
                #num_rows, num_cols = analysis_image_2d.shape
                #for x_idx in range(0,num_cols):
                #    for y_idx in range(0,num_rows):
                #        if math.isnan(analysis_image_2d[y_idx,x_idx]): 
                #            analysis_image_2d[y_idx,x_idx] = 0.
                #analysis_image_smooth = smooth_image(analysis_image_2d,x_axis,y_axis,100.)
                #analysis_image_smooth_1d = geom.image_from_cartesian_representation(analysis_image_smooth)
                #image_mask = np.zeros_like(analysis_image_smooth)
                #image_mask = find_mask(analysis_image_smooth)
                #analysis_image_2d = my_clean_image(analysis_image_smooth,image_mask)
                #clean_image = geom.image_from_cartesian_representation(analysis_image_2d)
                #for pix in range(0,len(clean_image)):
                #    if clean_image[pix]==0.:
                #        event.dl1.tel[tel_key].image[pix] = 0.
                #        event.dl1.tel[tel_key].peak_time[pix] = 0.
                #    else:
                #        event.dl1.tel[tel_key].image[pix] = clean_image[pix]
    
                image_mask = tailcuts_clean(geom,dl1tel.image,boundary_thresh=2,picture_thresh=4,min_number_picture_neighbors=2)
                for pix in range(0,len(image_mask)):
                    if not image_mask[pix]:
                        event.dl1.tel[tel_key].image[pix] = 0.
                        event.dl1.tel[tel_key].peak_time[pix] = 0.
                if np.sum(image_mask)<5:
                    for pix in range(0,len(image_mask)):
                        event.dl1.tel[tel_key].image[pix] = 0.
                        event.dl1.tel[tel_key].peak_time[pix] = 0.
                print (f'np.sum(image_mask) = {np.sum(image_mask)}')


        stereo = event.dl2.stereo.geometry["HillasReconstructor"]
        hillas_shower_alt = 0.
        hillas_shower_az = 0.
        hillas_shower_height = 0.
        hillas_shower_core_x = 0.
        hillas_shower_core_y = 0.
        hillas_valid = False
        if stereo.is_valid:
            hillas_shower_alt = float(stereo.alt/u.rad)
            hillas_shower_az = float(stereo.az/u.rad)
            if hillas_shower_az>np.pi:
                hillas_shower_az = hillas_shower_az - 2.*np.pi
            hillas_shower_height = float(stereo.h_max/u.m)
            hillas_shower_core_x = float(stereo.core_x/u.m)
            hillas_shower_core_y = float(stereo.core_y/u.m)
            hillas_valid = True
            print ('Hillas Reconstruction is successful.')
        else:
            hillas_valid = False
            print ('Hillas Reconstruction is invalid.')
    

        mean_tel_x = 0.
        mean_tel_y = 0.
        n_tel = 0.
        for tel_idx in range(0,len(ranked_tel_key_array)):
            tel_key = ranked_tel_key_array[tel_idx][0]
            tel_x = float(subarray.positions[tel_key][0]/u.m)
            tel_y = float(subarray.positions[tel_key][1]/u.m)
            mean_tel_x += tel_x
            mean_tel_y += tel_y
            n_tel += 1.
        mean_tel_x = mean_tel_x/n_tel
        mean_tel_y = mean_tel_y/n_tel

        list_img_a = []
        list_img_b = []
        list_img_w = []
    
        for tel_idx in range(0,len(ranked_tel_key_array)):
    
            tel_key = ranked_tel_key_array[tel_idx][0]
            tel_instrument = subarray.tel[tel_key]
    
            sim_tel = event.simulation.tel[tel_key]
            sim_tel_impact = float(sim_tel.impact['distance']/u.m)
            true_image = sim_tel.true_image
            #true_time = sim_tel.peak_time # simulation camera container does not have peak_time

            dl1tel = event.dl1.tel[tel_key]
            noisy_image = dl1tel.image
            noisy_time = dl1tel.peak_time
    
            # Define a camera geometry
            geom = subarray.tel[tel_key].camera.geometry
            # The CameraGeometry has functions to convert the 1d image arrays to 2d arrays and back to the 1d array
            analysis_image_2d = geom.image_to_cartesian_representation(noisy_image)
            if use_truth:
                analysis_image_2d = geom.image_to_cartesian_representation(true_image)
            analysis_time_2d = geom.image_to_cartesian_representation(noisy_time)

            num_rows, num_cols = analysis_image_2d.shape
            for x_idx in range(0,num_cols):
                for y_idx in range(0,num_rows):
                    if math.isnan(analysis_image_2d[y_idx,x_idx]): 
                        analysis_image_2d[y_idx,x_idx] = 0.
                        analysis_time_2d[y_idx,x_idx] = 0.

            x_axis, y_axis = get_cam_coord_axes(geom,analysis_image_2d)

            for x_idx in range(0,num_cols):
                for y_idx in range(0,num_rows):
                    if analysis_image_2d[y_idx,x_idx]==0.: 
                        analysis_time_2d[y_idx,x_idx] = 0.

            image_size = np.sum(analysis_image_2d)
            if image_size<=0.: continue

            analysis_time_2d = smooth_time_image(analysis_time_2d,analysis_image_2d,x_axis,y_axis,50.)

            analysis_image_1d = geom.image_from_cartesian_representation(analysis_image_2d)
    
            tel_focal_length = float(geom.frame.focal_length/u.m)
            tel_x = float(subarray.positions[tel_key][0]/u.m)
            tel_y = float(subarray.positions[tel_key][1]/u.m)

            tel_info = [tel_pointing_alt,tel_pointing_az,tel_x,tel_y,tel_focal_length]
            array_coord = [shower_alt,shower_az,shower_core_x,shower_core_y]
            tel_coord = convert_array_coord_to_tel_coord(array_coord,tel_info)
            evt_cam_x = tel_coord[0]
            evt_cam_y = tel_coord[1]
            evt_impact_x = tel_coord[2]
            evt_impact_y = tel_coord[3]

            image_center_x, image_center_y, image_foci_x1, image_foci_y1, image_foci_x2, image_foci_y2, center_time, delta_foci_time, semi_major_sq, semi_minor_sq = find_image_moments_guided(analysis_image_2d, analysis_time_2d, x_axis, y_axis, guided=True, arrival_x=evt_cam_x, arrival_y=evt_cam_y)
            image_foci_x = image_foci_x1
            image_foci_y = image_foci_y1

            for x_idx in range(0,num_cols):
                for y_idx in range(0,num_rows):
                    if analysis_time_2d[y_idx,x_idx]==0.: continue
                    analysis_time_2d[y_idx,x_idx] = analysis_time_2d[y_idx,x_idx]-center_time

            analysis_time_1d = geom.image_from_cartesian_representation(analysis_time_2d)

            if do_reposition:
                analysis_image_recenter = np.zeros_like(analysis_image_2d)
                analysis_time_recenter = np.zeros_like(analysis_time_2d)
                shift_x = -image_center_x
                shift_y = -image_center_y
                analysis_image_recenter = image_translation(analysis_image_2d, x_axis, y_axis, shift_x, shift_y)
                analysis_time_recenter = image_translation(analysis_time_2d, x_axis, y_axis, shift_x, shift_y)

                analysis_image_rotate = np.zeros_like(analysis_image_2d)
                analysis_time_rotate = np.zeros_like(analysis_time_2d)
                angle_rad = -np.arctan2(image_foci_y-image_center_y,image_foci_x-image_center_x)
                analysis_image_rotate = image_rotation(analysis_image_recenter, x_axis, y_axis, angle_rad)
                analysis_time_rotate = image_rotation(analysis_time_recenter, x_axis, y_axis, angle_rad)
    
                analysis_image_rotate_1d = geom.image_from_cartesian_representation(analysis_image_rotate)
                analysis_time_rotate_1d = geom.image_from_cartesian_representation(analysis_time_rotate)

            
            #hillas_params = hillas_parameters(geom, analysis_image_1d)
            #hillas_intensity = hillas_params['intensity']
            #hillas_r = hillas_params['r']/u.m
            #hillas_length = hillas_params['length']/u.m
            #hillas_width = hillas_params['width']/u.m
            #hillas_npix = 0
            #for x_idx in range(0,num_cols):
            #    for y_idx in range(0,num_rows):
            #        if analysis_image_2d[y_idx,x_idx]>0.: 
            #            hillas_npix += 1

            #fig.clf()
            #axbig = fig.add_subplot()
            #label_x = 'X'
            #label_y = 'Y'
            #axbig.set_xlabel(label_x)
            #axbig.set_ylabel(label_y)
            #im = axbig.imshow(analysis_image_recenter,origin='lower')
            #cbar = fig.colorbar(im)
            #fig.savefig(f'{ctapipe_output}/output_plots/training_evt{event_id}_tel{tel_idx}_image_recenter.png',bbox_inches='tight')
            #axbig.remove()
    
            #fig.clf()
            #axbig = fig.add_subplot()
            #label_x = 'X'
            #label_y = 'Y'
            #axbig.set_xlabel(label_x)
            #axbig.set_ylabel(label_y)
            #im = axbig.imshow(analysis_time_rotate,origin='lower')
            #cbar = fig.colorbar(im)
            #fig.savefig(f'{ctapipe_output}/output_plots/training_evt{event_id}_tel{tel_idx}_time_rotate.png',bbox_inches='tight')
            #axbig.remove()

            #fig.clf()
            #axbig = fig.add_subplot()
            #label_x = 'X'
            #label_y = 'Y'
            #axbig.set_xlabel(label_x)
            #axbig.set_ylabel(label_y)
            #im = axbig.imshow(analysis_image_rotate,origin='lower')
            #cbar = fig.colorbar(im)
            #fig.savefig(f'{ctapipe_output}/output_plots/training_evt{event_id}_tel{tel_idx}_image_rotate.png',bbox_inches='tight')
            #axbig.remove()

            #fig.clf()
            #axbig = fig.add_subplot()
            #label_x = 'X'
            #label_y = 'Y'
            #axbig.set_xlabel(label_x)
            #axbig.set_ylabel(label_y)
            #im = axbig.imshow(analysis_image_2d,origin='lower')
            #cbar = fig.colorbar(im)
            #fig.savefig(f'{ctapipe_output}/output_plots/training_evt{event_id}_tel{tel_idx}_image_original.png',bbox_inches='tight')
            #axbig.remove()
    
            #if evt_idx==10: exit()
    
            evt_truth_energy = math.log10(shower_energy)
            evt_truth_impact = pow(pow(evt_impact_x,2)+pow(evt_impact_y,2),0.5)/1000.
            evt_truth_height = shower_height
            evt_truth_x_max = shower_x_max

            evt_truth_arrival = pow(pow(evt_cam_x-image_center_x,2)+pow(evt_cam_y-image_center_y,2),0.5)

            id_list += [[run_id,event_id,tel_key,subarray]]
            telesc_position_matrix += [[tel_pointing_alt,tel_pointing_az,tel_x,tel_y,tel_focal_length]]
            if do_reposition:
                big_image_matrix += [analysis_image_rotate_1d]
                big_time_matrix += [analysis_time_rotate_1d]
            else:
                big_image_matrix += [analysis_image_1d]
                big_time_matrix += [analysis_time_1d]
            big_param_matrix += [[evt_truth_arrival,evt_truth_energy,evt_truth_impact]]
            big_moment_matrix += [[image_center_x, image_center_y, image_foci_x1, image_foci_y1, image_foci_x2, image_foci_y2, center_time, delta_foci_time, semi_major_sq, semi_minor_sq,image_size]]
            truth_shower_position_matrix += [[shower_alt,shower_az,shower_core_x,shower_core_y,evt_truth_energy]]
            hillas_shower_position_matrix += [[hillas_shower_alt,hillas_shower_az,hillas_shower_core_x,hillas_shower_core_y,hillas_valid]]
            cam_axes += [[x_axis,y_axis]]

    

    output_filename = f'{ctapipe_output}/output_samples/{ana_tag}_run{run_id}.pkl'
    if not is_gamma:
        output_filename = f'{ctapipe_output}/output_proton_samples/{ana_tag}_run{run_id}.pkl'
    print (f'writing file to {output_filename}')
    with open(output_filename,"wb") as file:
        pickle.dump([id_list, telesc_position_matrix, truth_shower_position_matrix, cam_axes, big_image_matrix, big_time_matrix, big_param_matrix, big_moment_matrix, hillas_shower_position_matrix], file)


class NeuralNetwork:
    # based on https://realpython.com/python-ai-neural-network/
    def __init__(self, input_vector_1d, target_1d, learning_rate, input_n_nodes):
        image_dim = len(input_vector_1d) # n
        param_dim = len(target_1d) # j
        n_nodes = input_n_nodes # i
        self.weight_1 = np.random.randn(n_nodes,image_dim) # ixn
        self.bias_1 = np.random.randn(n_nodes,1) # ix1
        self.weight_2 = np.random.randn(param_dim,n_nodes) # jxi
        self.bias_2 = np.random.randn(param_dim,1) # jx1
        self.learning_rate = learning_rate

    def _sigmoid(self, x):
        #y = np.maximum(-100., x)
        return 1 / (1 + np.exp(-x))

    def _sigmoid_deriv(self, x):
        return self._sigmoid(x) * (1 - self._sigmoid(x))

    def predict(self, input_vector): # forward propagation

        input_vector_col = input_vector[:, np.newaxis]
        
        layer_1 = np.dot(self.weight_1, input_vector_col) + self.bias_1 # ix1
        layer_1 = self._sigmoid(layer_1)
        layer_2 = np.dot(self.weight_2, layer_1) + self.bias_2 # jx1
        prediction = layer_2

        return prediction

    def _compute_gradients(self, input_vector, target):

        input_vector_col = input_vector[:, np.newaxis]
        target_col = target[:, np.newaxis]
        input_vector_T = np.array(input_vector_col).T  # nx1
        target_T = np.array(target_col).T # jx1

        layer_1 = np.dot(self.weight_1, input_vector_col) + self.bias_1 # ix1 
        layer_1 = self._sigmoid(layer_1)
        layer_2 = np.dot(self.weight_2, layer_1) + self.bias_2 # jx1
        prediction = layer_2
     
        dlayer_2 = (prediction - target_col) # jx1
        dweight_2 = dlayer_2.dot(layer_1.T) # jxi
        dbias_2 = dlayer_2 # jx1
        dlayer_1 = self.weight_2.T.dot(dlayer_2) * self._sigmoid_deriv(layer_1) # ixm
        dweight_1 = dlayer_1.dot(input_vector_T) # ixn
        dbias_1 = dlayer_1 # ix1

        return dweight_1, dbias_1, dweight_2, dbias_2

    def _update_parameters(self, dweight_1, dbias_1, dweight_2, dbias_2):
        self.weight_1 = self.weight_1 - self.learning_rate * dweight_1
        self.bias_1 = self.bias_1 - self.learning_rate * dbias_1
        self.weight_2 = self.weight_2 - self.learning_rate * dweight_2
        self.bias_2 = self.bias_2 - self.learning_rate * dbias_2

    def train(self, input_vectors, targets, iterations):

        cumulative_errors = []
        for current_iteration in range(iterations):
            # Pick a sample image at random
            random_data_index = np.random.randint(len(input_vectors))
            input_vector = input_vectors[random_data_index]
            target = targets[random_data_index]

            prediction = self.predict(input_vector)
            dweight_1, dbias_1, dweight_2, dbias_2 = self._compute_gradients(input_vector, target)
            self._update_parameters(dweight_1, dbias_1, dweight_2, dbias_2)

            # Measure the cumulative error for all the instances
            if current_iteration % 100 == 0:
                cumulative_error = 0
                # Loop through all the instances to measure the error
                for data_instance_index in range(len(input_vectors)):
                    data_point = input_vectors[data_instance_index]
                    target = targets[data_instance_index]
                    target_col = target[:, np.newaxis]
                    prediction = self.predict(data_point)
                    error = np.sum(np.square(prediction - target_col))
                    cumulative_error = cumulative_error + error
                cumulative_errors.append(cumulative_error)

        return cumulative_errors

class MyArray3D:

    def __init__(self,x_bins=10,start_x=0.,end_x=10.,y_bins=10,start_y=0.,end_y=10.,z_bins=10,start_z=0.,end_z=10.):
        delta_x = (end_x-start_x)/float(x_bins)
        delta_y = (end_y-start_y)/float(y_bins)
        delta_z = (end_z-start_z)/float(z_bins)
        array_shape = (x_bins,y_bins,z_bins)
        self.xaxis = np.zeros(array_shape[0])
        self.yaxis = np.zeros(array_shape[1])
        self.zaxis = np.zeros(array_shape[2])
        self.waxis = np.zeros(array_shape)
        for idx in range(0,len(self.xaxis)):
            self.xaxis[idx] = start_x + idx*delta_x
        for idx in range(0,len(self.yaxis)):
            self.yaxis[idx] = start_y + idx*delta_y
        for idx in range(0,len(self.zaxis)):
            self.zaxis[idx] = start_z + idx*delta_z

    def reset(self):
        for idx_x in range(0,len(self.xaxis)):
            for idx_y in range(0,len(self.yaxis)):
                for idx_z in range(0,len(self.zaxis)):
                    self.waxis[idx_x,idx_y,idx_z] = 0.

    def add(self, add_array, factor=1.):
        for idx_x in range(0,len(self.xaxis)):
            for idx_y in range(0,len(self.yaxis)):
                for idx_z in range(0,len(self.zaxis)):
                    self.waxis[idx_x,idx_y,idx_z] = self.waxis[idx_x,idx_y,idx_z]+add_array.waxis[idx_x,idx_y,idx_z]*factor

    def get_bin(self, value_x, value_y, value_z):
        key_idx_x = 0
        key_idx_y = 0
        key_idx_z = 0
        for idx_x in range(0,len(self.xaxis)-1):
            if self.xaxis[idx_x]<=value_x and self.xaxis[idx_x+1]>value_x:
                key_idx_x = idx_x
        for idx_y in range(0,len(self.yaxis)-1):
            if self.yaxis[idx_y]<=value_y and self.yaxis[idx_y+1]>value_y:
                key_idx_y = idx_y
        for idx_z in range(0,len(self.zaxis)-1):
            if self.zaxis[idx_z]<=value_z and self.zaxis[idx_z+1]>value_z:
                key_idx_z = idx_z
        if value_x>self.xaxis.max():
            key_idx_x = len(self.xaxis)-1
        if value_y>self.yaxis.max():
            key_idx_y = len(self.yaxis)-1
        if value_z>self.zaxis.max():
            key_idx_z = len(self.zaxis)-1
        return [key_idx_x,key_idx_y,key_idx_z]

    def fill(self, value_x, value_y, value_z, weight=1.):
        key_idx = self.get_bin(value_x, value_y, value_z)
        key_idx_x = key_idx[0]
        key_idx_y = key_idx[1]
        key_idx_z = key_idx[2]
        self.waxis[key_idx_x,key_idx_y,key_idx_z] += 1.*weight
    
    def divide(self, add_array):
        for idx_x in range(0,len(self.xaxis)):
            for idx_y in range(0,len(self.yaxis)):
                for idx_z in range(0,len(self.zaxis)):
                    if add_array.waxis[idx_x,idx_y,idx_z]==0.:
                        self.waxis[idx_x,idx_y,idx_z] = 0.
                    else:
                        self.waxis[idx_x,idx_y,idx_z] = self.waxis[idx_x,idx_y,idx_z]/add_array.waxis[idx_x,idx_y,idx_z]

    def get_bin_center(self, idx_x, idx_y, idx_z):
        return [self.xaxis[idx_x],self.yaxis[idx_y],self.zaxis[idx_z]]

    def get_bin_content(self, value_x, value_y, value_z):
        key_idx = self.get_bin(value_x, value_y, value_z)
        key_idx_x = key_idx[0]
        key_idx_y = key_idx[1]
        key_idx_z = key_idx[2]
        return self.waxis[key_idx_x,key_idx_y,key_idx_z]

class MyArray2D:

    def __init__(self,x_bins=10,start_x=0.,end_x=10.,y_bins=10,start_y=0.,end_y=10.):
        delta_x = (end_x-start_x)/float(x_bins)
        delta_y = (end_y-start_y)/float(y_bins)
        array_shape = (x_bins,y_bins)
        self.xaxis = np.zeros(array_shape[0])
        self.yaxis = np.zeros(array_shape[1])
        self.zaxis = np.zeros(array_shape)
        for idx in range(0,len(self.xaxis)):
            self.xaxis[idx] = start_x + idx*delta_x
        for idx in range(0,len(self.yaxis)):
            self.yaxis[idx] = start_y + idx*delta_y

    def reset(self):
        for idx_x in range(0,len(self.xaxis)):
            for idx_y in range(0,len(self.yaxis)):
                self.zaxis[idx_x,idx_y] = 0.

    def add(self, add_array, factor=1.):
        for idx_x in range(0,len(self.xaxis)):
            for idx_y in range(0,len(self.yaxis)):
                self.zaxis[idx_x,idx_y] = self.zaxis[idx_x,idx_y]+add_array.zaxis[idx_x,idx_y]*factor

    def get_bin(self, value_x, value_y):
        key_idx_x = 0
        key_idx_y = 0
        for idx_x in range(0,len(self.xaxis)-1):
            if self.xaxis[idx_x]<=value_x and self.xaxis[idx_x+1]>value_x:
                key_idx_x = idx_x
        for idx_y in range(0,len(self.yaxis)-1):
            if self.yaxis[idx_y]<=value_y and self.yaxis[idx_y+1]>value_y:
                key_idx_y = idx_y
        if value_x>self.xaxis.max():
            key_idx_x = len(self.xaxis)-1
        if value_y>self.yaxis.max():
            key_idx_y = len(self.yaxis)-1
        return [key_idx_x,key_idx_y]

    def fill(self, value_x, value_y, weight=1.):
        key_idx = self.get_bin(value_x, value_y)
        key_idx_x = key_idx[0]
        key_idx_y = key_idx[1]
        self.zaxis[key_idx_x,key_idx_y] += 1.*weight
    
    def divide(self, add_array):
        for idx_x in range(0,len(self.xaxis)):
            for idx_y in range(0,len(self.yaxis)):
                if add_array.zaxis[idx_x,idx_y]==0.:
                    self.zaxis[idx_x,idx_y] = 0.
                else:
                    self.zaxis[idx_x,idx_y] = self.zaxis[idx_x,idx_y]/add_array.zaxis[idx_x,idx_y]

    def get_bin_center(self, idx_x, idx_y):
        return [self.xaxis[idx_x],self.yaxis[idx_y]]

    def get_bin_content(self, value_x, value_y):
        key_idx = self.get_bin(value_x, value_y)
        key_idx_x = key_idx[0]
        key_idx_y = key_idx[1]
        return self.zaxis[key_idx_x,key_idx_y]

class MyArray1D:

    def __init__(self,x_axis):
        self.xaxis = np.array(x_axis)
        self.yaxis = np.zeros_like(x_axis)

    def reset(self):
        for idx_x in range(0,len(self.xaxis)):
            self.yaxis[idx_x] = 0.

    def add(self, add_array, factor=1.):
        for idx_x in range(0,len(self.xaxis)):
            self.yaxis[idx_x] = self.yaxis[idx_x]+add_array.yaxis[idx_x]*factor

    def get_bin(self, value_x):
        key_idx_x = 0
        for idx_x in range(0,len(self.xaxis)-1):
            if self.xaxis[idx_x]<=value_x and self.xaxis[idx_x+1]>value_x:
                key_idx_x = idx_x
        if value_x>self.xaxis.max():
            key_idx_x = len(self.xaxis)-1
        return key_idx_x

    def fill(self, value_x, weight=1.):
        key_idx_x = self.get_bin(value_x)
        self.yaxis[key_idx_x] += 1.*weight
    
    def divide(self, add_array):
        for idx_x in range(0,len(self.xaxis)):
            if add_array.yaxis[idx_x]==0.:
                self.yaxis[idx_x] = 0.
            else:
                self.yaxis[idx_x] = self.yaxis[idx_x]/add_array.yaxis[idx_x]


def sqaure_difference_between_1d_images(init_params,all_cam_axes,geom,image_1d_data,lookup_table,eigen_vectors):

    fit_log_energy = init_params[0]
    fit_arrival = init_params[1]
    fit_impact = init_params[2]
    if lookup_table[0].get_bin_content(fit_arrival,fit_impact,fit_log_energy)==0.:
        return 1e10

    fit_latent_space = []
    for r in range(0,len(lookup_table)):
        fit_latent_space += [lookup_table[r].get_bin_content(fit_arrival,fit_impact,fit_log_energy)]
    fit_latent_space = np.array(fit_latent_space)

    #image_1d_fit = eigen_vectors.T @ fit_latent_space
    #sum_chi2 = 0.
    #n_rows = len(image_1d_fit)
    #for row in range(0,n_rows):
    #    diff = image_1d_data[row] - image_1d_fit[row]
    #    error_sq = max(1.,image_1d_data[row])
    #    sum_chi2 += diff*diff/error_sq

    data_latent_space = eigen_vectors @ image_1d_data
    sum_chi2 = 0.
    n_rows = len(data_latent_space)
    for row in range(0,n_rows):
        diff = data_latent_space[row] - fit_latent_space[row]
        sum_chi2 += diff*diff

    return sum_chi2

def get_average(xdata_list,ydata_list,x_axis):

    avg = MyArray1D(x_axis)
    norm = MyArray1D(x_axis)

    for entry in range(0,len(xdata_list)):
        xdata = xdata_list[entry]
        ydata = ydata_list[entry]
        avg.fill(xdata,weight=ydata)
        norm.fill(xdata,weight=1.)

    avg.divide(norm)

    return avg

def single_image_reconstruction(input_image,input_time,geom,cam_axes,lookup_table_pkl,eigen_vectors_pkl,lookup_table_time_pkl,eigen_vectors_time_pkl, guided=False, arrival_x=0., arrival_y=0.):

    input_image_2d = geom.image_to_cartesian_representation(input_image)
    input_time_2d = geom.image_to_cartesian_representation(input_time)

    num_rows, num_cols = input_image_2d.shape
    for row in range(0,num_rows):
        for col in range(0,num_cols):
            if math.isnan(input_image_2d[row,col]): 
                input_image_2d[row,col] = 0.
                input_time_2d[row,col] = 0.

    image_center_x, image_center_y, image_foci_x1, image_foci_y1, image_foci_x2, image_foci_y2, center_time, delta_foci_time, semi_major_sq, semi_minor_sq = find_image_moments_guided(input_image_2d, input_time_2d, cam_axes[0], cam_axes[1], guided=guided, arrival_x=arrival_x, arrival_y=arrival_y)
    image_foci_x = image_foci_x1
    image_foci_y = image_foci_y1

    image_recenter = np.zeros_like(input_image_2d)
    time_recenter = np.zeros_like(input_time_2d)
    shift_x = -image_center_x
    shift_y = -image_center_y
    image_recenter = image_translation(input_image_2d, cam_axes[0], cam_axes[1], shift_x, shift_y)
    time_recenter = image_translation(input_time_2d, cam_axes[0], cam_axes[1], shift_x, shift_y)

    image_rotate = np.zeros_like(input_image_2d)
    time_rotate = np.zeros_like(input_time_2d)
    angle_rad = -np.arctan2(image_foci_y-image_center_y,image_foci_x-image_center_x)
    image_rotate = image_rotation(image_recenter, cam_axes[0], cam_axes[1], angle_rad)
    time_rotate = image_rotation(time_recenter, cam_axes[0], cam_axes[1], angle_rad)
    
    image_rotate_1d = geom.image_from_cartesian_representation(image_rotate)
    time_rotate_1d = geom.image_from_cartesian_representation(time_rotate)

    fit_log_energy = 0.
    fit_arrival = 0.2
    fit_impact = 0.2
    init_params = [fit_log_energy,fit_arrival,fit_impact]
    image_weight = 1./np.sum(np.array(input_image)*np.array(input_image))
    time_weight = 1./np.sum(np.array(input_time)*np.array(input_time))
    fit_chi2 = image_weight*sqaure_difference_between_1d_images(init_params,cam_axes,geom,image_rotate_1d,lookup_table_pkl,eigen_vectors_pkl)
    fit_chi2 += time_weight*sqaure_difference_between_1d_images(init_params,cam_axes,geom,time_rotate_1d,lookup_table_time_pkl,eigen_vectors_time_pkl)
    n_bins_arrival = len(lookup_table_pkl[0].xaxis)
    n_bins_impact = len(lookup_table_pkl[0].yaxis)
    n_bins_energy = len(lookup_table_pkl[0].zaxis)
    for idx_x  in range(0,n_bins_arrival):
        for idx_y  in range(0,n_bins_impact):
            for idx_z  in range(0,n_bins_energy):
                try_log_energy = lookup_table_pkl[0].zaxis[idx_z]
                try_arrival = lookup_table_pkl[0].xaxis[idx_x]
                try_impact = lookup_table_pkl[0].xaxis[idx_y]
                init_params = [try_log_energy,try_arrival,try_impact]
                try_chi2 = image_weight*sqaure_difference_between_1d_images(init_params,cam_axes,geom,image_rotate_1d,lookup_table_pkl,eigen_vectors_pkl)
                try_chi2 += time_weight*sqaure_difference_between_1d_images(init_params,cam_axes,geom,time_rotate_1d,lookup_table_time_pkl,eigen_vectors_time_pkl)
                if try_chi2<fit_chi2:
                    fit_chi2 = try_chi2
                    fit_log_energy = try_log_energy
                    fit_arrival = try_arrival
                    fit_impact = try_impact

    return fit_arrival, fit_impact, fit_log_energy

