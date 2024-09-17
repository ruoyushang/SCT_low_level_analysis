
import os
import sys
import math
import pickle
import time

import matplotlib.animation as animation
import numpy as np
from astropy import units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord, angular_separation
from astropy.time import Time
from matplotlib import pyplot as plt

from ctapipe.calib import CameraCalibrator
from ctapipe.coordinates import CameraFrame, NominalFrame
from ctapipe.image import ImageProcessor, tailcuts_clean, hillas_parameters, number_of_islands
from ctapipe.io import SimTelEventSource
from ctapipe.reco import ShowerProcessor
from ctapipe.visualization import CameraDisplay
from ctapipe.visualization import ArrayDisplay

from scipy.optimize import least_squares, minimize

import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

__all__ = [
    "image_translation_and_rotation",
]

font = {
    "family": "serif",
    "color": "black",
    "weight": "normal",
    "size": 10,
    #"rotation": 0.0,
}

# unoptimized cleaning levels
cleaning_level = {
    "DigiCam": (2, 4, 2),
    "ASTRICam": (2, 4, 2),
    "CHEC": (2, 4, 2),
    "LSTCam": (4, 8, 2),
    "FlashCam": (4, 8, 2),
    "NectarCam": (3, 5, 2),
    "SCTCam": (3, 6, 2),
}
cleaning_level_significance = {
    "DigiCam": (4, 8, 2),
    "ASTRICam": (4, 8, 2),
    "CHEC": (4, 8, 2),
    "LSTCam": (4, 8, 2),
    "FlashCam": (4, 8, 2),
    "NectarCam": (4, 8, 2),
    "SCTCam": (6, 10, 2),
}


image_size_cut = 100.0
mask_size_cut = 5.0
frac_leakage_intensity_cut = 0.05

fast_prediction_power = 3

frac_leakage_intensity_cut_analysis = 0.0
image_size_cut_analysis = 200.0
image_size_bins = [1.*image_size_cut_analysis]
image_size_bins += [2.*image_size_cut_analysis]
image_size_bins += [4.*image_size_cut_analysis]
image_size_bins += [8.*image_size_cut_analysis]
image_size_bins += [16.*image_size_cut_analysis]
image_size_bins += [32.*image_size_cut_analysis]
image_size_bins += [64.*image_size_cut_analysis]

n_bins_arrival = 40
arrival_lower = 0.0
arrival_upper = 0.4
n_bins_impact = 40
impact_lower = 0.0
impact_upper = 800.0
n_bins_energy = 15
log_energy_lower = -1.0
log_energy_upper = 2.0

n_samples_per_window = 2
total_samples = 64
select_samples = 16

select_run_id = 0
select_event_id = 0

run_diagnosis = False
plot_image_size_cut_lower = 3000.0
plot_image_size_cut_upper = 1e10
truth_energy_cut_lower = 0.
truth_energy_cut_upper = 1e10
n_tel_min = 1
n_tel_max = 10000

use_template = False

weighting_mode = 0

def image_translation_and_rotation(
    geometry, list_input_image_1d, shift_x, shift_y, angle_rad, reposition=True, return_pix_coord=False,
):
    """
    Function to perform rotation and translation of a list of images.

    Parameters
    ----------
    geometry:
        geometry of camera
    list_input_image_1d: ndarray
        Array of 1-D images
    shift_x: float
        Translation of position in x coordinates
    shift_y: float
        Translation of position in y coordinates
    angle_rad: float
        Rotation angle of pixels in rad

    Returns
    -------
    list_output_image_1d: ndarray
        Array of 1-D images

    """

    if not reposition:
        return list_input_image_1d

    pixel_width = float(geometry.pixel_width[0] / u.m)
    rotation_matrix = np.array(
        [
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)],
        ]
    )

    all_coord_x = []
    all_coord_y = []
    for pix in range(0, len(list_input_image_1d[0])):
        x = float(geometry.pix_x[pix] / u.m)
        y = float(geometry.pix_y[pix] / u.m)
        all_coord_x += [x]
        all_coord_y += [y]

    list_output_image_1d = []
    list_pix_coord_x = []
    list_pix_coord_y = []
    list_pix_intensity = []
    for img in range(0, len(list_input_image_1d)):
        old_coord_x = []
        old_coord_y = []
        old_image = []
        for pix in range(0, len(list_input_image_1d[img])):
            x = all_coord_x[pix]
            y = all_coord_y[pix]
            if list_input_image_1d[img][pix] == 0.0:
                continue
            old_coord_x += [x]
            old_coord_y += [y]
            old_image += [list_input_image_1d[img][pix]]

        trans_x = np.array(old_coord_x) - shift_x
        trans_y = np.array(old_coord_y) - shift_y

        initi_coord = np.array([trans_y,trans_x])
        rotat_coord = rotation_matrix @ initi_coord
        rotat_x = rotat_coord[0]
        rotat_y = -1. * rotat_coord[1]

        list_pix_coord_x += [rotat_x]
        list_pix_coord_y += [rotat_y]
        list_pix_intensity += [old_image]

        smooth_image = True
        output_image_1d = np.zeros_like(list_input_image_1d[img])
        for pix1 in range(0, len(old_coord_x)):
            if old_image[pix1]==0.: continue
            for pix2 in range(0, len(list_input_image_1d[img])):
                x = all_coord_x[pix2]
                y = all_coord_y[pix2]
                if smooth_image:
                    if abs(x - rotat_x[pix1]) > 2.*pixel_width:
                        continue
                    if abs(y - rotat_y[pix1]) > 2.*pixel_width:
                        continue
                    dist_sq = (x - rotat_x[pix1]) * (x - rotat_x[pix1]) + (y - rotat_y[pix1]) * (y - rotat_y[pix1])
                    output_image_1d[pix2] += old_image[pix1]*np.exp(-2.*(dist_sq/pixel_width)*(dist_sq/pixel_width))
                else:
                    if abs(x - rotat_x[pix1]) > 1.*pixel_width:
                        continue
                    if abs(y - rotat_y[pix1]) > 1.*pixel_width:
                        continue
                    output_image_1d[pix2] += old_image[pix1]
        old_image_size = np.sum(np.array(old_image))
        new_image_size = np.sum(output_image_1d)
        if new_image_size>0.:
            output_image_1d = output_image_1d * old_image_size / new_image_size
        list_output_image_1d += [output_image_1d]

    if return_pix_coord:
        return list_pix_coord_x, list_pix_coord_y, list_pix_intensity
    return list_output_image_1d


def remove_nan_pixels(image_2d):
    num_rows, num_cols = image_2d.shape
    for x_idx in range(0, num_cols):
        for y_idx in range(0, num_rows):
            if math.isnan(image_2d[y_idx, x_idx]):
                image_2d[y_idx, x_idx] = 0.0


def reset_time(input_image_1d, input_time_1d):
    center_time = 0.0
    image_size = 0.0
    for pix in range(0, len(input_image_1d)):
        if input_image_1d[pix] == 0.0:
            continue
        image_size += input_image_1d[pix]
        center_time += input_time_1d[pix] * input_image_1d[pix]

    if image_size == 0.0:
        return 0.0

    center_time = center_time / image_size

    for pix in range(0, len(input_image_1d)):
        if input_image_1d[pix] == 0.0:
            continue
        input_time_1d[pix] += -1.0 * center_time

    return center_time


def least_square_fit(power, input_data, target_data, weight):
    # solve x*A = y using SVD
    # y_{0} = ( x_{0,0} x_{0,1} ... 1 )  a_{0}
    # y_{1} = ( x_{1,0} x_{1,1} ... 1 )  a_{1}
    # y_{2} = ( x_{2,0} x_{2,1} ... 1 )  .
    #                                    b

    # for a line equation y = a*x + b
    # solve x*A = y using SVD
    # y_{0} = ( x_{0} 1 )  a
    # y_{1} = ( x_{1} 1 )  b
    # y_{2} = ( x_{2} 1 )

    x = []
    y = []
    w = []
    for evt in range(0, len(input_data)):
        input_x = []
        for p in range(0, power):
            input_x += [pow(input_data[evt], p)]
        x += [input_x]
        y += [target_data[evt]]
        w += [weight[evt]]
    x = np.array(x)
    y = np.array(y)
    w = np.diag(w)

    # Have a look: https://en.wikipedia.org/wiki/Weighted_least_squares
    # Compute the weighted SVD
    U, S, Vt = np.linalg.svd(x.T @ w @ x, full_matrices=False)
    # Calculate the weighted pseudo-inverse
    S_inv = np.diag(1 / S)
    x_pseudo_inv = (Vt.T @ S_inv @ U.T) @ x.T
    # Compute the weighted least-squares solution
    A_svd = x_pseudo_inv @ (w @ y)
    # Compute parameter error
    A_cov = Vt.T @ S_inv @ U.T
    A_err = np.sqrt(np.diag(A_cov))

    # Compute chi
    chi = (x.dot(A_svd) - y) @ np.sqrt(w)
    chi2 = np.sum(np.square(chi))

    return A_svd, A_err, chi2


def fit_image_to_line(geometry, image_input_1d, transpose=False):
    x = []
    y = []
    w = []
    for pix in range(0, len(image_input_1d)):
        if image_input_1d[pix] == 0.0:
            continue
        if not transpose:
            x += [float(geometry.pix_x[pix] / u.m)]
            y += [float(geometry.pix_y[pix] / u.m)]
            w += [image_input_1d[pix]]
        else:
            x += [float(geometry.pix_y[pix] / u.m)]
            y += [float(geometry.pix_x[pix] / u.m)]
            w += [image_input_1d[pix]]
    x = np.array(x)
    y = np.array(y)
    w = np.array(w)

    # if np.sum(w)==0.:
    #    return 0., 0., np.inf

    avg_x = np.sum(w * x) / np.sum(w)
    avg_y = np.sum(w * y) / np.sum(w)

    pix_width = float(geometry.pixel_width[0] / u.m)
    x_rms = 0.0
    y_rms = 0.0
    weight = 0.0
    for pix in range(0, len(image_input_1d)):
        if image_input_1d[pix] == 0.0:
            continue
        if not transpose:
            x_rms += (
                pow(float(geometry.pix_x[pix] / u.m) - avg_x, 2) * image_input_1d[pix]
            )
            y_rms += (
                pow(float(geometry.pix_y[pix] / u.m) - avg_y, 2) * image_input_1d[pix]
            )
            weight += image_input_1d[pix]
        else:
            x_rms += (
                pow(float(geometry.pix_y[pix] / u.m) - avg_y, 2) * image_input_1d[pix]
            )
            y_rms += (
                pow(float(geometry.pix_x[pix] / u.m) - avg_x, 2) * image_input_1d[pix]
            )
            weight += image_input_1d[pix]
    x_rms = pow(x_rms / weight, 0.5)
    y_rms = pow(y_rms / weight, 0.5)
    projection = pow(y_rms / x_rms,2)

    pix_width = float(geometry.pixel_width[0] / u.m)
    xc = []
    yc = []
    wc = []
    for pix in range(0, len(image_input_1d)):
        if image_input_1d[pix] == 0.0:
            continue
        if not transpose:
            xc += [float(geometry.pix_x[pix] / u.m)]
            yc += [float(geometry.pix_y[pix] / u.m)]
        else:
            xc += [float(geometry.pix_y[pix] / u.m)]
            yc += [float(geometry.pix_x[pix] / u.m)]
        wc += [image_input_1d[pix] / (pix_width * pix_width)]
    xc = np.array(xc)
    yc = np.array(yc)
    wc = np.array(wc)

    a_mtx, a_mtx_err, chi2 = least_square_fit(2, xc, yc, wc)
    fit_a = a_mtx[1]
    fit_a_err = 1.0 * a_mtx_err[1]
    fit_b = a_mtx[0]
    fit_b_err = 1.0 * a_mtx_err[0]

    #dist_2_cog = pow(avg_y-(fit_a*avg_x+fit_b),2)/(1.+fit_a*fit_a)
    #chi2 += dist_2_cog * np.sum(image_input_1d) / (pix_width * pix_width)

    if transpose and fit_a!=0.:
        fit_b = -fit_b/fit_a
        fit_b_err = fit_b_err/fit_a
        fit_a = 1.0/fit_a

    #return fit_a, fit_b, fit_a_err, fit_b_err, chi2
    return fit_a, fit_b, fit_a_err, fit_b_err, chi2*chi2 * projection


def find_intersection_multiple_lines(
    list_tel_id,
    list_x,
    list_y,
    list_a,
    list_b,
    list_a_err,
    list_b_err,
    list_intensity,
    list_angle_err,
    list_length,
    list_width,
    list_pixel_width,
    list_frac_leakage_intensity,
    list_sum_noise,
    list_npix,
):
    # y = a*x + b, weight = 1./b_err

    x = np.array(list_x)
    y = np.array(list_y)
    a = np.array(list_a)
    b = np.array(list_b)
    a_err = np.array(list_a_err)
    b_err = np.array(list_b_err)
    intensity = np.array(list_intensity)
    angle_err = np.array(list_angle_err)
    length = np.array(list_length)
    width = np.array(list_width)
    pixel_width = np.array(list_pixel_width)
    frac_leakage_intensity = np.array(list_frac_leakage_intensity)
    sum_noise = np.array(list_sum_noise)
    npix = np.array(list_npix)
    w = intensity*length/width

    max_significance = np.max(intensity/sum_noise)

    pair_weight = []
    pair_x = []
    pair_y = []
    pair_err = []
    for i1 in range(0, len(a) - 1):
        for i2 in range(i1 + 1, len(a)):

            open_angle = abs(np.arctan(a[i1]) - np.arctan(a[i2]))
            #pair_a = np.array([a[i1], a[i2]])
            #pair_b = np.array([b[i1], b[i2]])
            #pair_w = np.array([1.0, 1.0])
            #x_mtx, x_mtx_err, chi2 = least_square_fit(2, pair_a, pair_b, pair_w)
            #pair_fit_x = -x_mtx[1]
            #pair_fit_y = -x_mtx[0]
            pair_fit_x = (b[i2]-b[i1]) / (a[i1]-a[i2])
            pair_fit_y = a[i1]*pair_fit_x + b[i1]

            dist_sq_1 = pow(pair_fit_x - x[i1],2) + pow(pair_fit_y - y[i1],2)
            dist_sq_2 = pow(pair_fit_x - x[i2],2) + pow(pair_fit_y - y[i2],2)
            pair_dist_sq = pow(x[i2] - x[i1],2) + pow(y[i2] - y[i1],2)
            pair_fit_err_sq = dist_sq_1*a_err[i1]*a_err[i1]/np.pi + b_err[i1]*b_err[i1]/np.pi
            pair_fit_err_sq += dist_sq_2*a_err[i2]*a_err[i2]/np.pi + b_err[i2]*b_err[i2]/np.pi
            pair_fit_err = pow(pair_fit_err_sq/pow(np.sin(open_angle),1),0.5)

            separation = pow(pair_dist_sq,0.5)/(width[i1]+width[i2])
            asymmetry = min((length[i1]-width[i1])/(length[i1]+width[i1]),(length[i2]-width[i2])/(length[i2]+width[i2]))
            significance_1 = intensity[i1]/sum_noise[i1]
            significance_1 = pow(max(0.,significance_1*significance_1-np.log(npix[i1])),0.5)
            significance_2 = intensity[i2]/sum_noise[i2]
            significance_2 = pow(max(0.,significance_2*significance_2-np.log(npix[i2])),0.5)
            significance = min(significance_1,significance_2)
            if significance<0.3*max_significance:
                continue
            if significance<30.:
                continue
            if asymmetry<0.3:
                continue
            if separation<5.0:
                continue

            pair_x += [pair_fit_x]
            pair_y += [pair_fit_y]

            #separation = pow(pair_dist_sq,0.5)/(width[i1]+width[i2])
            #leakage = max(frac_leakage_intensity[i1],frac_leakage_intensity[i2])
            #ambiguity = 0.
            ##if separation<5.0:
            ##    ambiguity = 1.0
            #if asymmetry<0.3:
            #    ambiguity = 1.0
            #fov = pow(45.*np.pi/180.,2)
            #ambiguity_err = pow(fov*ambiguity,0.5)
            #pair_fit_err = max(pair_fit_err,ambiguity_err)

            pair_err += [pair_fit_err]

            pair_fit_weight = 1.
            if weighting_mode==0:
                pair_fit_weight = 1.
            if weighting_mode==1:
                pair_fit_weight = pow(intensity[i1] * intensity[i2],0.5)
            if weighting_mode==2:
                pair_fit_weight = pow(intensity[i1] * intensity[i2],1.0)
            pair_weight += [ pair_fit_weight ]

            if run_diagnosis:
                print ("=============================================================================================")
                print (f"tel_id_1 = {list_tel_id[i1]}, tel_id_2 = {list_tel_id[i2]}")
                print (f"pair_fit_x = {pair_fit_x*180./np.pi:0.3f} deg, pair_fit_y = {pair_fit_y*180./np.pi:0.3f} deg")
                print (f"dist_1 = {pow(dist_sq_1,0.5)*180./np.pi:0.3f} deg")
                print (f"dist_2 = {pow(dist_sq_2,0.5)*180./np.pi:0.3f} deg")
                print (f"b_err[i1] = {b_err[i1]*180./np.pi:0.3f} deg")
                print (f"b_err[i2] = {b_err[i2]*180./np.pi:0.3f} deg")
                print (f"pair_fit_err = {pair_fit_err*180./np.pi:0.3f} deg")
                print (f"np.sin(open_angle) = {np.sin(open_angle)}")
                print (f"separation = {separation}")
                print (f"asymmetry = {asymmetry}")
                print (f"length[i1] = {length[i1]*180./np.pi:0.3f} deg")
                print (f"length[i2] = {length[i2]*180./np.pi:0.3f} deg")
                print (f"significance[i1] = {significance_1}")
                print (f"significance[i2] = {significance_2}")
                print (f"intensity[i1] = {intensity[i1]}")
                print (f"intensity[i2] = {intensity[i2]}")
                print (f"frac_leakage_intensity[i1] = {frac_leakage_intensity[i1]}")
                print (f"frac_leakage_intensity[i2] = {frac_leakage_intensity[i2]}")

    pair_x = np.array(pair_x)
    pair_y = np.array(pair_y)
    pair_err = np.array(pair_err)
    pair_weight = np.array(pair_weight)

    if len(pair_weight)<1:
        return 0., 0., 45.*np.pi/180.

    fit_x = 0.0
    fit_y = 0.0
    fit_err = 0.0
    fit_weight = 0.0
    for xing in range(0, len(pair_x)):
        error_sq = pair_err[xing] * pair_err[xing]
        weight = pair_weight[xing] / error_sq
        fit_weight += weight
        fit_x += pair_x[xing] * weight
        fit_y += pair_y[xing] * weight
        fit_err += pair_err[xing] * pair_err[xing] * weight
        #print (f"pair_x[xing] = {pair_x[xing]:0.3f}, pair_y[xing] = {pair_y[xing]:0.3f}, pair_err[xing] = {pair_err[xing]:0.3f}")
    fit_x = fit_x / fit_weight
    fit_y = fit_y / fit_weight
    fit_err = pow(fit_err / fit_weight, 0.5)
    #print (f"fit_x = {fit_x:0.3f}, fit_y = {fit_y:0.3f}")
    #print (f"fit_err = {fit_err*180./np.pi} deg")

    #x_mtx, x_mtx_err, chi2 = least_square_fit(2, a, b, w)
    #fit_x = -x_mtx[1]
    #fit_y = -x_mtx[0]
    #fit_x_err = 1.0 * x_mtx_err[1]
    #fit_y_err = 1.0 * x_mtx_err[0]
    #fit_err = pow(fit_x_err*fit_x_err+fit_y_err*fit_y_err,0.5)

    return fit_x, fit_y, fit_err


def find_image_features(
    tel_id, geometry, input_image_1d, input_time_1d, avg_noise=0., frac_leakage_intensity=0., flip=False, star_cam_xy=None, use_truth=False
):
    image_center_x = 0.0
    image_center_y = 0.0
    mask_center_x = 0.0
    mask_center_y = 0.0
    center_time = 0.0
    image_size = 0.0
    mask_size = 0.0
    image_size = 0.0
    for pix in range(0, len(input_image_1d)):
        if input_image_1d[pix] == 0.0:
            continue
        mask_size += 1.0
        mask_center_x += float(geometry.pix_x[pix] / u.m)
        mask_center_y += float(geometry.pix_y[pix] / u.m)
        image_size += input_image_1d[pix]
        image_center_x += float(geometry.pix_x[pix] / u.m) * input_image_1d[pix]
        image_center_y += float(geometry.pix_y[pix] / u.m) * input_image_1d[pix]
        center_time += input_time_1d[pix] * input_image_1d[pix]

    if mask_size < mask_size_cut:
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    #if image_size < image_size_cut:
    #    return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    mask_center_x = mask_center_x / mask_size
    mask_center_y = mask_center_y / mask_size
    image_center_x = image_center_x / image_size
    image_center_y = image_center_y / image_size
    center_time = center_time / image_size

    cov_xx = 0.0
    cov_xy = 0.0
    cov_yx = 0.0
    cov_yy = 0.0
    for pix in range(0, len(input_image_1d)):
        if input_image_1d[pix] == 0.0:
            continue
        diff_x = float(geometry.pix_x[pix] / u.m) - image_center_x
        diff_y = float(geometry.pix_y[pix] / u.m) - image_center_y
        weight = input_image_1d[pix]
        cov_xx += diff_x * diff_x * weight
        cov_xy += diff_x * diff_y * weight
        cov_yx += diff_y * diff_x * weight
        cov_yy += diff_y * diff_y * weight
    cov_xx = cov_xx / image_size
    cov_xy = cov_xy / image_size
    cov_yx = cov_yx / image_size
    cov_yy = cov_yy / image_size

    covariance_matrix = np.array([[cov_xx, cov_xy], [cov_yx, cov_yy]])
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    semi_major_sq = eigenvalues[0]
    semi_minor_sq = eigenvalues[1]
    if semi_minor_sq > semi_major_sq:
        x = semi_minor_sq
        semi_minor_sq = semi_major_sq
        semi_major_sq = x

    if pow(semi_major_sq/semi_minor_sq,0.5) < 2.0:
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


    truth_a = image_center_y / image_center_x

    index_primary = 0
    if eigenvalues[1]>eigenvalues[0]:
        index_primary = 1
    vx, vy = eigenvectors[index_primary, 0], eigenvectors[index_primary, 1]
    angle = np.pi/2.
    a = 0.
    b = 0.
    if vx!=0.:
        a = -vy / vx
        angle = np.arctan(a)
        b = image_center_y - a * image_center_x

    pix_width = float(geometry.pixel_width[0] / u.m)
    list_pix_coord_x, list_pix_coord_y, list_pix_intensity = image_translation_and_rotation(
        geometry,
        [input_image_1d],
        image_center_x,
        image_center_y,
        (angle + 0.5 * np.pi) * u.rad,
        #(angle) * u.rad,
        reposition=True,
        return_pix_coord=True,
    )
    pix_coord_x = np.array(list_pix_coord_x[0])
    pix_coord_y = np.array(list_pix_coord_y[0])
    pix_weight = np.array(list_pix_intensity[0])/(4.*pix_width*pix_width)
    #new_list_pix_intensity = []
    #boundary_significance, picture_significance, min_neighbors_significance = cleaning_level_significance[geometry.name]
    #for pix in range(0,len(list_pix_intensity[0])):
    #    new_list_pix_intensity += [max(0.,list_pix_intensity[0][pix]-boundary_significance*avg_noise)]
    #pix_weight = np.array(new_list_pix_intensity)/(4.*pix_width*pix_width)

    a_mtx, a_mtx_err, chi2 = least_square_fit(2, pix_coord_x, pix_coord_y, pix_weight)
    a_derot = a_mtx[1]
    a_err = 2.0 * a_mtx_err[1]
    b_derot = a_mtx[0]
    b_err = 2.0 * a_mtx_err[0]

    #leakage_scale = np.exp(1.*frac_leakage_intensity/frac_leakage_intensity_cut)
    leakage_err = frac_leakage_intensity
    a_err = pow(a_err*a_err + leakage_err*leakage_err,0.5)
    angle_err = a_err

    if run_diagnosis:
        print ("=======================================================================================")
        print (f"tel_id = {tel_id}")
        print (f"eigenvalues = {eigenvalues}")
        print (f"avg_noise = {avg_noise}")
        print (f"PCA a = {a:0.3f}, a_derot = {a_derot:0.5f}, b_derot = {b_derot:0.5f}, a_err = {a_err:0.5f}, b_err = {b_err:0.5f}")
        print (f"leakage_err = {leakage_err}")

    rotation_matrix = np.array(
        [[np.cos(-angle), -np.sin(-angle)], [np.sin(-angle), np.cos(-angle)]]
    )
    diff_x = image_center_x - mask_center_x
    diff_y = image_center_y - mask_center_y
    delta_coord = np.array([diff_x, diff_y])
    rot_coord = rotation_matrix @ delta_coord
    image_direction = rot_coord[0]

    image_direction_err = pow(semi_major_sq,0.5) / pow(image_size,0.5)

    if image_direction < 0.0:
        angle = angle + np.pi


    truth_angle = np.arctan2(-image_center_y, -image_center_x)
    if not star_cam_xy == None:
        truth_angle = np.arctan2(
            star_cam_xy[1] - image_center_y, star_cam_xy[0] - image_center_x
        )

    truth_projection = np.cos(truth_angle - angle)

    if not star_cam_xy == None:
        if truth_projection < 0.0:
            angle = angle + np.pi
        if use_truth:
            angle = truth_angle

    if flip:
        angle = angle + np.pi

    return [
        mask_size,
        image_center_x,
        image_center_y,
        angle,
        pow(semi_major_sq, 0.5),
        pow(semi_minor_sq, 0.5),
        image_direction_err,
        image_direction,
        a,
        b,
        truth_projection,
        1.*a_err,
        1.*b_err,
        1.*angle_err,
        image_size,
    ]


def image_cutout(geometry, image_input_1d, pixs_to_keep=[]):
    eco_image_1d = []
    for pix in pixs_to_keep:
        eco_image_1d += [image_input_1d[pix]]
    return eco_image_1d


def image_cutout_restore(geometry, eco_image_1d, origin_image_1d):
    eco_pix = 0
    for pix in range(0, len(origin_image_1d)):
        x = float(geometry.pix_x[pix] / u.m)
        y = float(geometry.pix_y[pix] / u.m)
        if abs(y) < 0.05:
            origin_image_1d[pix] = eco_image_1d[eco_pix]
            eco_pix += 1
        else:
            origin_image_1d[pix] = 0.0


def find_image_truth(source, subarray, run_id, tel_id, event):
    truth_energy = event.simulation.shower.energy
    truth_core_x = event.simulation.shower.core_x
    truth_core_y = event.simulation.shower.core_y
    truth_alt = event.simulation.shower.alt
    truth_az = event.simulation.shower.az
    truth_height = event.simulation.shower.h_first_int
    truth_x_max = event.simulation.shower.x_max

    geometry = subarray.tel[tel_id].camera.geometry

    obstime = Time("2013-11-01T03:00")
    location = EarthLocation.of_site("Roque de los Muchachos")
    altaz = AltAz(location=location, obstime=obstime)

    tel_pointing_alt = source.observation_blocks[run_id].subarray_pointing_lat
    tel_pointing_az = source.observation_blocks[run_id].subarray_pointing_lon
    tel_pointing = SkyCoord(
        alt=tel_pointing_alt,
        az=tel_pointing_az,
        frame=altaz,
    )
    alt_offset = truth_alt - tel_pointing_alt
    az_offset = truth_az - tel_pointing_az
    star_altaz = SkyCoord(
        alt=tel_pointing_alt + alt_offset,
        az=tel_pointing_az + az_offset,
        frame=altaz,
    )

    focal_length = source.subarray.tel[tel_id].optics.equivalent_focal_length
    tel_x = subarray.positions[tel_id][0]
    tel_y = subarray.positions[tel_id][1]
    impact_x = float((truth_core_x - tel_x) / u.m)
    impact_y = float((truth_core_y - tel_y) / u.m)

    camera_frame = CameraFrame(
        telescope_pointing=tel_pointing,
        focal_length=focal_length,
    )

    star_cam = star_altaz.transform_to(camera_frame)
    star_cam_x = star_cam.x.to_value(u.m)
    star_cam_y = star_cam.y.to_value(u.m)

    truth_info_array = [
        truth_energy,
        truth_core_x,
        truth_core_y,
        truth_alt,
        truth_az,
        truth_height,
        truth_x_max,
        star_cam_x,
        star_cam_y,
        impact_x,
        impact_y,
        focal_length,
    ]

    return truth_info_array

def keep_main_island(geometry,image_mask):

    n_islands = number_of_islands(geometry,image_mask)
    for pix in range(0,len(n_islands[1])):
        if n_islands[1][pix]==1:
            image_mask[pix] = True
        else:
            image_mask[pix] = False

def dynamic_cleaning(geometry,event,tel_id):

    boundary, picture, min_neighbors = cleaning_level[geometry.name]
    image_mask = tailcuts_clean(
        geometry,
        event.dl1.tel[tel_id].image,
        boundary_thresh=boundary,
        picture_thresh=picture,
        min_number_picture_neighbors=min_neighbors,
    )
    keep_main_island(geometry,image_mask)

    night_sky_pixels = 0.
    night_sky_mean = 0.
    night_sky_rms = 0.
    for pix in range(0, len(image_mask)):
        if not image_mask[pix]:
            night_sky_pixels += 1.
            night_sky_mean += event.dl1.tel[tel_id].image[pix]
    if night_sky_pixels>0.:
        night_sky_mean = night_sky_mean/night_sky_pixels
        for pix in range(0, len(image_mask)):
            if not image_mask[pix]:
                night_sky_rms += pow(event.dl1.tel[tel_id].image[pix]-night_sky_mean,2)
        night_sky_rms = pow(night_sky_rms/night_sky_pixels,0.5)

    if night_sky_rms==0.:
        return image_mask

    for pix in range(0, len(image_mask)):
        image_mask[pix] = False

    boundary_significance, picture_significance, min_neighbors_significance = cleaning_level_significance[geometry.name]
    boundary = boundary_significance*night_sky_rms
    picture = picture_significance*night_sky_rms

    image_mask = tailcuts_clean(
        geometry,
        event.dl1.tel[tel_id].image,
        boundary_thresh=boundary,
        picture_thresh=picture,
        min_number_picture_neighbors=min_neighbors_significance,
    )
    keep_main_island(geometry,image_mask)

    sum_noise = 0.
    for pix in range(0, len(image_mask)):
        if image_mask[pix]:
            sum_noise += night_sky_rms*night_sky_rms
    sum_noise = pow(sum_noise,0.5)
    avg_noise = night_sky_rms

    return image_mask, sum_noise, avg_noise


def make_standard_movie(
    ctapipe_output,
    telescope_type,
    subarray,
    run_id,
    tel_id,
    event,
    flip=False,
    star_cam_xy=None,
    reposition=True,
    use_truth=False,
):
    event_id = event.index["event_id"]
    geometry = subarray.tel[tel_id].camera.geometry

    clean_image_1d = np.zeros_like(event.dl1.tel[tel_id].image)
    clean_time_1d = np.zeros_like(event.dl1.tel[tel_id].peak_time)

    image_mask, sum_noise, avg_noise = dynamic_cleaning(geometry,event,tel_id)

    for pix in range(0, len(image_mask)):
        if not image_mask[pix]:
            clean_image_1d[pix] = 0.0
            clean_time_1d[pix] = 0.0
        else:
            clean_image_1d[pix] = event.dl1.tel[tel_id].image[pix]
            clean_time_1d[pix] = event.dl1.tel[tel_id].peak_time[pix]

    center_time = reset_time(clean_image_1d, clean_time_1d)

    waveform = event.dl0.tel[tel_id].waveform
    # print(f"waveform.shape = {waveform.shape}")
    n_pix = waveform.shape[1]
    n_samp = waveform.shape[2]

    n_windows = int(total_samples / n_samples_per_window)
    clean_movie_1d = []
    for win in range(0, n_windows):
        clean_movie_1d += [np.zeros_like(clean_image_1d)]
    for pix in range(0, n_pix):
        if not image_mask[pix]:
            continue  # select signal
        for win in range(0, n_windows):
            for sample in range(0, n_samples_per_window):
                sample_idx = int(sample + win * n_samples_per_window)
                if sample_idx < 0:
                    continue
                if sample_idx >= n_samp:
                    continue
                clean_movie_1d[win][pix] += waveform[0, pix, sample_idx]

    is_edge_image = False
    # image_leakage = leakage_parameters(geometry,clean_image_1d,image_mask) # this function has memory problem
    # if image_leakage.pixels_width_1>0.:
    #    is_edge_image = True
    #    for pix in range(0,len(movie_mask)):
    #        clean_movie_1d[win][pix] = 0.
    border_pixels = geometry.get_border_pixel_mask(1)
    border_mask = border_pixels & image_mask
    leakage_intensity = np.sum(clean_image_1d[border_mask])
    n_pe_cleaning = np.sum(clean_image_1d)
    frac_leakage_intensity = 1.
    if n_pe_cleaning>0.:
        frac_leakage_intensity = leakage_intensity / n_pe_cleaning
        if frac_leakage_intensity > frac_leakage_intensity_cut:
           is_edge_image = True

    #for pix in range(0, len(image_mask)):
    #    clean_image_1d[pix] = 0.0
    #for win in range(0, n_windows):
    #    image_sum = np.sum(clean_movie_1d[win])
    #    if image_sum == 0.0:
    #        continue
    #    movie_mask = image_mask
    #    mask_sum = np.sum(movie_mask)
    #    if mask_sum == 0.0:
    #        continue
    #    border_pixels = geometry.get_border_pixel_mask(1)
    #    border_mask = border_pixels & movie_mask
    #    leakage_intensity = np.sum(clean_movie_1d[win][border_mask])
    #    n_pe_cleaning = np.sum(clean_movie_1d[win])
    #    if n_pe_cleaning>0.:
    #        frac_leakage_intensity = leakage_intensity / n_pe_cleaning
    #        if frac_leakage_intensity < frac_leakage_intensity_cut:
    #            for pix in range(0, len(image_mask)):
    #                clean_image_1d[pix] += clean_movie_1d[win][pix]

    # clean_image_2d = geometry.image_to_cartesian_representation(clean_image_1d)
    # remove_nan_pixels(clean_image_2d)
    # clean_time_2d = geometry.image_to_cartesian_representation(clean_time_1d)
    # remove_nan_pixels(clean_time_2d)

    # image_max = np.max(clean_image_2d[:, :])

    pixel_width = float(geometry.pixel_width[0] / u.m)

    tic_feature = time.perf_counter()
    image_feature_array = find_image_features(
        tel_id, 
        geometry, 
        clean_image_1d, 
        clean_time_1d, 
        avg_noise=avg_noise, 
        frac_leakage_intensity=frac_leakage_intensity, 
        flip=flip, 
        star_cam_xy=star_cam_xy, 
        use_truth=use_truth,
    )
    image_feature_array += [frac_leakage_intensity]
    image_feature_array += [sum_noise]
    image_feature_array += [n_pix]
    toc_feature = time.perf_counter()
    #print(f"find image feature time: {toc_feature-tic_feature:0.1f} sec")

    mask_size = image_feature_array[0]
    image_center_x = image_feature_array[1]
    image_center_y = image_feature_array[2]
    angle = image_feature_array[3]
    semi_major = image_feature_array[4]
    semi_minor = image_feature_array[5]
    image_direction_err = image_feature_array[6]
    image_direction = image_feature_array[7]
    line_a = image_feature_array[8]
    line_b = image_feature_array[9]
    truth_projection = image_feature_array[10]
    line_a_err = image_feature_array[11]
    line_b_err = image_feature_array[12]
    angle_err = image_feature_array[13]
    image_size = image_feature_array[14]

    if mask_size < mask_size_cut:
        return is_edge_image, image_feature_array, [], [], []
    #if image_size < image_size_cut:
    #    return is_edge_image, image_feature_array, [], [], []

    center_time_window = 0.0
    total_weight = 0.0
    for win in range(0, n_windows):
        total_weight += np.sum(clean_movie_1d[win][:])
        center_time_window += float(win) * np.sum(clean_movie_1d[win][:])
    if total_weight == 0.0:
        center_time_window = 0
    else:
        center_time_window = round(center_time_window / total_weight)
    # print(f"center_time_window = {center_time_window}")

    #print (f"len(clean_movie_1d) = {len(clean_movie_1d)}")
    n_windows_slim = int(select_samples / n_samples_per_window)
    slim_movie_1d = []
    for win in range(0, n_windows_slim):
        slim_movie_1d += [np.zeros_like(clean_image_1d)]

    for pix in range(0, n_pix):
        for win in range(0, n_windows_slim):
            old_win = int(center_time_window - n_windows_slim / 2 + win)
            if old_win < 0:
                continue
            if old_win >= len(clean_movie_1d):
                continue
            slim_movie_1d[win][pix] = clean_movie_1d[old_win][pix]

    # image_max = np.max(slim_movie_1d[:][:])

    tic_trans_rotate = time.perf_counter()
    list_rotat_movie_1d = image_translation_and_rotation(
        geometry,
        slim_movie_1d,
        image_center_x,
        image_center_y,
        (angle + 0.5 * np.pi) * u.rad,
        reposition=reposition,
    )
    toc_trans_rotate = time.perf_counter()
    #print(f"movie translation and rotation time: {toc_trans_rotate-tic_trans_rotate:0.1f} sec")

    whole_movie_1d = []
    pixs_to_keep = []
    for pix in range(0, len(clean_image_1d)):
        x = float(geometry.pix_x[pix] / u.m)
        y = float(geometry.pix_y[pix] / u.m)
        if abs(y) < 0.05:
            pixs_to_keep += [pix]
    for win in range(0, n_windows_slim):
        rotate_movie_1d = list_rotat_movie_1d[win]
        eco_movie_1d = image_cutout(
            geometry, rotate_movie_1d, pixs_to_keep=pixs_to_keep
        )
        whole_movie_1d.extend(eco_movie_1d)

    list_rotat_image_1d = image_translation_and_rotation(
        geometry,
        [clean_image_1d, clean_time_1d],
        image_center_x,
        image_center_y,
        (angle + 0.5 * np.pi) * u.rad,
        reposition=reposition,
    )
    rotate_image_1d = list_rotat_image_1d[0]
    rotate_time_1d = list_rotat_image_1d[1]

    pixs_to_keep = []
    for pix in range(0, len(clean_image_1d)):
        x = float(geometry.pix_x[pix] / u.m)
        y = float(geometry.pix_y[pix] / u.m)
        if abs(y) < 0.05:
            pixs_to_keep += [pix]
    eco_image_1d = image_cutout(geometry, rotate_image_1d, pixs_to_keep=pixs_to_keep)
    eco_time_1d = image_cutout(geometry, rotate_time_1d, pixs_to_keep=pixs_to_keep)


    return is_edge_image, image_feature_array, whole_movie_1d, eco_image_1d, eco_time_1d


def analyze_a_training_image(
    ctapipe_output, telescope_type, event, source, run_id, tel_id
):
    truth_info_array = find_image_truth(source, source.subarray, run_id, tel_id, event)
    star_cam_x = truth_info_array[7]
    star_cam_y = truth_info_array[8]
    star_cam_xy = [star_cam_x, star_cam_y]

    (
        is_edge_image,
        image_feature_array,
        eco_movie_1d,
        eco_image_1d,
        eco_time_1d,
    ) = make_standard_movie(
        ctapipe_output,
        telescope_type,
        source.subarray,
        run_id,
        tel_id,
        event,
        flip=False,
        star_cam_xy=star_cam_xy,
        use_truth=True,
    )

    mask_size = image_feature_array[0]
    image_size = image_feature_array[14]

    if image_size < image_size_cut:
        print("failed image_size_cut.")
        return None
    if mask_size < mask_size_cut:
        print("failed mask_size_cut.")
        return None
    if is_edge_image:
        print("failed: edge image.")
        return None

    make_plot = False
    if make_plot:
        event_id = event.index["event_id"]
        geometry = source.subarray.tel[tel_id].camera.geometry
        full_image_1d = np.zeros_like(event.dl1.tel[tel_id].image)
        image_cutout_restore(geometry, eco_image_1d, full_image_1d)

        fig, ax = plt.subplots()
        figsize_x = 8.6
        figsize_y = 6.4
        fig.set_figheight(figsize_y)
        fig.set_figwidth(figsize_x)
        display = CameraDisplay(geometry, ax=ax)
        display.image = full_image_1d
        display.cmap = "Reds"
        display.add_colorbar(ax=ax)
        fig.savefig(
            f"{ctapipe_output}/output_plots/{telescope_type}_run{run_id}_evt{event_id}_tel{tel_id}_train_image.png",
            bbox_inches="tight",
        )
        del fig
        del ax
        plt.close()


    return [
        eco_movie_1d,
        eco_image_1d,
        eco_time_1d,
        image_feature_array,
        truth_info_array,
    ]


def analyze_a_training_event(
    ctapipe_output,
    telescope_type,
    event,
    source,
    run_id,
    movie_matrix,
    image_matrix,
    time_matrix,
    feature_matrix,
    truth_matrix,
):
    event_id = event.index["event_id"]

    ntel = len(event.r0.tel)

    for tel_idx in range(0, len(list(event.dl0.tel.keys()))):
        tel_id = list(event.dl0.tel.keys())[tel_idx]
        # if event_id!=27002: continue
        # if tel_id!=31: continue

        if str(telescope_type) != str(source.subarray.tel[tel_id]):
            continue

        print(
            "===================================================================================="
        )
        print(f"Select telescope type: {telescope_type}")
        print(f"event_id = {event_id}, tel_id = {tel_id}")
        print("TEL{:03}: {}".format(tel_id, source.subarray.tel[tel_id]))

        analysis_results = analyze_a_training_image(
            ctapipe_output, telescope_type, event, source, run_id, tel_id
        )

        if analysis_results == None:
            continue

        eco_movie_1d = analysis_results[0]
        eco_image_1d = analysis_results[1]
        eco_time_1d = analysis_results[2]
        image_feature_array = analysis_results[3]
        truth_info_array = analysis_results[4]

        movie_matrix += [eco_movie_1d]
        image_matrix += [eco_image_1d]
        time_matrix += [eco_time_1d]
        feature_matrix += [image_feature_array]
        truth_matrix += [truth_info_array]


def run_save_training_matrix(training_sample_path, telescope_type, ctapipe_output):
    big_movie_matrix = []
    big_image_matrix = []
    big_time_matrix = []
    big_feature_matrix = []
    big_truth_matrix = []

    print(f"loading file: {training_sample_path}")
    source = SimTelEventSource(training_sample_path, focal_length_choice="EQUIVALENT")

    # Explore the instrument description
    subarray = source.subarray
    print("Array info:")
    print(subarray.info())
    print(subarray.to_table())

    ob_keys = source.observation_blocks.keys()
    run_id = list(ob_keys)[0]

    tel_pointing_alt = float(
        source.observation_blocks[run_id].subarray_pointing_lat / u.rad
    )
    tel_pointing_az = float(
        source.observation_blocks[run_id].subarray_pointing_lon / u.rad
    )
    print(f"tel_pointing_alt = {tel_pointing_alt}")
    print(f"tel_pointing_az = {tel_pointing_az}")

    calib = CameraCalibrator(subarray=subarray)
    image_processor = ImageProcessor(subarray=subarray)

    for event in source:
        calib(event)  # fills in r1, dl0, and dl1
        image_processor(event)
        analyze_a_training_event(
            ctapipe_output,
            telescope_type,
            event,
            source,
            run_id,
            big_movie_matrix,
            big_image_matrix,
            big_time_matrix,
            big_feature_matrix,
            big_truth_matrix,
        )

    ana_tag = "training_sample"
    output_filename = (
        f"{ctapipe_output}/output_samples/{ana_tag}_run{run_id}_{telescope_type}.pkl"
    )
    print(f"writing file to {output_filename}")
    with open(output_filename, "wb") as file:
        pickle.dump(
            [
                big_truth_matrix,
                big_feature_matrix,
                big_image_matrix,
                big_time_matrix,
                big_movie_matrix,
            ],
            file,
        )

    print(f"total images saved = {len(big_image_matrix)}")

    return


class veritas_histogram_3D:
    def __init__(
        self,
        x_bins=10,
        start_x=0.0,
        end_x=10.0,
        y_bins=10,
        start_y=0.0,
        end_y=10.0,
        z_bins=10,
        start_z=0.0,
        end_z=10.0,
        overflow=True,
    ):
        array_shape = (x_bins, y_bins, z_bins)
        self.delta_x = (end_x - start_x) / float(x_bins)
        self.delta_y = (end_y - start_y) / float(y_bins)
        self.delta_z = (end_z - start_z) / float(z_bins)
        self.xaxis = np.zeros(array_shape[0] + 1)
        self.yaxis = np.zeros(array_shape[1] + 1)
        self.zaxis = np.zeros(array_shape[2] + 1)
        self.waxis = np.zeros(array_shape)
        self.overflow = overflow
        for idx in range(0, len(self.xaxis)):
            self.xaxis[idx] = start_x + idx * self.delta_x
        for idx in range(0, len(self.yaxis)):
            self.yaxis[idx] = start_y + idx * self.delta_y
        for idx in range(0, len(self.zaxis)):
            self.zaxis[idx] = start_z + idx * self.delta_z

    def reset(self):
        for idx_x in range(0, len(self.xaxis) - 1):
            for idx_y in range(0, len(self.yaxis) - 1):
                for idx_z in range(0, len(self.zaxis) - 1):
                    self.waxis[idx_x, idx_y, idx_z] = 0.0

    def add(self, add_array, factor=1.0):
        for idx_x in range(0, len(self.xaxis) - 1):
            for idx_y in range(0, len(self.yaxis) - 1):
                for idx_z in range(0, len(self.zaxis) - 1):
                    self.waxis[idx_x, idx_y, idx_z] = (
                        self.waxis[idx_x, idx_y, idx_z]
                        + add_array.waxis[idx_x, idx_y, idx_z] * factor
                    )

    def get_bin(self, value_x, value_y, value_z):
        key_idx_x = -1
        key_idx_y = -1
        key_idx_z = -1
        for idx_x in range(0, len(self.xaxis) - 1):
            if self.xaxis[idx_x] <= value_x and self.xaxis[idx_x + 1] > value_x:
                key_idx_x = idx_x
        for idx_y in range(0, len(self.yaxis) - 1):
            if self.yaxis[idx_y] <= value_y and self.yaxis[idx_y + 1] > value_y:
                key_idx_y = idx_y
        for idx_z in range(0, len(self.zaxis) - 1):
            if self.zaxis[idx_z] <= value_z and self.zaxis[idx_z + 1] > value_z:
                key_idx_z = idx_z
        if value_x > self.xaxis.max():
            key_idx_x = len(self.xaxis) - 2
        if value_y > self.yaxis.max():
            key_idx_y = len(self.yaxis) - 2
        if value_z > self.zaxis.max():
            key_idx_z = len(self.zaxis) - 2
        return [key_idx_x, key_idx_y, key_idx_z]

    def get_heaviest_axis(self):
        max_weight = 0.0
        key_idx_x = -1
        key_idx_y = -1
        key_idx_z = -1
        for idx_x in range(0, len(self.xaxis) - 1):
            for idx_y in range(0, len(self.yaxis) - 1):
                for idx_z in range(0, len(self.zaxis) - 1):
                    local_weight = abs(self.waxis[idx_x, idx_y, idx_z])
                    if max_weight < local_weight:
                        max_weight = local_weight
                        key_idx_x = idx_x
                        key_idx_y = idx_y
                        key_idx_z = idx_z
        return [self.xaxis[key_idx_x], self.yaxis[key_idx_y], self.zaxis[key_idx_z]]

    def fill(self, value_x, value_y, value_z, weight=1.0):
        key_idx = self.get_bin(value_x, value_y, value_z)
        key_idx_x = key_idx[0]
        key_idx_y = key_idx[1]
        key_idx_z = key_idx[2]
        if key_idx_x == -1:
            key_idx_x = 0
            if not self.overflow:
                weight = 0.0
        if key_idx_y == -1:
            key_idx_y = 0
            if not self.overflow:
                weight = 0.0
        if key_idx_z == -1:
            key_idx_z = 0
            if not self.overflow:
                weight = 0.0
        if key_idx_x == len(self.xaxis):
            key_idx_x = len(self.xaxis) - 2
            if not self.overflow:
                weight = 0.0
        if key_idx_y == len(self.yaxis):
            key_idx_y = len(self.yaxis) - 2
            if not self.overflow:
                weight = 0.0
        if key_idx_z == len(self.zaxis):
            key_idx_z = len(self.zaxis) - 2
            if not self.overflow:
                weight = 0.0
        self.waxis[key_idx_x, key_idx_y, key_idx_z] += 1.0 * weight

    def divide(self, add_array):
        for idx_x in range(0, len(self.xaxis) - 1):
            for idx_y in range(0, len(self.yaxis) - 1):
                for idx_z in range(0, len(self.zaxis) - 1):
                    if add_array.waxis[idx_x, idx_y, idx_z] == 0.0:
                        self.waxis[idx_x, idx_y, idx_z] = 0.0
                    else:
                        self.waxis[idx_x, idx_y, idx_z] = (
                            self.waxis[idx_x, idx_y, idx_z]
                            / add_array.waxis[idx_x, idx_y, idx_z]
                        )

    def get_bin_center(self, idx_x, idx_y, idx_z):
        return [
            self.xaxis[idx_x] + 0.5 * self.delta_x,
            self.yaxis[idx_y] + 0.5 * self.delta_y,
            self.zaxis[idx_z] + 0.5 * self.delta_z,
        ]

    def get_bin_content(self, value_x, value_y, value_z):
        key_idx = self.get_bin(value_x, value_y, value_z)
        key_idx_x = key_idx[0]
        key_idx_y = key_idx[1]
        key_idx_z = key_idx[2]
        if key_idx_x == -1:
            key_idx_x = 0
        if key_idx_y == -1:
            key_idx_y = 0
        if key_idx_z == -1:
            key_idx_z = 0
        if key_idx_x == len(self.xaxis):
            key_idx_x = len(self.xaxis) - 2
        if key_idx_y == len(self.yaxis):
            key_idx_y = len(self.yaxis) - 2
        if key_idx_z == len(self.zaxis):
            key_idx_z = len(self.zaxis) - 2
        return self.waxis[key_idx_x, key_idx_y, key_idx_z]

    def get_bin_content_by_index(self, key_idx_x, key_idx_y, key_idx_z):
        if key_idx_x == -1:
            key_idx_x = 0
        if key_idx_y == -1:
            key_idx_y = 0
        if key_idx_z == -1:
            key_idx_z = 0
        if key_idx_x == len(self.xaxis):
            key_idx_x = len(self.xaxis) - 2
        if key_idx_y == len(self.yaxis):
            key_idx_y = len(self.yaxis) - 2
        if key_idx_z == len(self.zaxis):
            key_idx_z = len(self.zaxis) - 2
        return self.waxis[key_idx_x, key_idx_y, key_idx_z]


def MakeLookupTable(
    ctapipe_output,
    telescope_type,
    eigenvectors,
    big_matrix,
    feature_matrix,
    truth_matrix,
    image_rank,
    pkl_name,
    nvar=3,
):
    lookup_table = []
    for idx in range(0,len(image_size_bins)-1):
        lookup_table += [[]]

    for idx in range(0,len(image_size_bins)-1):
        lookup_table_norm = veritas_histogram_3D(
            x_bins=n_bins_arrival,
            start_x=arrival_lower,
            end_x=arrival_upper,
            y_bins=n_bins_impact,
            start_y=impact_lower,
            end_y=impact_upper,
            z_bins=n_bins_energy,
            start_z=log_energy_lower,
            end_z=log_energy_upper,
        )

        list_impact = []
        list_arrival = []
        list_log_energy = []
        list_height = []
        list_xmax = []
        list_image_qual = []

        for r in range(0, image_rank):
            lookup_table[idx] += [
                veritas_histogram_3D(
                    x_bins=n_bins_arrival,
                    start_x=arrival_lower,
                    end_x=arrival_upper,
                    y_bins=n_bins_impact,
                    start_y=impact_lower,
                    end_y=impact_upper,
                    z_bins=n_bins_energy,
                    start_z=log_energy_lower,
                    end_z=log_energy_upper,
                )
            ]

        for img in range(0, len(big_matrix[idx])):
            image_center_x = feature_matrix[idx][img][1]
            image_center_y = feature_matrix[idx][img][2]
            image_direction_err = feature_matrix[idx][img][6]
            image_direction = feature_matrix[idx][img][7]
            image_angle_err = feature_matrix[idx][img][13]

            if image_angle_err==0.: continue

            truth_energy = float(truth_matrix[idx][img][0] / u.TeV)
            truth_height = float(truth_matrix[idx][img][5] / u.m)
            truth_x_max = float(truth_matrix[idx][img][6] / (u.g / (u.cm * u.cm)))
            star_cam_x = truth_matrix[idx][img][7]
            star_cam_y = truth_matrix[idx][img][8]
            impact_x = truth_matrix[idx][img][9]
            impact_y = truth_matrix[idx][img][10]

            arrival = pow(
                pow(star_cam_x - image_center_x, 2) + pow(star_cam_y - image_center_y, 2),
                0.5,
            )
            impact = pow(impact_x * impact_x + impact_y * impact_y, 0.5)
            log_energy = np.log10(truth_energy)

            list_log_energy += [log_energy]
            list_height += [truth_height]
            list_xmax += [truth_x_max]
            list_arrival += [arrival]
            list_impact += [impact]

            image_1d = np.array(big_matrix[idx][img])
            image_latent_space = eigenvectors[idx] @ image_1d
            for r in range(0, image_rank):
                lookup_table[idx][r].fill(
                    arrival,
                    impact,
                    log_energy,
                    weight=image_latent_space[r] * 0.0001 / image_angle_err,
                    #weight=image_latent_space[r] * 1.0,
                )

            lookup_table_norm.fill(
                arrival, impact, log_energy, weight=0.0001 / image_angle_err
                #arrival, impact, log_energy, weight=1.0
            )

        for r in range(0, image_rank):
            lookup_table[idx][r].divide(lookup_table_norm)

        n_empty_cells = 0.0
        n_filled_cells = 0.0
        n_training_images = float(len(list_log_energy))
        for idx_x in range(0, len(lookup_table_norm.xaxis) - 1):
            for idx_y in range(0, len(lookup_table_norm.yaxis) - 1):
                for idx_z in range(0, len(lookup_table_norm.zaxis) - 1):
                    count = lookup_table_norm.waxis[idx_x, idx_y, idx_z]
                    if count == 0:
                        n_empty_cells += 1.0
                    else:
                        n_filled_cells += 1.0
        avg_images_per_cell = n_training_images / n_filled_cells
        print(
            f"n_empty_cells = {n_empty_cells}, n_filled_cells = {n_filled_cells}, n_training_images = {n_training_images}, avg_images_per_cell = {avg_images_per_cell:0.1f}"
        )

    output_filename = (
        f"{ctapipe_output}/output_machines/{pkl_name}_lookup_table_{telescope_type}.pkl"
    )
    with open(output_filename, "wb") as file:
        pickle.dump(lookup_table, file)

def BigMatrixSVD(
    ctapipe_output,
    telescope_type,
    big_matrix,
    feature_matrix,
    truth_matrix,
    image_rank,
    pkl_name,
):

    for idx in range(0,len(image_size_bins)-1):
        big_matrix[idx] = np.array(big_matrix[idx])
        print (f"big_matrix[{idx}].shape = {big_matrix[idx].shape}")

    eigenvectors = []
    for idx in range(0,len(image_size_bins)-1):
        n_images, n_pixels = big_matrix[idx].shape
        print(f"idx = {idx}, n_images = {n_images}, n_pixels = {n_pixels}")
        U_full, S_full, VT_full = np.linalg.svd(big_matrix[idx], full_matrices=False)
        U_eco = U_full[:, :image_rank]
        VT_eco = VT_full[:image_rank, :]
        eigenvectors += [VT_eco]

        fig, ax = plt.subplots()
        figsize_x = 6.4
        figsize_y = 4.6
        fig.set_figheight(figsize_y)
        fig.set_figwidth(figsize_x)
        label_x = "Rank"
        label_y = "Signular value"
        ax.set_xlabel(label_x,fontdict=font)
        ax.set_ylabel(label_y,fontdict=font)
        ax.set_xscale("log")
        ax.set_ylim(0., np.max(S_full))
        ax.set_xlim(1., len(S_full)+1)
        x_axis = np.linspace(1, len(S_full)+1, len(S_full))
        ax.plot(x_axis,S_full)
        fig.savefig(
            f"{ctapipe_output}/output_plots/training_{pkl_name}_signularvalue_{telescope_type}_size{idx}.png",
            bbox_inches="tight",
        )
        del fig
        del ax
        plt.close()


    print(f"saving image eigenvector to {ctapipe_output}/output_machines...")
    output_filename = f"{ctapipe_output}/output_machines/{pkl_name}_eigen_vectors_{telescope_type}.pkl"
    with open(output_filename, "wb") as file:
        pickle.dump(eigenvectors, file)

    MakeLookupTable(
        ctapipe_output,
        telescope_type,
        eigenvectors,
        big_matrix,
        feature_matrix,
        truth_matrix,
        image_rank,
        pkl_name + "_box3d",
        nvar=3,
    )


    physics_eigenvectors = []
    physics_mean_rms = []
    for idx in range(0,len(image_size_bins)-1):
        list_log_energy = []
        list_arrival = []
        list_log_arrival = []
        list_impact = []
        list_log_impact = []
        for img in range(0, len(truth_matrix[idx])):
            image_center_x = feature_matrix[idx][img][1]
            image_center_y = feature_matrix[idx][img][2]
            truth_energy = float(truth_matrix[idx][img][0] / u.TeV)
            truth_height = float(truth_matrix[idx][img][5] / u.m)
            truth_x_max = float(truth_matrix[idx][img][6] / (u.g / (u.cm * u.cm)))
            star_cam_x = truth_matrix[idx][img][7]
            star_cam_y = truth_matrix[idx][img][8]
            impact_x = truth_matrix[idx][img][9]
            impact_y = truth_matrix[idx][img][10]
            arrival = pow(
                pow(star_cam_x - image_center_x, 2) + pow(star_cam_y - image_center_y, 2),
                0.5,
            )
            list_log_arrival += [np.log10(arrival)]
            impact = pow(impact_x * impact_x + impact_y * impact_y, 0.5)
            list_log_impact += [np.log10(impact)]
            list_log_energy += [np.log10(truth_energy)]
            list_arrival += [arrival]
            list_impact += [impact]
        list_log_energy = np.array(list_log_energy)
        list_arrival = np.array(list_arrival)
        list_log_arrival = np.array(list_log_arrival)
        list_impact = np.array(list_impact)
        list_log_impact = np.array(list_log_impact)
        mean_log_energy = np.mean(list_log_energy)
        mean_arrival = np.mean(list_arrival)
        mean_log_arrival = np.mean(list_log_arrival)
        mean_impact = np.mean(list_impact)
        mean_log_impact = np.mean(list_log_impact)
        rms_log_energy = np.sqrt(np.mean(np.square(list_log_energy-mean_log_energy)))
        rms_arrival = np.sqrt(np.mean(np.square(list_arrival-mean_arrival)))
        rms_log_arrival = np.sqrt(np.mean(np.square(list_log_arrival-mean_log_arrival)))
        rms_impact = np.sqrt(np.mean(np.square(list_impact-mean_impact)))
        rms_log_impact = np.sqrt(np.mean(np.square(list_log_impact-mean_log_impact)))
        list_norm_log_energy = (list_log_energy-mean_log_energy)/rms_log_energy
        list_norm_arrival = (list_arrival-mean_arrival)/rms_arrival
        list_norm_log_arrival = (list_log_arrival-mean_log_arrival)/rms_log_arrival
        list_norm_impact = (list_impact-mean_impact)/rms_impact
        list_norm_log_impact = (list_log_impact-mean_log_impact)/rms_log_impact
        physics_mean_rms += [[mean_log_energy,rms_log_energy,mean_log_arrival,rms_log_arrival,mean_log_impact,rms_log_impact]]


        fig, ax = plt.subplots()
        figsize_x = 6.4
        figsize_y = 4.6
        fig.set_figheight(figsize_y)
        fig.set_figwidth(figsize_x)
        ax.scatter(list_norm_log_arrival, list_norm_log_impact, s=90, facecolors="none", c="r", marker="+", alpha=0.1)
        ax.set_xlabel("log arrival",fontdict=font)
        ax.set_ylabel("log impact",fontdict=font)
        ax.set_xlabel("normalized source location $\log d$")
        ax.set_ylabel("normalized impact location $\log b$")
        ax.set_xlim(-4., 3.)
        ax.set_ylim(-4., 3.)
        fig.savefig(
            f"{ctapipe_output}/output_plots/training_{pkl_name}_log_arrival_vs_log_impact_{telescope_type}_size{idx}.png",
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
        ax.scatter(list_norm_log_arrival, list_norm_log_energy, s=90, facecolors="none", c="r", marker="+", alpha=0.1)
        ax.set_xlabel("normalized source location $\log d$",fontdict=font)
        ax.set_ylabel("normalized shower energy $\log E$",fontdict=font)
        ax.set_xlim(-4., 3.)
        fig.savefig(
            f"{ctapipe_output}/output_plots/training_{pkl_name}_log_arrival_vs_log_energy_{telescope_type}_size{idx}.png",
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
        ax.scatter(list_norm_log_impact, list_norm_log_energy, s=90, facecolors="none", c="r", marker="+", alpha=0.1)
        ax.set_xlabel("normalized impact location $\log b$",fontdict=font)
        ax.set_ylabel("normalized shower energy $\log E$",fontdict=font)
        ax.set_xlim(-4., 3.)
        fig.savefig(
            f"{ctapipe_output}/output_plots/training_{pkl_name}_log_impact_vs_log_energy_{telescope_type}_size{idx}.png",
            bbox_inches="tight",
        )
        del fig
        del ax
        plt.close()

        physics_matrix = []
        for img in range(0, len(list_norm_log_energy)):
            #if list_norm_log_impact[img]<0.: continue
            physics_matrix += [[list_norm_log_energy[img],list_norm_log_arrival[img],list_norm_log_impact[img]]]

        physics_param_rank = 1
        u_full, s_full, vT_full = np.linalg.svd(physics_matrix, full_matrices=False)
        u_eco = u_full[:, :physics_param_rank]
        vT_eco = vT_full[:physics_param_rank, :]
        physics_eigenvectors += [vT_eco]

        fig, ax = plt.subplots()
        figsize_x = 6.4
        figsize_y = 4.6
        fig.set_figheight(figsize_y)
        fig.set_figwidth(figsize_x)
        label_x = "Rank"
        label_y = "Signular value"
        ax.set_xlabel(label_x,fontdict=font)
        ax.set_ylabel(label_y,fontdict=font)
        ax.set_xscale("log")
        ax.set_ylim(0., np.max(s_full))
        ax.set_xlim(1., len(s_full)+1)
        x_axis = np.linspace(1, len(s_full)+1, len(s_full))
        ax.plot(x_axis,s_full)
        fig.savefig(
            f"{ctapipe_output}/output_plots/training_{pkl_name}_physics_signularvalue_{telescope_type}_size{idx}.png",
            bbox_inches="tight",
        )
        del fig
        del ax
        plt.close()


    return eigenvectors, physics_eigenvectors, physics_mean_rms

def linear_regression(power,input_data, target_data, weight):
    # solve x*A = y using SVD
    # y_{0} = ( x_{0,0} x_{0,1} ... 1 )  a_{0}
    # y_{1} = ( x_{1,0} x_{1,1} ... 1 )  a_{1}
    # y_{2} = ( x_{2,0} x_{2,1} ... 1 )  .
    #                                    b

    x = []
    y = []
    w = []
    var = []
    for evt in range(0, len(input_data)):
        single_x = []
        for entry in range(0, len(input_data[evt])):
            for p in range(0, power):
                single_x += [pow(input_data[evt][entry],p)]
        x += [single_x]
        y += [target_data[evt]]
        w += [weight[evt]]
        var += [1./weight[evt]]
    x = np.array(x)
    y = np.array(y)
    w = np.diag(w)
    var = np.diag(var)

    # Compute the weighted SVD
    U, S, Vt = np.linalg.svd(w @ x, full_matrices=False)
    # Calculate the weighted pseudo-inverse
    S_pseudo_w = np.diag(1 / S)
    for row in range(0,len(S_pseudo_w)):
        for col in range(0,len(S_pseudo_w[row])):
            if S_pseudo_w[row][col] > 1e6*S_pseudo_w[0][0]:
                S_pseudo_w[row][col] = 0.
    x_pseudo_w = Vt.T @ S_pseudo_w @ U.T
    # Compute the weighted least-squares solution
    A_svd = x_pseudo_w @ (w @ y)
    # Compute parameter error
    P = x_pseudo_w @ w
    A_cov = P @ var @ P.T 
    A_err = np.sqrt(np.diag(A_cov))
    A_err_new = []
    for entry in range(0,len(A_err)):
        A_err_new += [[10.*A_err[entry]]]
    # Compute chi2
    chi2 = np.linalg.norm((w @ x).dot(A_svd) - (w @ y), 2) / np.trace(w)

    return A_svd, A_err_new, chi2


    ## Compute the weighted SVD
    #U, S, Vt = np.linalg.svd(x.T @ w @ x, full_matrices=False)
    ## Calculate the weighted pseudo-inverse
    #S_inv = np.diag(1 / S)
    #for row in range(0,len(S_inv)):
    #    for col in range(0,len(S_inv[row])):
    #        if S_inv[row][col] > 1e6*S_inv[0][0]:
    #            S_inv[row][col] = 0.
    #x_pseudo_inv = (Vt.T @ S_inv @ U.T) @ x.T
    ## Compute the weighted least-squares solution
    #A_svd = x_pseudo_inv @ (w @ y)
    ## Compute parameter error
    #A_cov = Vt.T @ S_inv @ U.T
    #A_err = np.sqrt(np.diag(A_cov))
    #A_err_new = []
    #for entry in range(0,len(A_err)):
    #    A_err_new += [[A_err[entry]]]

    ## Compute chi
    ##chi = (x.dot(A_svd) - y) @ np.sqrt(w)
    ##chi2 = np.sum(np.square(chi))
    #chi2 = 0.

    #return A_svd, A_err_new, chi2


def linear_model(power,input_data, A):
    x = []
    for entry in range(0, len(input_data)):
        for p in range(0, power):
            x += [pow(input_data[entry],p)]
    x = np.array(x)

    y = x @ A

    return y

def linear_model_err(power,input_data, A_err):
    x = []
    for entry in range(0, len(input_data)):
        for p in range(0, power):
            x += [pow(input_data[entry],p)]
    x = np.array(x)

    y_err = np.sqrt(x @ np.square(A_err))

    return y_err



class TinyModel(torch.nn.Module):

    def __init__(self,output_size):
        super(TinyModel, self).__init__()

        self.linear1 = torch.nn.Linear(output_size, 5)
        self.linear2 = torch.nn.Linear(5, 1)
        self.activation = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x

class CustomDataset(Dataset):
    def __init__(self, input_data, target_data):
        self.input_data = input_data
        self.target_data = target_data

    def __len__(self):
        return len(self.target_data)

    def __getitem__(self, idx):
        input_data = self.input_data[idx]
        target_data = self.target_data[idx]
        return input_data, target_data

def mapping_physical_params_to_latent_params(
    ctapipe_output,
    telescope_type,
    physics_eigenvectors,
    physics_mean_rms,
    image_eigenvectors,
    big_image_matrix,
    feature_matrix,
    truth_matrix,
    matrix_rank,
    pkl_name,
):

    model_image_to_physics = []
    for idx in range(0,len(image_size_bins)-1):

        list_mask_size = []
        list_evt_weight = []
        list_physics = []
        list_latent_space = []

        for img in range(0, len(big_image_matrix[idx])):

            mask_size = feature_matrix[idx][img][0]
            image_center_x = feature_matrix[idx][img][1]
            image_center_y = feature_matrix[idx][img][2]
            angle = feature_matrix[idx][img][3]
            semi_major = feature_matrix[idx][img][4]
            semi_minor = feature_matrix[idx][img][5]
            image_direction_err = feature_matrix[idx][img][6]
            image_direction = feature_matrix[idx][img][7]
            line_a = feature_matrix[idx][img][8]
            line_b = feature_matrix[idx][img][9]
            truth_projection = feature_matrix[idx][img][10]
            line_a_err = feature_matrix[idx][img][11]
            line_b_err = feature_matrix[idx][img][12]
            angle_err = feature_matrix[idx][img][13]
            image_size = feature_matrix[idx][img][14]

            if angle_err==0.: continue
            if angle_err>0.1: continue

            truth_energy = float(truth_matrix[idx][img][0] / u.TeV)
            truth_height = float(truth_matrix[idx][img][5] / u.m)
            truth_x_max = float(truth_matrix[idx][img][6] / (u.g / (u.cm * u.cm)))
            star_cam_x = truth_matrix[idx][img][7]
            star_cam_y = truth_matrix[idx][img][8]
            impact_x = truth_matrix[idx][img][9]
            impact_y = truth_matrix[idx][img][10]
            focal_length = float(truth_matrix[idx][img][11] / u.m)

            arrival = pow(
                pow(star_cam_x - image_center_x, 2) + pow(star_cam_y - image_center_y, 2),
                0.5,
            )
            impact = pow(impact_x * impact_x + impact_y * impact_y, 0.5)
            log_arrival = np.log10(arrival)
            log_impact = np.log10(impact)

            arrival_unc_1 = line_b_err*semi_major/semi_minor
            arrival_unc_2 = angle_err*arrival
            arrival_unc = pow(arrival_unc_1*arrival_unc_1 + arrival_unc_2*arrival_unc_2,0.5)
            log_arrival_unc = arrival_unc/arrival

            impact_unc_1 = line_b_err*semi_major/semi_minor
            impact_unc_2 = angle_err*impact
            impact_unc = pow(impact_unc_1*impact_unc_1 + impact_unc_2*impact_unc_2,0.5)
            log_impact_unc = impact_unc/impact

            truth_energy_unc = pow(image_size,0.5)/image_size * truth_energy
            log_truth_energy = np.log10(truth_energy)
            log_truth_energy_unc = truth_energy_unc/truth_energy

            image_1d = np.array(big_image_matrix[idx][img])
            image_latent_space = image_eigenvectors[idx] @ image_1d
            list_latent_space += [image_latent_space]

            list_mask_size += [mask_size]
            norm_log_energy = (log_truth_energy-physics_mean_rms[idx][0])/physics_mean_rms[idx][1]
            norm_log_arrival = (log_arrival-physics_mean_rms[idx][2])/physics_mean_rms[idx][3]
            norm_log_impact = (log_impact-physics_mean_rms[idx][4])/physics_mean_rms[idx][5]
            physics_latent_space = physics_eigenvectors[idx] @ np.array([norm_log_energy,norm_log_arrival,norm_log_impact])
            list_physics += [physics_latent_space]

            norm_log_energy_unc = (log_truth_energy_unc)/physics_mean_rms[idx][1]
            norm_log_arrival_unc = (log_arrival_unc)/physics_mean_rms[idx][3]
            norm_log_impact_unc = (log_impact_unc)/physics_mean_rms[idx][5]
            physics_latent_space_unc_1 = physics_eigenvectors[idx] @ np.array([norm_log_energy_unc,0.,0.])
            physics_latent_space_unc_2 = physics_eigenvectors[idx] @ np.array([0.,norm_log_arrival_unc,0.])
            physics_latent_space_unc_3 = physics_eigenvectors[idx] @ np.array([0.,0.,norm_log_impact_unc])
            physics_latent_space_unc = np.sqrt(np.square(physics_latent_space_unc_1) + np.square(physics_latent_space_unc_2) + np.square(physics_latent_space_unc_3))
            #list_evt_weight += [1. / (physics_latent_space_unc[0]*physics_latent_space_unc[0])]
            list_evt_weight += [1. / (angle_err*angle_err)]

        list_evt_weight = np.array(list_evt_weight)
        list_physics = np.array(list_physics)
        list_latent_space = np.array(list_latent_space)
        print (f"list_evt_weight.shape = {list_evt_weight.shape}")
        print (f"list_physics.shape = {list_physics.shape}")
        print (f"list_latent_space.shape = {list_latent_space.shape}")

        target = list_physics
        input_param = list_latent_space
        #target = list_latent_space
        #input_param = list_physics
        model, model_err, chi = linear_regression(fast_prediction_power,input_param, target, list_evt_weight)

        #target = torch.tensor(np.array(list_physics))
        #target = target.to(torch.float32)
        #input_param = torch.tensor(np.array(list_latent_space))
        #input_param = input_param.to(torch.float32)
        #train_dataset = CustomDataset(input_param,target)
        #train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        #print (f"len(train_dataset) = {len(train_dataset)}")
        #print (f"train_dataset[0] = {train_dataset[0]}")
        #model = TinyModel(len(list_latent_space[0]))
        #print(model)

        #criterion = nn.CrossEntropyLoss()
        #optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        #num_epoch = 5
        #train_loss = []
        #for epoch in range(num_epoch):
        #    model.train()
        #    for set_input_params, set_targets in train_dataloader:
        #        optimizer.zero_grad()
        #        set_target_predicts = model(set_input_params)
        #        loss = criterion(set_target_predicts, set_targets)
        #        loss.backward()
        #        optimizer.step()

        #    model.eval()
        #    with torch.no_grad():
        #        for set_input_params, set_targets in train_dataloader:
        #            set_target_predicts = model(set_input_params)
        #            loss = criterion(set_target_predicts, set_targets)
        #            print(loss.item())


        list_log_energy_err = []
        list_arrival_err = []
        list_arrival_unc = []
        for img in range(0, len(big_image_matrix[idx])):

            mask_size = feature_matrix[idx][img][0]
            image_center_x = feature_matrix[idx][img][1]
            image_center_y = feature_matrix[idx][img][2]
            image_direction_err = feature_matrix[idx][img][6]
            image_direction = feature_matrix[idx][img][7]
            image_angle_err = feature_matrix[idx][img][13]

            if image_angle_err==0.: continue
            if image_angle_err>0.1: continue

            truth_energy = float(truth_matrix[idx][img][0] / u.TeV)
            truth_height = float(truth_matrix[idx][img][5] / u.m)
            truth_x_max = float(truth_matrix[idx][img][6] / (u.g / (u.cm * u.cm)))
            star_cam_x = truth_matrix[idx][img][7]
            star_cam_y = truth_matrix[idx][img][8]
            impact_x = truth_matrix[idx][img][9]
            impact_y = truth_matrix[idx][img][10]
            focal_length = float(truth_matrix[idx][img][11] / u.m)

            arrival = pow(
                pow(star_cam_x - image_center_x, 2) + pow(star_cam_y - image_center_y, 2),
                0.5,
            )
            impact = pow(impact_x * impact_x + impact_y * impact_y, 0.5)
            log_arrival = np.log10(arrival)
            log_impact = np.log10(impact)

            norm_log_energy = (np.log10(truth_energy)-physics_mean_rms[idx][0])/physics_mean_rms[idx][1]
            norm_log_arrival = (log_arrival-physics_mean_rms[idx][2])/physics_mean_rms[idx][3]
            norm_log_impact = (log_impact-physics_mean_rms[idx][4])/physics_mean_rms[idx][5]
            physics_latent_space = physics_eigenvectors[idx] @ np.array([norm_log_energy,norm_log_arrival,norm_log_impact])

            image_1d = np.array(big_image_matrix[idx][img])

            (
                fit_log_energy, 
                err_log_energy, 
                fit_arrival, 
                err_arrival, 
                fit_impact, 
                err_impact
            ) = fast_image_to_physics_prediction(
                image_angle_err,
                image_1d,
                image_eigenvectors[idx],
                [model,model_err,physics_eigenvectors[idx],physics_mean_rms[idx]],
            )
            unc_arrival = err_arrival

            #trial_rms_arrival = 0.
            #n_trials = 10
            #for t in range(0,n_trials):
            #    image_1d_perturbation = np.zeros_like(image_1d)
            #    for entry in range(0,len(image_1d)):
            #        image_1d_perturbation[entry] = np.random.poisson(max(0.,image_1d[entry]))

            #    (
            #        try_log_energy, 
            #        err_log_energy, 
            #        try_arrival, 
            #        err_arrival, 
            #        try_impact, 
            #        err_impact
            #    ) = fast_image_to_physics_prediction(
            #        image_1d_perturbation,
            #        image_eigenvectors[idx],
            #        [model,model_err,physics_eigenvectors[idx],physics_mean_rms[idx]],
            #    )
            #    trial_rms_arrival += pow(try_arrival-fit_arrival,2)
            #trial_rms_arrival = pow(trial_rms_arrival/float(n_trials),0.5)
            #unc_arrival = trial_rms_arrival

            #init_params = [physics_latent_space[0]]
            #stepsize = [0.01]
            #ftol = 0.01
            #solution = minimize(
            #    single_template_chi2,
            #    x0=init_params,
            #    args=(image_1d,image_eigenvectors[idx],model),
            #    method='L-BFGS-B',
            #    jac=None,
            #    options={'eps':stepsize,'ftol':ftol},
            #)
            #final_chi2 = solution['fun']
            #fit_params = solution['x']
            #fit_physics_latent_space = fit_params[0]
            #fit_log_physics = physics_eigenvectors[idx].T @ np.array([fit_physics_latent_space])
            #fit_log_energy = fit_log_physics[0]*physics_mean_rms[idx][1] + physics_mean_rms[idx][0]
            #fit_log_arrival = fit_log_physics[1]*physics_mean_rms[idx][3] + physics_mean_rms[idx][2]
            #fit_arrival = pow(10.,fit_log_arrival)

            list_log_energy_err += [(fit_log_energy-np.log10(truth_energy))]
            list_arrival_err += [(fit_arrival-arrival)]
            list_arrival_unc += [unc_arrival]

            #physics_param = torch.tensor(np.array([truth_energy,arrival,impact]))
            #physics_param = torch.tensor(np.array([arrival]))
            #physics_param = physics_param.to(torch.float32)
            #input_latent_space = torch.tensor(np.array(image_latent_space))
            #input_latent_space = input_latent_space.to(torch.float32)
            #predict_physics = model(input_latent_space)


        print ("================================================")
        log_energy_err_rms = np.sqrt(np.mean(np.square(np.array(list_log_energy_err))))
        arrival_err_rms = np.sqrt(np.mean(np.square(np.array(list_arrival_err))))
        arrival_unc_rms = np.sqrt(np.mean(np.square(np.array(list_arrival_unc))))
        print (f"arrival_err_rms = {arrival_err_rms/focal_length*180./np.pi}")
        print (f"arrival_unc_rms = {arrival_unc_rms/focal_length*180./np.pi}")

        fig, ax = plt.subplots()
        figsize_x = 6.4
        figsize_y = 4.6
        fig.set_figheight(figsize_y)
        fig.set_figwidth(figsize_x)
        label_x = "arrival direction error [deg]"
        label_y = "number of images"
        ax.set_xlabel(label_x,fontdict=font)
        ax.set_ylabel(label_y,fontdict=font)
        ax.hist(np.array(list_arrival_err)/focal_length*180./np.pi,histtype='step',bins=100,range=(-0.5,0.5),label=f"RMS = {arrival_err_rms/focal_length*180./np.pi:0.2f} deg")
        ax.legend(loc='best')
        fig.savefig(
            f"{ctapipe_output}/output_plots/training_{pkl_name}_arrival_err_{telescope_type}_size{idx}.png",
            bbox_inches="tight",
        )
        del fig
        del ax
        plt.close()


        model_image_to_physics += [[model,model_err,physics_eigenvectors[idx],physics_mean_rms[idx]]]


    output_filename = f"{ctapipe_output}/output_machines/{pkl_name}_fast_conversion_model_{telescope_type}.pkl"
    with open(output_filename, "wb") as file:
        pickle.dump(model_image_to_physics, file)

        

#def MakeFastConversionImage(
#    ctapipe_output,
#    telescope_type,
#    image_eigenvectors,
#    big_image_matrix,
#    time_eigenvectors,
#    big_time_matrix,
#    feature_matrix,
#    truth_matrix,
#    pkl_name,
#):
#    list_mask_size = []
#    list_evt_weight = []
#    list_arrival = []
#    list_impact = []
#    list_log_energy = []
#    list_latent_space = []
#
#    for img in range(0, len(big_image_matrix)):
#
#        mask_size = feature_matrix[img][0]
#        image_center_x = feature_matrix[img][1]
#        image_center_y = feature_matrix[img][2]
#        time_direction = feature_matrix[img][6]
#        image_direction = feature_matrix[img][7]
#        image_angle_err = feature_matrix[img][13]
#        image_qual = abs(image_direction + time_direction)
#
#        if image_angle_err==0.: continue
#
#        truth_energy = float(truth_matrix[img][0] / u.TeV)
#        truth_height = float(truth_matrix[img][5] / u.m)
#        truth_x_max = float(truth_matrix[img][6] / (u.g / (u.cm * u.cm)))
#        star_cam_x = truth_matrix[img][7]
#        star_cam_y = truth_matrix[img][8]
#        impact_x = truth_matrix[img][9]
#        impact_y = truth_matrix[img][10]
#
#        arrival = pow(
#            pow(star_cam_x - image_center_x, 2) + pow(star_cam_y - image_center_y, 2),
#            0.5,
#        )
#        impact = pow(impact_x * impact_x + impact_y * impact_y, 0.5)
#        log_energy = np.log10(truth_energy)
#
#        image_1d = np.array(big_image_matrix[img])
#        image_latent_space = image_eigenvectors @ image_1d
#        time_1d = np.array(big_time_matrix[img])
#        time_latent_space = time_eigenvectors @ time_1d
#        list_latent_space += [np.concatenate((image_latent_space, time_latent_space))]
#
#        list_mask_size += [mask_size]
#        list_evt_weight += [0.0001 / image_angle_err]
#        list_arrival += [arrival]
#        list_impact += [impact]
#        list_log_energy += [log_energy]
#
#    target = list_arrival
#    model, chi = linear_regression(list_latent_space, target, list_evt_weight)
#
#    output_filename = f"{ctapipe_output}/output_machines/{pkl_name}_fast_conversion_arrival_{telescope_type}.pkl"
#    with open(output_filename, "wb") as file:
#        pickle.dump(model, file)
#
#    target = list_impact
#    model, chi = linear_regression(list_latent_space, target, list_evt_weight)
#
#    output_filename = f"{ctapipe_output}/output_machines/{pkl_name}_fast_conversion_impact_{telescope_type}.pkl"
#    with open(output_filename, "wb") as file:
#        pickle.dump(model, file)
#
#    target = list_log_energy
#    model, chi = linear_regression(list_latent_space, target, list_evt_weight)
#
#    output_filename = f"{ctapipe_output}/output_machines/{pkl_name}_fast_conversion_log_energy_{telescope_type}.pkl"
#    with open(output_filename, "wb") as file:
#        pickle.dump(model, file)


def sqaure_difference_between_1d_images(
    init_params, data_latent_space, lookup_table, eigen_vectors, full_table=False
):
    fit_arrival = init_params[0]
    fit_impact = init_params[1]
    fit_log_energy = init_params[2]
    if not full_table:
        if (
            lookup_table[0].get_bin_content(fit_arrival, fit_impact, fit_log_energy)
            == 0.0
        ):
            return 1e10

    fit_latent_space = []
    key_idx = lookup_table[0].get_bin(fit_arrival, fit_impact, fit_log_energy)
    key_idx_x = key_idx[0]
    key_idx_y = key_idx[1]
    key_idx_z = key_idx[2]
    for r in range(0, len(lookup_table)):
        fit_latent_space += [
            lookup_table[r].get_bin_content_by_index(key_idx_x, key_idx_y, key_idx_z)
        ]
    fit_latent_space = np.array(fit_latent_space)

    sum_chi2 = 0.0
    n_rows = len(data_latent_space)
    for row in range(0, n_rows):
        if data_latent_space[row] == 0.0 and fit_latent_space[row] == 0.0:
            continue
        diff = data_latent_space[row] - fit_latent_space[row]
        sum_chi2 += diff * diff

    return sum_chi2


def sortFirst(val):
    return val[0]


def box_search(
    init_params,
    input_image_1d,
    image_lookup_table,
    image_eigen_vectors,
    input_time_1d,
    time_lookup_table,
    time_eigen_vectors,
    input_movie_1d,
    movie_lookup_table,
    movie_eigen_vectors,
    arrival_range,
    impact_range,
    log_energy_range,
):
    movie_latent_space = movie_eigen_vectors @ input_movie_1d
    image_latent_space = image_eigen_vectors @ input_image_1d
    time_latent_space = time_eigen_vectors @ input_time_1d

    image_norm = np.sum(np.abs(image_latent_space))
    time_norm = np.sum(np.abs(time_latent_space))
    movie_norm = np.sum(np.abs(movie_latent_space))

    init_arrival = init_params[0]
    init_impact = init_params[1]
    init_log_energy = init_params[2]
    short_list = []

    while len(short_list) == 0:
        fit_idx_x = 0
        fit_idx_y = 0
        fit_idx_z = 0
        for idx_x in range(0, n_bins_arrival):
            try_arrival = image_lookup_table[0].xaxis[idx_x]
            if abs(init_arrival - try_arrival) > arrival_range:
                continue
            for idx_y in range(0, n_bins_impact):
                try_impact = image_lookup_table[0].yaxis[idx_y]
                if abs(init_impact - try_impact) > impact_range:
                    continue
                for idx_z in range(0, n_bins_energy):
                    try_log_energy = image_lookup_table[0].zaxis[idx_z]
                    if abs(init_log_energy - try_log_energy) > log_energy_range:
                        continue

                    try_params = [try_arrival, try_impact, try_log_energy]

                    try_chi2 = 0.0

                    #try_chi2_image = (
                    #    sqaure_difference_between_1d_images(
                    #        try_params,
                    #        image_latent_space,
                    #        image_lookup_table,
                    #        image_eigen_vectors,
                    #    )
                    #    / image_norm
                    #)
                    #try_chi2 += try_chi2_image
                    #try_chi2_time = (
                    #    sqaure_difference_between_1d_images(
                    #        try_params,
                    #        time_latent_space,
                    #        time_lookup_table,
                    #        time_eigen_vectors,
                    #    )
                    #    / time_norm
                    #)
                    #try_chi2 += try_chi2_time

                    try_chi2_movie = (
                       sqaure_difference_between_1d_images(
                           try_params,
                           movie_latent_space,
                           movie_lookup_table,
                           movie_eigen_vectors,
                       )
                       / movie_norm
                    )
                    try_chi2 += try_chi2_movie

                    short_list += [(try_chi2, try_arrival, try_impact, try_log_energy)]

        if len(short_list) == 0:
            print("short_list is zero. expand search range.")
            arrival_range = 1e10
            impact_range = 1e10
            log_energy_range = 1e10
        else:
            break

    short_list.sort(key=sortFirst)
    fit_chi2 = short_list[0][0]
    fit_arrival = short_list[0][1]
    fit_impact = short_list[0][2]
    fit_log_energy = short_list[0][3]

    err_arrival = 0.
    err_impact = 0.
    err_log_energy = 0.
    if len(short_list) > 1:
        sum_likelihood = 0.0
        #for entry in range(0, min(5, len(short_list))):
        for entry in range(0, len(short_list)):
            chi2 = short_list[entry][0]
            arrival = short_list[entry][1]
            impact = short_list[entry][2]
            log_energy = short_list[entry][3]
            norm_likelihood = 1. / (chi2-fit_chi2+1.)
            #norm_likelihood = np.exp( (fit_chi2 - chi2) )
            err_arrival += pow(arrival - fit_arrival, 2) * norm_likelihood
            err_impact += pow(impact - fit_impact, 2) * norm_likelihood
            err_log_energy += pow(log_energy - fit_log_energy, 2) * norm_likelihood
            sum_likelihood += norm_likelihood
        if sum_likelihood>0.:
            err_arrival = pow(err_arrival / sum_likelihood, 0.5)
            err_impact = pow(err_impact / sum_likelihood, 0.5)
            err_log_energy = pow(err_log_energy / sum_likelihood, 0.5)
        else:
            err_arrival = 1e10
            err_impact = 1e10
            err_log_energy = 1e10

    return (
        short_list,
        fit_arrival,
        fit_impact,
        fit_log_energy,
        err_arrival,
        err_impact,
        err_log_energy,
        fit_chi2,
    )

def correlation_between_1d_images_poisson(
    init_params, image_1d_data, lookup_table, eigen_vectors, full_table=False
):
    fit_arrival = init_params[0]
    fit_impact = init_params[1]
    fit_log_energy = init_params[2]
    if not full_table:
        if (
            lookup_table[0].get_bin_content(fit_arrival, fit_impact, fit_log_energy)
            == 0.0
        ):
            return 0.

    fit_latent_space = []
    key_idx = lookup_table[0].get_bin(fit_arrival, fit_impact, fit_log_energy)
    key_idx_x = key_idx[0]
    key_idx_y = key_idx[1]
    key_idx_z = key_idx[2]
    for r in range(0, len(lookup_table)):
        fit_latent_space += [
            lookup_table[r].get_bin_content_by_index(key_idx_x, key_idx_y, key_idx_z)
        ]
    fit_latent_space = np.array(fit_latent_space)

    data_latent_space = eigen_vectors @ image_1d_data

    correlation = 0.0
    correlation_norm = 0.0
    for entry in range(0,len(data_latent_space)):
        correlation += pow(data_latent_space[entry]-fit_latent_space[entry],2)
        correlation_norm += pow(data_latent_space[entry],2)

    return correlation/correlation_norm


def sqaure_difference_between_1d_images_poisson(
    init_params, image_1d_data, lookup_table, eigen_vectors, full_table=False
):
    fit_arrival = init_params[0]
    fit_impact = init_params[1]
    fit_log_energy = init_params[2]
    if not full_table:
        if (
            lookup_table[0].get_bin_content(fit_arrival, fit_impact, fit_log_energy)
            == 0.0
        ):
            return 1e10

    fit_latent_space = []
    key_idx = lookup_table[0].get_bin(fit_arrival, fit_impact, fit_log_energy)
    key_idx_x = key_idx[0]
    key_idx_y = key_idx[1]
    key_idx_z = key_idx[2]
    for r in range(0, len(lookup_table)):
        fit_latent_space += [
            lookup_table[r].get_bin_content_by_index(key_idx_x, key_idx_y, key_idx_z)
        ]
    fit_latent_space = np.array(fit_latent_space)

    data_latent_space = eigen_vectors @ image_1d_data

    sum_log_likelihood = 0.0
    image_1d_fit = eigen_vectors.T @ fit_latent_space
    n_rows = len(image_1d_fit)
    for row in range(0, n_rows):
        n_expect = max(0.0001, image_1d_fit[row])
        n_data = max(0.0, image_1d_data[row])
        if n_data == 0.0:
            sum_log_likelihood += n_expect
        else:
            sum_log_likelihood += -1.0 * (
                n_data * np.log(n_expect)
                - n_expect
                - (n_data * np.log(n_data) - n_data)
            )

    chi2 = 0.
    chi2_norm = 0.
    for row in range(0, n_rows):
        chi2 += pow(image_1d_data[row]-image_1d_fit[row],2)
        chi2_norm += image_1d_data[row]


    return chi2/chi2_norm


def analyze_short_list(
    short_list, init_params, input_movie_1d, movie_lookup_table, movie_eigen_vectors
):
    # print(f"len(short_list) = {len(short_list)}")
    fit_chi2 = 1e10
    fit_likelihood = 0.0
    n_short_list = 5
    fit_arrival = init_params[0]
    fit_impact = init_params[1]
    fit_log_energy = init_params[2]
    list_chi2 = []
    list_likelihood = []
    list_arrival = []
    list_impact = []
    list_log_energy = []
    movie_latent_space = movie_eigen_vectors @ input_movie_1d
    movie_latent_space_norm = np.sum(np.sqrt(np.square(movie_latent_space)))
    #for entry in range(0, min(n_short_list, len(short_list))):
    for entry in range(0, len(short_list)):
        try_arrival = short_list[entry][1]
        try_impact = short_list[entry][2]
        try_log_energy = short_list[entry][3]
        init_params = [try_arrival, try_impact, try_log_energy]
        try_chi2 = sqaure_difference_between_1d_images_poisson(
            init_params, input_movie_1d, movie_lookup_table, movie_eigen_vectors
        )
        #try_chi2 = (
        #   sqaure_difference_between_1d_images(
        #       init_params,
        #       movie_latent_space,
        #       movie_lookup_table,
        #       movie_eigen_vectors,
        #   )
        #) / movie_latent_space_norm
        try_likelihood = np.exp(-try_chi2)
        list_chi2 += [try_chi2]
        list_likelihood += [try_likelihood]
        list_arrival += [try_arrival]
        list_impact += [try_impact]
        list_log_energy += [try_log_energy]
        if try_chi2 < fit_chi2:
            fit_chi2 = try_chi2
            fit_likelihood = try_likelihood
            fit_arrival = try_arrival
            fit_impact = try_impact
            fit_log_energy = try_log_energy

    err_arrival = 1e10
    err_impact = 1e10
    err_log_energy = 1e10
    if len(list_likelihood) > 1:
        err_arrival = 0.0
        err_impact = 0.0
        err_log_energy = 0.0
        sum_likelihood = 0.0
        for entry in range(0, len(list_likelihood)):
            #norm_likelihood = fit_chi2 / (list_chi2[entry])
            norm_likelihood = 1. / (list_chi2[entry]-fit_chi2+1.)
            #norm_likelihood = np.exp(-list_chi2[entry]+fit_chi2)
            #norm_likelihood = list_likelihood[entry]
            err_arrival += pow(list_arrival[entry] - fit_arrival, 2) * norm_likelihood
            err_impact += pow(list_impact[entry] - fit_impact, 2) * norm_likelihood
            err_log_energy += (
                pow(list_log_energy[entry] - fit_log_energy, 2) * norm_likelihood
            )
            sum_likelihood += norm_likelihood
        err_arrival = pow(err_arrival / sum_likelihood, 0.5)
        err_impact = pow(err_impact / sum_likelihood, 0.5)
        err_log_energy = pow(err_log_energy / sum_likelihood, 0.5)

    return (
        fit_arrival,
        fit_impact,
        fit_log_energy,
        err_arrival,
        err_impact,
        err_log_energy,
        fit_chi2,
    )

def fast_image_to_physics_prediction(
    image_angle_err,
    input_image_1d,
    image_eigenvectors,
    image_to_physics_model,
):

    physics_to_image_latent_space = image_to_physics_model[0]
    physics_to_image_latent_space_err = image_to_physics_model[1]
    physics_eigenvectors = image_to_physics_model[2]
    physics_mean_rms = image_to_physics_model[3]

    image_latent_space = image_eigenvectors @ input_image_1d

    predict_physics_latent_space = linear_model(fast_prediction_power, image_latent_space, physics_to_image_latent_space)
    predict_physics_latent_space_err = linear_model_err(fast_prediction_power, image_latent_space, physics_to_image_latent_space_err)
    predict_physics = physics_eigenvectors.T @ predict_physics_latent_space
    predict_physics_err = np.sqrt(physics_eigenvectors.T @ np.square(predict_physics_latent_space_err))
    fit_log_energy = predict_physics[0]*physics_mean_rms[1] + physics_mean_rms[0]
    fit_log_arrival = predict_physics[1]*physics_mean_rms[3] + physics_mean_rms[2]
    fit_log_impact = predict_physics[2]*physics_mean_rms[5] + physics_mean_rms[4]
    err_log_energy = predict_physics_err[0]*physics_mean_rms[1]
    err_log_arrival = predict_physics_err[1]*physics_mean_rms[3]
    err_log_impact = predict_physics_err[2]*physics_mean_rms[5]
    fit_arrival = pow(10.,fit_log_arrival)
    err_arrival = fit_arrival*np.log(10)*err_log_arrival
    fit_impact = pow(10.,fit_log_impact)
    err_impact = fit_impact*np.log(10)*err_log_impact

    unc_arrival = pow(pow(fit_arrival*image_angle_err,2)+pow(err_arrival,2),0.5)

    return fit_log_energy, err_log_energy, fit_arrival, unc_arrival, fit_impact, err_impact

def single_template_chi2(
    try_params,
    eco_image_1d,
    image_eigen_vectors,
    physics_to_image_latent_space,
    ):

    try_physics_latent_space = np.array([try_params[0]])

    predict_image_latent_space = linear_model(fast_prediction_power, try_physics_latent_space, physics_to_image_latent_space)
    image_latent_space = image_eigen_vectors @ eco_image_1d

    chi2 = 0.
    for entry in range(0,len(image_latent_space)):
        chi2 += pow(image_latent_space[entry]-predict_image_latent_space[entry],2)

    return chi2


def single_movie_reconstruction(
    telescope_type,
    pixel_width,
    angle_err,
    image_size,
    input_image_1d,
    dict_image_lookup_table,
    dict_image_eigen_vectors,
    input_time_1d,
    dict_time_lookup_table,
    dict_time_eigen_vectors,
    input_movie_1d,
    dict_movie_lookup_table,
    dict_movie_eigen_vectors,
    dict_image_to_physics_model,
):

    image_lookup_table = dict_image_lookup_table[telescope_type]
    image_eigen_vectors = dict_image_eigen_vectors[telescope_type]
    time_lookup_table = dict_time_lookup_table[telescope_type]
    time_eigen_vectors = dict_time_eigen_vectors[telescope_type]
    movie_lookup_table = dict_movie_lookup_table[telescope_type]
    movie_eigen_vectors = dict_movie_eigen_vectors[telescope_type]
    image_to_physics_model = dict_image_to_physics_model[telescope_type]

    arrival_step_size = 0.5*(arrival_upper - arrival_lower) / float(n_bins_arrival)
    impact_step_size = 0.5*(impact_upper - impact_lower) / float(n_bins_impact)
    log_energy_step_size = 0.5*(log_energy_upper - log_energy_lower) / float(n_bins_energy)

    image_idx = 0
    for idx in range(0,len(image_size_bins)-1):
        if image_size>=image_size_bins[idx] and image_size<image_size_bins[idx+1]:
            image_idx = idx

    print (f"image_size = {image_size}")
    print (f"image_idx = {image_idx}")

    (
        fit_log_energy, 
        err_log_energy, 
        fit_arrival, 
        err_arrival, 
        fit_impact, 
        err_impact
    ) = fast_image_to_physics_prediction(
        angle_err,
        input_movie_1d,
        movie_eigen_vectors[image_idx],
        image_to_physics_model[image_idx],
    )
    #err_arrival = max(0.5*pixel_width,err_arrival)

    #init_log_arrival = 0.5*(np.log10(arrival_upper) + np.log10(arrival_lower+1e-5))
    #init_log_energy = 0.5*(log_energy_upper + log_energy_lower)
    #norm_log_energy = (init_log_energy-physics_mean_rms[0])/physics_mean_rms[1]
    #norm_log_arrival = (init_log_arrival-physics_mean_rms[2])/physics_mean_rms[3]
    #init_physics_latent_space = physics_eigenvectors @ np.array([norm_log_energy,norm_log_arrival])

    #init_params = [init_physics_latent_space[0]]
    ##bounds = ((arrival_lower+1e-5,arrival_upper), (log_energy_lower,log_energy_upper))
    #stepsize = [0.01]
    #ftol = 0.01
    #solution = minimize(
    #    single_template_chi2,
    #    x0=init_params,
    #    args=(input_movie_1d,movie_eigen_vectors[image_idx],physics_to_image_latent_space),
    #    #bounds=bounds,
    #    method='L-BFGS-B',
    #    jac=None,
    #    options={'eps':stepsize,'ftol':ftol},
    #)
    #final_chi2 = solution['fun']
    #print (f"final_chi2 = {final_chi2}")
    #fit_params = solution['x']
    #fit_physics_latent_space = fit_params[0]
    #fit_log_physics = physics_eigenvectors.T @ np.array([fit_physics_latent_space])
    #fit_log_energy = fit_log_physics[0]*physics_mean_rms[1] + physics_mean_rms[0]
    #fit_log_arrival = fit_log_physics[1]*physics_mean_rms[3] + physics_mean_rms[2]
    #fit_arrival = pow(10.,fit_log_arrival)

    #tmp_i = np.zeros(len(fit_params))
    #hess_inv = []
    #for i in range(len(fit_params)):
    #    tmp_i[i] = 1.
    #    hess_inv += [solution.hess_inv(tmp_i)[i]]
    #    tmp_i[i] = 0.

    #err_physics_latent_space = np.sqrt(max(1., abs(final_chi2)) * ftol * hess_inv[0])
    #err_log_physics = physics_eigenvectors.T @ np.array([err_physics_latent_space])
    #err_log_energy = err_log_physics[0]*physics_mean_rms[1]
    #err_log_arrival = err_log_physics[1]*physics_mean_rms[3]
    #err_arrival = fit_arrival*err_log_arrival*np.log(10.)
    #err_arrival = pow(pow(err_arrival,2) + pow(angle_err*fit_arrival,2),0.5)

    print (f"fit_arrival = {fit_arrival:0.4f} +/- {err_arrival:0.4f}")
    print (f"fit_log_energy = {fit_log_energy:0.4f} +/- {err_log_energy:0.4f}")

    return (
        fit_arrival,
        fit_impact,
        fit_log_energy,
        err_arrival,
        err_impact,
        err_log_energy,
    )



def camxy_to_altaz(source, subarray, run_id, tel_id, star_cam_x, star_cam_y):
    obstime = Time("2013-11-01T03:00")
    location = EarthLocation.of_site("Roque de los Muchachos")
    altaz = AltAz(location=location, obstime=obstime)

    tel_pointing_alt = source.observation_blocks[run_id].subarray_pointing_lat
    tel_pointing_az = source.observation_blocks[run_id].subarray_pointing_lon

    focal_length = source.subarray.tel[tel_id].optics.equivalent_focal_length

    tel_pointing = SkyCoord(
        alt=tel_pointing_alt,
        az=tel_pointing_az,
        frame=altaz,
    )

    camera_frame = CameraFrame(
        telescope_pointing=tel_pointing,
        focal_length=focal_length,
    )

    star_cam = SkyCoord(
        x=star_cam_x * u.m,
        y=star_cam_y * u.m,
        frame=camera_frame,
    )

    star_altaz = star_cam.transform_to(altaz)
    star_alt = star_altaz.alt.to_value(u.rad)
    star_az = star_altaz.az.to_value(u.rad)

    star_az_2pi = star_az - 2.0 * np.pi
    if abs(star_az_2pi - 0.0) < abs(star_az - 0.0):
        star_az = star_az_2pi

    return star_alt, star_az


def altaz_to_camxy(source, subarray, run_id, tel_id, star_alt, star_az):
    obstime = Time("2013-11-01T03:00")
    location = EarthLocation.of_site("Roque de los Muchachos")
    altaz = AltAz(location=location, obstime=obstime)

    if star_alt>np.pi/2.*u.rad:
        star_alt = np.pi*u.rad - star_alt

    star_altaz = SkyCoord(
        alt=star_alt,
        az=star_az,
        frame=altaz,
    )

    tel_pointing_alt = source.observation_blocks[run_id].subarray_pointing_lat
    tel_pointing_az = source.observation_blocks[run_id].subarray_pointing_lon

    focal_length = source.subarray.tel[tel_id].optics.equivalent_focal_length

    tel_pointing = SkyCoord(
        alt=tel_pointing_alt,
        az=tel_pointing_az,
        frame=altaz,
    )

    camera_frame = CameraFrame(
        telescope_pointing=tel_pointing,
        focal_length=focal_length,
    )


    star_cam = star_altaz.transform_to(camera_frame)
    star_cam_x = star_cam.x.to_value(u.m)
    star_cam_y = star_cam.y.to_value(u.m)

    return star_cam_x, star_cam_y

def camxy_to_nominal(source, subarray, run_id, tel_id, star_cam_x, star_cam_y):
    obstime = Time("2013-11-01T03:00")
    location = EarthLocation.of_site("Roque de los Muchachos")
    altaz = AltAz(location=location, obstime=obstime)

    tel_pointing_alt = source.observation_blocks[run_id].subarray_pointing_lat
    tel_pointing_az = source.observation_blocks[run_id].subarray_pointing_lon

    focal_length = source.subarray.tel[tel_id].optics.equivalent_focal_length

    tel_pointing = SkyCoord(
        alt=tel_pointing_alt,
        az=tel_pointing_az,
        frame=altaz,
    )

    nominal_frame = NominalFrame(origin=tel_pointing)

    camera_frame = CameraFrame(
        telescope_pointing=tel_pointing,
        focal_length=focal_length,
    )

    star_cam = SkyCoord(
        x=star_cam_x * u.m,
        y=star_cam_y * u.m,
        frame=camera_frame,
    )

    star_nom_xy = star_cam.transform_to(nominal_frame)
    star_nom_x = star_nom_xy.fov_lon.to(u.rad).value
    star_nom_y = star_nom_xy.fov_lat.to(u.rad).value


    return star_nom_x, star_nom_y

def nominal_to_altaz(source, subarray, run_id, tel_id, star_nom_x, star_nom_y):
    obstime = Time("2013-11-01T03:00")
    location = EarthLocation.of_site("Roque de los Muchachos")

    altaz = AltAz(location=location, obstime=obstime)

    tel_pointing_alt = source.observation_blocks[run_id].subarray_pointing_lat
    tel_pointing_az = source.observation_blocks[run_id].subarray_pointing_lon

    focal_length = source.subarray.tel[tel_id].optics.equivalent_focal_length

    tel_pointing = SkyCoord(
        alt=tel_pointing_alt,
        az=tel_pointing_az,
        frame=altaz,
    )

    nominal_frame = NominalFrame(origin=tel_pointing)

    star_nom_xy = SkyCoord(
        fov_lon=star_nom_x,
        fov_lat=star_nom_y,
        frame=nominal_frame,
    )

    star_altaz = star_nom_xy.transform_to(altaz)
    star_alt = star_altaz.alt.to_value(u.rad)
    star_az = star_altaz.az.to_value(u.rad)

    star_az_2pi = star_az - 2.0 * np.pi
    if abs(star_az_2pi - 0.0) < abs(star_az - 0.0):
        star_az = star_az_2pi

    return star_alt, star_az


def plot_xing_reconstruction(
    ctapipe_output,
    source,
    list_telescope_type,
    run_id,
    event,
    list_tel_id,
    list_image_feature,
    star_alt,
    star_az,
    xing_alt,
    xing_az,
    xing_err,
    tag,
):
    event_id = event.index["event_id"]

    focal_length = source.subarray.tel[list_tel_id[0]].optics.equivalent_focal_length / u.m
    star_cam_x, star_cam_y = altaz_to_camxy(
        source,
        source.subarray,
        run_id,
        list_tel_id[0],
        star_alt * u.rad,
        star_az * u.rad,
    )
    star_cam_x = star_cam_x / focal_length * 180./np.pi
    star_cam_y = star_cam_y / focal_length * 180./np.pi

    xing_cam_x, xing_cam_y = altaz_to_camxy(
        source,
        source.subarray,
        run_id,
        list_tel_id[0],
        xing_alt * u.rad,
        xing_az * u.rad,
    )
    xing_cam_x = xing_cam_x / focal_length * 180./np.pi
    xing_cam_y = xing_cam_y / focal_length * 180./np.pi
    xing_cam_err = xing_err * 180./np.pi

    xmax = 4.0
    xmin = -4.0
    ymax = 4.0
    ymin = -4.0

    fig, ax = plt.subplots()
    figsize_x = 6.4
    figsize_y = 6.4
    fig.set_figheight(figsize_y)
    fig.set_figwidth(figsize_x)

    list_cen_x = []
    list_cen_y = []
    list_angle = []
    list_a = []
    list_b = []
    list_a_err = []
    list_b_err = []
    list_pix_x = []
    list_pix_y = []
    list_pix_w = []
    list_pix_s = []


    for img in range(0, len(list_tel_id)):

        tel_id = list_tel_id[img]
        geometry = source.subarray.tel[tel_id].camera.geometry
        focal_length = source.subarray.tel[tel_id].optics.equivalent_focal_length / u.m

        clean_image_1d = np.zeros_like(event.dl1.tel[tel_id].image)
        image_mask, sum_noise, avg_noise = dynamic_cleaning(geometry,event,tel_id)

        pix_width = float(geometry.pixel_width[0] / u.m)
        for pix in range(0, len(image_mask)):
            if image_mask[pix]:
                list_pix_w += [event.dl1.tel[tel_id].image[pix]]
                list_pix_x += [float(geometry.pix_x[pix] / u.m)/focal_length*180./np.pi]
                list_pix_y += [float(geometry.pix_y[pix] / u.m)/focal_length*180./np.pi]
                list_pix_s += [pix_width/focal_length*180./np.pi/2.]

        line_a = list_image_feature[img][8]
        line_b = list_image_feature[img][9] / focal_length * 180./np.pi
        line_a_err = list_image_feature[img][11]
        line_b_err = list_image_feature[img][12] / focal_length * 180./np.pi
        angle = list_image_feature[img][3]
        image_center_x = list_image_feature[img][1] / focal_length * 180./np.pi
        image_center_y = list_image_feature[img][2] / focal_length * 180./np.pi

        list_a += [line_a]
        list_b += [line_b]
        list_a_err += [line_a_err]
        list_b_err += [line_b_err]
        list_angle += [angle]
        list_cen_x += [image_center_x]
        list_cen_y += [image_center_y]

    for pix in range(0,len(list_pix_w)):
        #ax.scatter(list_pix_x[pix], list_pix_y[pix], s=list_pix_s[pix], alpha=0.2, c="g", marker="o")
        #ax.scatter(list_pix_x[pix], list_pix_y[pix], s=1, alpha=0.2, c="g", marker="o")
        mycircle = plt.Circle( (list_pix_x[pix], list_pix_y[pix]), list_pix_s[pix], fill=True, color='green', alpha=0.2)
        ax.add_patch(mycircle)

    for img in range(0, len(list_a)):
        line_a = list_a[img]
        line_b = list_b[img]
        line_a_err = list_a_err[img]
        line_b_err = list_b_err[img]
        angle = list_angle[img]
        image_center_x = list_cen_x[img]
        image_center_y = list_cen_y[img]
        if xing_cam_x > image_center_x:
            line_x = np.linspace(image_center_x, xmax, 100)
            line_y = line_a * line_x + line_b
            ax.plot(line_x, line_y, color="k", alpha=0.1, linestyle="dashed")
            line_yup = (
               (line_a + abs(line_a_err/np.cos(angle * u.rad))) * line_x + line_b + pow( pow(line_a_err * image_center_x,2) + pow(line_b_err,2),0.5)
            )
            line_ylow = (
               (line_a - abs(line_a_err/np.cos(angle * u.rad))) * line_x + line_b - pow( pow(line_a_err * image_center_x,2) + pow(line_b_err,2),0.5)
            )
            ax.fill_between(line_x,line_ylow,line_yup,alpha=0.05,color='b')
        else:
            line_x = np.linspace(xmin, image_center_x, 100)
            line_y = line_a * line_x + line_b
            ax.plot(line_x, line_y, color="k", alpha=0.1, linestyle="dashed")
            line_yup = (
               (line_a + abs(line_a_err/np.cos(angle * u.rad))) * line_x + line_b - pow( pow(line_a_err * image_center_x,2) + pow(line_b_err,2),0.5)
            )
            line_ylow = (
               (line_a - abs(line_a_err/np.cos(angle * u.rad))) * line_x + line_b + pow( pow(line_a_err * image_center_x,2) + pow(line_b_err,2),0.5)
            )
            ax.fill_between(line_x,line_ylow,line_yup,alpha=0.05,color='b')

    ax.scatter(star_cam_x, star_cam_y, s=90, facecolors="none", c="r", marker="+")
    mycircle = plt.Circle( (xing_cam_x, xing_cam_y), xing_cam_err, fill = False, color='blue')
    ax.add_patch(mycircle)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    label_x = "X [deg]"
    label_y = "Y [deg]"
    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)

    fig.savefig(
        f"{ctapipe_output}/output_plots/run{run_id}_evt{event_id}_xing_{tag}.png",
        bbox_inches="tight",
    )
    del fig
    del ax
    plt.close()


    list_tel_id = []
    for tel_idx in range(0, source.subarray.n_tels):
        tel_id = source.subarray.tel_ids[tel_idx]
        telescope_type = str(source.subarray.tel[tel_id])
        if not telescope_type in list_telescope_type:
            continue
        list_tel_id += [tel_id]
    new_subarray = source.subarray.select_subarray(list_tel_id)

    fig, ax = plt.subplots()
    figsize_x = 6.4
    figsize_y = 6.4
    fig.set_figheight(figsize_y)
    fig.set_figwidth(figsize_x)

    disp = ArrayDisplay(new_subarray)
    #disp.add_labels()

    fig.savefig(
        f"{ctapipe_output}/output_plots/run{run_id}_evt{event_id}_array_{tag}.png",
        bbox_inches="tight",
    )
    del fig
    del ax
    plt.close()


def plot_monotel_reconstruction(
    ctapipe_output,
    subarray,
    run_id,
    tel_id,
    event,
    image_feature_array,
    star_cam_x,
    star_cam_y,
    fit_cam_x,
    fit_cam_y,
    fit_cam_err,
    telescope_type,
    tag,
):
    event_id = event.index["event_id"]
    geometry = subarray.tel[tel_id].camera.geometry

    dirty_image_1d = event.dl1.tel[tel_id].image
    dirty_image_2d = geometry.image_to_cartesian_representation(dirty_image_1d)
    remove_nan_pixels(dirty_image_2d)

    clean_image_1d = np.zeros_like(event.dl1.tel[tel_id].image)
    clean_time_1d = np.zeros_like(event.dl1.tel[tel_id].peak_time)
    image_mask, sum_noise, avg_noise = dynamic_cleaning(geometry,event,tel_id)

    mask_size = 0
    for pix in range(0, len(image_mask)):
        if not image_mask[pix]:
            clean_image_1d[pix] = 0.0
            clean_time_1d[pix] = 0.0
        else:
            clean_image_1d[pix] = event.dl1.tel[tel_id].image[pix]
            clean_time_1d[pix] = event.dl1.tel[tel_id].peak_time[pix]
            mask_size += 1

    center_time = reset_time(clean_image_1d, clean_time_1d)

    clean_image_2d = geometry.image_to_cartesian_representation(clean_image_1d)
    remove_nan_pixels(clean_image_2d)
    clean_time_2d = geometry.image_to_cartesian_representation(clean_time_1d)
    remove_nan_pixels(clean_time_2d)

    mask_size = image_feature_array[0]
    image_center_x = image_feature_array[1]
    image_center_y = image_feature_array[2]
    angle = image_feature_array[3]
    semi_major = image_feature_array[4]
    semi_minor = image_feature_array[5]
    image_direction_err = image_feature_array[6]
    image_direction = image_feature_array[7]
    line_a = image_feature_array[8]
    line_b = image_feature_array[9]
    line_a_err = image_feature_array[11]
    line_b_err = image_feature_array[12]

    xmax = max(geometry.pix_x) / u.m
    xmin = min(geometry.pix_x) / u.m
    ymax = max(geometry.pix_y) / u.m
    ymin = min(geometry.pix_y) / u.m

    fig, ax = plt.subplots()
    figsize_x = 8.6
    figsize_y = 6.4
    fig.set_figheight(figsize_y)
    fig.set_figwidth(figsize_x)
    display = CameraDisplay(geometry, ax=ax)
    display.image = clean_image_1d
    display.cmap = "Reds"
    display.add_colorbar(ax=ax)
    ax.scatter(star_cam_x, star_cam_y, s=90, facecolors="none", c="r", marker="+")
    #ax.scatter(image_center_x, image_center_y, s=90, facecolors="none", edgecolors='g', marker="o")
    mycircle = plt.Circle( (fit_cam_x, fit_cam_y), fit_cam_err, fill = False, color='blue')
    ax.add_patch(mycircle)
    if np.cos(angle * u.rad) > 0.0:
        line_x = np.linspace(image_center_x, xmax, 100)
        line_y = line_a * line_x + line_b
        ax.plot(line_x, line_y, color="k", alpha=0.1, linestyle="dashed")
        line_yup = (
           (line_a + abs(line_a_err/np.cos(angle * u.rad))) * line_x + line_b + pow( pow(line_a_err * image_center_x,2) + pow(line_b_err,2),0.5)
        )
        line_ylow = (
           (line_a - abs(line_a_err/np.cos(angle * u.rad))) * line_x + line_b - pow( pow(line_a_err * image_center_x,2) + pow(line_b_err,2),0.5)
        )
        ax.fill_between(line_x,line_ylow,line_yup,alpha=0.05,color='b')
    else:
        line_x = np.linspace(xmin, image_center_x, 100)
        line_y = line_a * line_x + line_b
        ax.plot(line_x, line_y, color="k", alpha=0.1, linestyle="dashed")
        line_yup = (
           (line_a + abs(line_a_err/np.cos(angle * u.rad))) * line_x + line_b - pow( pow(line_a_err * image_center_x,2) + pow(line_b_err,2),0.5)
        )
        line_ylow = (
           (line_a - abs(line_a_err/np.cos(angle * u.rad))) * line_x + line_b + pow( pow(line_a_err * image_center_x,2) + pow(line_b_err,2),0.5)
        )
        ax.fill_between(line_x,line_ylow,line_yup,alpha=0.05,color='b')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    fig.savefig(
        f"{ctapipe_output}/output_plots/{telescope_type}_run{run_id}_evt{event_id}_tel{tel_id}_clean_image_{tag}.png",
        bbox_inches="tight",
    )
    del fig
    del ax
    plt.close()


def movie_simulation(
    telescope_type,
    subarray,
    run_id,
    tel_id,
    event,
    init_params,
    dict_movie_lookup_table,
    dict_movie_eigen_vectors,
    eco_image_1d,
):

    image_size = np.sum(eco_image_1d)
    image_idx = 0
    for idx in range(0,len(image_size_bins)-1):
        if image_size>=image_size_bins[idx] and image_size<image_size_bins[idx+1]:
            image_idx = idx

    movie_lookup_table = dict_movie_lookup_table[telescope_type]
    movie_eigen_vectors = dict_movie_eigen_vectors[telescope_type]

    event_id = event.index["event_id"]
    geometry = subarray.tel[tel_id].camera.geometry

    n_eco_pix = len(eco_image_1d)

    fit_arrival = init_params[0]
    fit_impact = init_params[1]
    fit_log_energy = init_params[2]

    fit_movie_latent_space = []
    key_idx = movie_lookup_table[image_idx][0].get_bin(fit_arrival, fit_impact, fit_log_energy)
    key_idx_x = key_idx[0]
    key_idx_y = key_idx[1]
    key_idx_z = key_idx[2]
    for r in range(0, len(movie_lookup_table[image_idx])):
        fit_movie_latent_space += [
            movie_lookup_table[image_idx][r].get_bin_content_by_index(
                key_idx_x, key_idx_y, key_idx_z,
            )
        ]
    fit_movie_latent_space = np.array(fit_movie_latent_space)

    eco_movie_1d_fit = movie_eigen_vectors[image_idx].T @ fit_movie_latent_space
    n_windows = int(select_samples / n_samples_per_window)
    sim_eco_image_1d = []
    sim_image_1d = []
    for win in range(0, n_windows):
        sim_eco_image_1d += [np.zeros_like(eco_image_1d)]
        sim_image_1d += [np.zeros_like(event.dl1.tel[tel_id].image)]

    sim_image_2d = []
    for win in range(0, n_windows):
        for pix in range(0, n_eco_pix):
            movie_pix_idx = pix + win * n_eco_pix
            sim_eco_image_1d[win][pix] = eco_movie_1d_fit[movie_pix_idx]
        image_cutout_restore(geometry, sim_eco_image_1d[win], sim_image_1d[win])
        sim_image_2d += [geometry.image_to_cartesian_representation(sim_image_1d[win])]

    return sim_image_2d

def display_a_movie(subarray, run_id, tel_id, event, eco_image_size, eco_movie_1d):
    n_windows = int(select_samples / n_samples_per_window)
    eco_image_1d = []
    for win in range(0, n_windows):
        eco_image_1d += [np.zeros(eco_image_size)]

    for win in range(0, n_windows):
        for pix in range(0, eco_image_size):
            entry = pix + win * eco_image_size
            eco_image_1d[win][pix] = eco_movie_1d[entry]

    event_id = event.index["event_id"]
    geometry = subarray.tel[tel_id].camera.geometry

    dirty_image_1d = event.dl1.tel[tel_id].image
    image_max = np.max(dirty_image_1d[:])
    image_1d = []
    for win in range(0, n_windows):
        image_1d += [np.zeros_like(dirty_image_1d)]

    list_image_2d = []
    for win in range(0, n_windows):
        image_cutout_restore(geometry, eco_image_1d[win], image_1d[win])
        image_2d = geometry.image_to_cartesian_representation(image_1d[win])
        list_image_2d += [image_2d]

    return dirty_image_1d, list_image_2d

def make_a_eigenmovie_gif(
    ctapipe_output,
    telescope_type,
    subarray,
    run_id,
    tel_id,
    event,
    dict_movie_lookup_table,
    dict_movie_eigen_vectors,
    eco_image_1d,
    eco_movie_1d,
):

    image_size = np.sum(eco_image_1d)
    image_idx = 0
    for idx in range(0,len(image_size_bins)-1):
        if image_size>=image_size_bins[idx] and image_size<image_size_bins[idx+1]:
            image_idx = idx

    movie_lookup_table = dict_movie_lookup_table[telescope_type]
    movie_eigen_vectors = dict_movie_eigen_vectors[telescope_type]

    event_id = event.index["event_id"]
    geometry = subarray.tel[tel_id].camera.geometry

    truth_movie_latent_space = movie_eigen_vectors[image_idx] @ eco_movie_1d
    eigen_movie_latent_space = []
    eigen_movie_latent_space += [np.zeros_like(truth_movie_latent_space)]
    eigen_movie_latent_space += [np.zeros_like(truth_movie_latent_space)]
    eigen_movie_latent_space += [np.zeros_like(truth_movie_latent_space)]
    eigen_movie_latent_space += [np.zeros_like(truth_movie_latent_space)]
    eigen_movie_latent_space[0][0] = 1.
    eigen_movie_latent_space[1][1] = 1.
    eigen_movie_latent_space[2][2] = 1.
    eigen_movie_latent_space[3][3] = 1.

    eco_eigen_movie_1d = []
    for mode in range(0,len(eigen_movie_latent_space)):
        eco_eigen_movie_1d += [movie_eigen_vectors[image_idx].T @ eigen_movie_latent_space[mode]]

    n_windows = int(select_samples / n_samples_per_window)
    eigen_eco_image_1d = []
    eigen_image_1d = []
    for mode in range(0,len(eigen_movie_latent_space)):
        sim_eco_image_1d = []
        sim_image_1d = []
        for win in range(0, n_windows):
            sim_eco_image_1d += [np.zeros_like(eco_image_1d)]
            sim_image_1d += [np.zeros_like(event.dl1.tel[tel_id].image)]
        eigen_eco_image_1d += [sim_eco_image_1d]
        eigen_image_1d += [sim_image_1d]

    n_eco_pix = len(eco_image_1d)
    eigen_movie_2d = []
    for mode in range(0,len(eigen_movie_latent_space)):
        sim_movie_2d = []
        for win in range(0, n_windows):
            for pix in range(0, n_eco_pix):
                movie_pix_idx = pix + win * n_eco_pix
                eigen_eco_image_1d[mode][win][pix] = eco_eigen_movie_1d[mode][movie_pix_idx]
            image_cutout_restore(geometry, eigen_eco_image_1d[mode][win], eigen_image_1d[mode][win])
            sim_movie_2d += [geometry.image_to_cartesian_representation(eigen_image_1d[mode][win])]
        eigen_movie_2d += [sim_movie_2d]

    xmax = max(geometry.pix_x) / u.m
    xmin = min(geometry.pix_x) / u.m
    ymax = max(geometry.pix_y) / u.m
    ymin = min(geometry.pix_y) / u.m

    movie_2d = []
    for m in range(0, len(eigen_movie_2d[0])):
        movie_2d += [np.vstack((eigen_movie_2d[3][m], eigen_movie_2d[2][m], eigen_movie_2d[1][m], eigen_movie_2d[0][m]))]

    fig, ax = plt.subplots()
    figsize_x = 8.6
    figsize_y = 6.4
    fig.set_figheight(figsize_y)
    fig.set_figwidth(figsize_x)
    #label_x = "X"
    #label_y = "Y"
    #ax.set_xlabel(label_x)
    #ax.set_ylabel(label_y)
    im = ax.imshow(
        movie_2d[0],
        origin="lower",
        extent=(xmin, xmax, ymin, ymax),
        cmap="coolwarm",
    )
    #plt.text(0.8*xmin, 0.8*ymax, 'input data', fontdict=font)
    #plt.text(0.8*xmin, 0.8*ymax-0.5*(ymax-ymin), 'reconstruction', fontdict=font)

    def animate(i):
        im.set_array(movie_2d[i])
        return (im,)

    ani = animation.FuncAnimation(
        fig, animate, repeat=True, frames=len(movie_2d), interval=1000
    )
    ani.save(
        f"{ctapipe_output}/output_plots/run{run_id}_evt{event_id}_tel{tel_id}_eigen_movie.gif",
        writer=animation.PillowWriter(fps=4),
    )
    del fig
    del ax
    del ani
    plt.close()



def make_a_gif(
    ctapipe_output, subarray, run_id, tel_id, event, eco_image_1d, movie1_2d, movie2_2d
):
    event_id = event.index["event_id"]
    geometry = subarray.tel[tel_id].camera.geometry

    xmax = max(geometry.pix_x) / u.m
    xmin = min(geometry.pix_x) / u.m
    ymax = max(geometry.pix_y) / u.m
    ymin = min(geometry.pix_y) / u.m

    movie_2d = []
    for m in range(0, len(movie1_2d)):
        movie_2d += [np.vstack((movie2_2d[m], movie1_2d[m]))]

    image_max = np.max(eco_image_1d[:])
    n_windows = len(movie_2d)

    fig, ax = plt.subplots()
    figsize_x = 8.6
    figsize_y = 6.4
    fig.set_figheight(figsize_y)
    fig.set_figwidth(figsize_x)
    label_x = "X"
    label_y = "Y"
    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)
    im = ax.imshow(
        movie_2d[0],
        origin="lower",
        extent=(xmin, xmax, ymin, ymax),
        vmin=0.0,
        vmax=2.0 * image_max / float(n_windows),
        cmap="Reds",
    )
    cbar = fig.colorbar(im)
    plt.text(0.8*xmin, 0.8*ymax, 'input data', fontdict=font)
    plt.text(0.8*xmin, 0.8*ymax-0.5*(ymax-ymin), 'reconstruction', fontdict=font)

    def animate(i):
        im.set_array(movie_2d[i])
        return (im,)

    ani = animation.FuncAnimation(
        fig, animate, repeat=True, frames=len(movie_2d), interval=1000
    )
    ani.save(
        f"{ctapipe_output}/output_plots/run{run_id}_evt{event_id}_tel{tel_id}_sim_movie.gif",
        writer=animation.PillowWriter(fps=4),
    )
    del fig
    del ax
    del ani
    plt.close()


    movie_2d = []
    for m in range(0, len(movie1_2d)):
        movie_2d += [np.vstack((movie1_2d[m]))]

    image_max = np.max(eco_image_1d[:])
    n_windows = len(movie_2d)

    fig, ax = plt.subplots()
    figsize_x = 8.6
    figsize_y = 6.4
    fig.set_figheight(figsize_y)
    fig.set_figwidth(figsize_x)
    label_x = "X"
    label_y = "Y"
    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)
    im = ax.imshow(
        movie_2d[0],
        origin="lower",
        extent=(xmin, xmax, ymin, ymax),
        vmin=0.0,
        vmax=2.0 * image_max / float(n_windows),
        cmap="Reds",
    )
    cbar = fig.colorbar(im)

    def animate(i):
        im.set_array(movie_2d[i])
        return (im,)

    ani = animation.FuncAnimation(
        fig, animate, repeat=True, frames=len(movie_2d), interval=1000
    )
    ani.save(
        f"{ctapipe_output}/output_plots/run{run_id}_evt{event_id}_tel{tel_id}_data_movie.gif",
        writer=animation.PillowWriter(fps=4),
    )
    del fig
    del ax
    del ani
    plt.close()

def plot_template_reconstruction(
    ctapipe_output,
    source,
    list_telescope_type,
    run_id,
    event,
    list_tel_id,
    star_alt,
    star_az,
    xing_alt,
    xing_az,
    xing_err,
):
    event_id = event.index["event_id"]
    if len(list_tel_id)==0: return

    focal_length = source.subarray.tel[list_tel_id[0]].optics.equivalent_focal_length / u.m
    star_cam_x, star_cam_y = altaz_to_camxy(
        source,
        source.subarray,
        run_id,
        list_tel_id[0],
        star_alt * u.rad,
        star_az * u.rad,
    )
    star_cam_x = star_cam_x / focal_length * 180./np.pi
    star_cam_y = star_cam_y / focal_length * 180./np.pi

    xing_cam_x, xing_cam_y = altaz_to_camxy(
        source,
        source.subarray,
        run_id,
        list_tel_id[0],
        xing_alt * u.rad,
        xing_az * u.rad,
    )
    xing_cam_x = xing_cam_x / focal_length * 180./np.pi
    xing_cam_y = xing_cam_y / focal_length * 180./np.pi
    xing_cam_err = xing_err * 180./np.pi

    xmax = 4.0
    xmin = -4.0
    ymax = 4.0
    ymin = -4.0

    fig, ax = plt.subplots()
    figsize_x = 6.4
    figsize_y = 6.4
    fig.set_figheight(figsize_y)
    fig.set_figwidth(figsize_x)

    list_pix_x = []
    list_pix_y = []
    list_pix_w = []
    list_pix_s = []


    for img in range(0, len(list_tel_id)):

        tel_id = list_tel_id[img]
        geometry = source.subarray.tel[tel_id].camera.geometry
        focal_length = source.subarray.tel[tel_id].optics.equivalent_focal_length / u.m

        clean_image_1d = np.zeros_like(event.dl1.tel[tel_id].image)
        image_mask, sum_noise, avg_noise = dynamic_cleaning(geometry,event,tel_id)

        pix_width = float(geometry.pixel_width[0] / u.m)
        for pix in range(0, len(image_mask)):
            if image_mask[pix]:
                list_pix_w += [event.dl1.tel[tel_id].image[pix]]
                list_pix_x += [float(geometry.pix_x[pix] / u.m)/focal_length*180./np.pi]
                list_pix_y += [float(geometry.pix_y[pix] / u.m)/focal_length*180./np.pi]
                list_pix_s += [pix_width/focal_length*180./np.pi/2.]

    for pix in range(0,len(list_pix_w)):
        mycircle = plt.Circle( (list_pix_x[pix], list_pix_y[pix]), list_pix_s[pix], fill=True, color='green', alpha=0.2)
        ax.add_patch(mycircle)

    ax.scatter(star_cam_x, star_cam_y, s=90, facecolors="none", c="r", marker="+")
    mycircle = plt.Circle( (xing_cam_x, xing_cam_y), xing_cam_err, fill = False, color='blue')
    ax.add_patch(mycircle)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    label_x = "X [deg]"
    label_y = "Y [deg]"
    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)

    fig.savefig(
        f"{ctapipe_output}/output_plots/run{run_id}_evt{event_id}_template.png",
        bbox_inches="tight",
    )
    del fig
    del ax
    plt.close()

def run_monoscopic_analysis(
    ctapipe_output,
    list_telescope_type,
    run_id,
    source,
    event,
    dict_image_to_physics_model,
    dict_movie_lookup_table,
    dict_movie_eigen_vectors,
    dict_image_lookup_table,
    dict_image_eigen_vectors,
    dict_time_lookup_table,
    dict_time_eigen_vectors,
    seed_alt,
    seed_az,
    seed_weight,
):
    analysis_result = []

    event_id = event.index["event_id"]
    ntel = len(event.r0.tel)

    truth_alt = float(event.simulation.shower.alt / u.rad)
    truth_az = float(event.simulation.shower.az / u.rad)
    truth_energy = event.simulation.shower.energy / u.TeV
    truth_log_energy = np.log10(truth_energy)

    list_tel_alt = []
    list_tel_az = []
    list_tel_log_energy = []
    list_tel_weight = []
    list_tel_off_angle = []
    list_tel_unc = []
    #list_tel_chi2 = []

    list_tel_id = []
    list_image_feature = []

    seed_err = 1e10
    if seed_weight > 0.0:
        seed_err = 1.0 / (pow(seed_weight, 0.5))
    use_seed = False
    if seed_err < 0.3 * np.pi / 180.0:
        use_seed = True
    print(f"use_seed = {use_seed}")

    ref_tel_id = 0
    for tel_idx in range(0, len(list(event.dl0.tel.keys()))):
        tel_id = list(event.dl0.tel.keys())[tel_idx]

        if not use_template:
            continue
   
        telescope_type = str(source.subarray.tel[tel_id])
        if not telescope_type in list_telescope_type:
            continue
        ref_tel_id = tel_id

        seed_cam_x, seed_cam_y = altaz_to_camxy(
            source,
            source.subarray,
            run_id,
            tel_id,
            seed_alt * u.rad,
            seed_az * u.rad,
        )
        seed_camxy = None
        if use_seed:
            seed_camxy = [seed_cam_x, seed_cam_y]

        truth_info_array = find_image_truth(
            source, source.subarray, run_id, tel_id, event
        )
        truth_energy = float(truth_info_array[0] / u.TeV)
        truth_core_x = truth_info_array[1]
        truth_core_y = truth_info_array[2]
        # truth_alt = float(truth_info_array[3]/u.rad)
        # truth_az = float(truth_info_array[4]/u.rad)
        truth_height = truth_info_array[5]
        truth_xmax = truth_info_array[6]
        star_cam_x = truth_info_array[7]
        star_cam_y = truth_info_array[8]
        impact_x = truth_info_array[9]
        impact_y = truth_info_array[10]
        focal_length = source.subarray.tel[tel_id].optics.equivalent_focal_length / u.m

        tic_standard = time.perf_counter()
        (
            is_edge_image,
            image_feature_array,
            eco_movie_1d,
            eco_image_1d,
            eco_time_1d,
        ) = make_standard_movie(
            ctapipe_output,
            telescope_type,
            source.subarray,
            run_id,
            tel_id,
            event,
            flip=False,
            star_cam_xy=seed_camxy,
        )
        toc_standard = time.perf_counter()
        #print(f"making standard movie time: {toc_standard-tic_standard:0.1f} sec")

        print (f"tel_id = {tel_id}")

        mask_size = image_feature_array[0]
        image_center_x = image_feature_array[1]
        image_center_y = image_feature_array[2]
        angle = image_feature_array[3]
        image_semi_major = image_feature_array[4]
        image_semi_minor = image_feature_array[5]
        image_direction_err = image_feature_array[6]
        image_direction = image_feature_array[7]
        line_a = image_feature_array[8]
        line_b = image_feature_array[9]
        truth_projection = image_feature_array[10]
        line_a_err = image_feature_array[11]
        line_b_err = image_feature_array[12]
        angle_err = image_feature_array[13]
        image_size = image_feature_array[14]
        frac_leakage_intensity = image_feature_array[15]
        sum_noise = image_feature_array[16]

        if mask_size < mask_size_cut:
            #print (f"mask_size = {mask_size}, failed mask_size cut")
            continue
        if image_size < image_size_cut_analysis:
            #print (f"image_size = {image_size:0.1f}, failed image_size cut")
            continue
        if image_size > image_size_bins[len(image_size_bins)-1]:
            #print (f"image_size = {image_size:0.1f}, failed image_size cut")
            continue
        if is_edge_image:
            #print (f"failed is_edge cut")
            continue
        if image_direction_err==0.:
            continue
        if frac_leakage_intensity>frac_leakage_intensity_cut_analysis:
            continue

        image_direction_significance = abs(image_direction)/image_direction_err

        pixel_width = float(source.subarray.tel[tel_id].camera.geometry.pixel_width[0] / u.m)
        tic_reco = time.perf_counter()
        (
            image_fit_arrival,
            image_fit_impact,
            image_fit_log_energy,
            image_fit_arrival_err,
            image_fit_imapct_err,
            image_fit_log_energy_err,
        ) = single_movie_reconstruction(
            telescope_type,
            pixel_width,
            angle_err,
            image_size,
            eco_image_1d,
            dict_image_lookup_table,
            dict_image_eigen_vectors,
            eco_time_1d,
            dict_time_lookup_table,
            dict_time_eigen_vectors,
            eco_movie_1d,
            dict_movie_lookup_table,
            dict_movie_eigen_vectors,
            dict_image_to_physics_model,
        )
        toc_reco = time.perf_counter()
        print(f"single template reco time: {toc_reco-tic_reco:0.1f} sec")
        cam_fov = focal_length * 10.*np.pi/180.
        print (f"cam_fov = {cam_fov}")

        truth_impact = pow(pow(impact_x,2)+pow(impact_y,2),0.5)
        truth_arrival = pow(pow(star_cam_x-image_center_x,2)+pow(star_cam_y-image_center_y,2),0.5)
        print (f"truth_energy = {truth_energy:0.3f} TeV")
        print (f"fit_energy = {pow(10.,image_fit_log_energy):0.3f} TeV")
        print (f"truth_arrival = {truth_arrival:0.4f}")
        print (f"fit_arrival = {image_fit_arrival:0.4f}")
        print (f"truth_impact = {truth_impact:0.4f}")
        print (f"fit_impact = {image_fit_impact:0.4f}")

        #image_idx = 0
        #for idx in range(0,len(image_size_bins)-1):
        #    if image_size>=image_size_bins[idx] and image_size<image_size_bins[idx+1]:
        #        image_idx = idx
        #movie_lookup_table = dict_movie_lookup_table[telescope_type][image_idx]
        #movie_eigen_vectors = dict_movie_eigen_vectors[telescope_type][image_idx]
        #fit_params = [image_fit_arrival, image_fit_impact, image_fit_log_energy]
        #rel_chi2 = sqaure_difference_between_1d_images_poisson(
        #    fit_params, eco_movie_1d, movie_lookup_table, movie_eigen_vectors, full_table=True
        #)
        #print (f"rel_chi2 = {rel_chi2}")

        line_a = image_feature_array[8]
        line_b = image_feature_array[9]
        line_a_err = image_feature_array[11]
        line_b_err = image_feature_array[12]

        image_fit_cam_x = image_center_x + image_fit_arrival * np.cos(angle * u.rad)
        image_fit_cam_y = image_center_y + image_fit_arrival * np.sin(angle * u.rad)
        image_fit_nom_x, image_fit_nom_y = camxy_to_nominal(
            source, source.subarray, run_id, tel_id, image_fit_cam_x, 1.*image_fit_cam_y
        )
        image_center_nom_x, image_center_nom_y = camxy_to_nominal(
            source, source.subarray, run_id, tel_id, image_center_x, 1.*image_center_y
        )

        line_nom_a = (image_fit_nom_y - image_center_nom_y) / (
            image_fit_nom_x - image_center_nom_x
        )
        line_nom_b = image_center_nom_y - line_nom_a * image_center_nom_x

        seed_arrival = pow(
            pow(seed_cam_x - image_center_x, 2) + pow(seed_cam_y - image_center_y, 2),
            0.5,
        )
        image_method_unc = (
            pow(
                pow(image_fit_arrival_err, 2),
                0.5,
            )
            / focal_length
        )

        image_method_error = (
            pow(
                pow(image_fit_cam_x - star_cam_x, 2)
                + pow(image_fit_cam_y - star_cam_y, 2),
                0.5,
            )
            / focal_length
        )

        separation_from_seed = (
            pow(
                pow(image_fit_cam_x - seed_cam_x, 2)
                + pow(image_fit_cam_y - seed_cam_y, 2),
                0.5,
            )
            / focal_length
        )
        if image_fit_arrival>cam_fov: continue
        if image_direction_significance<5. and not use_seed: continue
        #if image_method_unc>0.3*np.pi/180.: continue
        #if rel_chi2>0.2: continue
        #if log_likelihood<49. or rel_chi2<0.2: continue
        #if separation_from_seed>3.*seed_err: continue

        # print(f"truth_energy     = {truth_energy}")
        # print(f"image_fit_energy = {pow(10.,image_fit_log_energy)}")
        # print(f"truth_fit_impact = {pow(impact_x*impact_x+impact_y*impact_y,0.5)}")
        # print(f"image_fit_impact = {image_fit_impact}")
        # print(f"image_method_error = {image_method_error:0.3f} deg")
        # print(f"image_method_unc = {image_method_unc:0.3f} deg")


        image_fit_alt = 0.
        image_fit_az = 0.
        if abs(image_fit_nom_x)<np.pi/2. and abs(image_fit_nom_y)<np.pi/2.:
            image_fit_alt, image_fit_az = nominal_to_altaz(
                source,
                source.subarray,
                run_id,
                tel_id,
                image_fit_nom_x * u.rad,
                image_fit_nom_y * u.rad,
            )

        list_tel_alt += [image_fit_alt]
        list_tel_az += [image_fit_az]
        list_tel_log_energy += [image_fit_log_energy]
        leakage_weight = 1.
        if frac_leakage_intensity>frac_leakage_intensity_cut_analysis:
            leakage_weight = 0.
        list_tel_weight += [pow(1.0 / image_method_unc, 2)]
        list_tel_off_angle += [image_method_error]
        list_tel_unc += [image_method_unc]
        #list_tel_chi2 += [rel_chi2]
        print(
                f"mask_size = {mask_size}, image_size = {image_size:0.1f}, image_direction_significance = {image_direction_significance:0.1f}, image_method_unc = {image_method_unc*180./np.pi:0.3f}, image_method_off_angle = {image_method_error*180./np.pi:0.3f}"
        )

        list_tel_id += [tel_id]
        list_image_feature += [image_feature_array]

        is_bad_result = False
        if image_size > image_size_cut_analysis and image_method_error / image_method_unc > 4.0:
            is_bad_result = True

        make_a_plot = False
        if select_run_id != 0:
            make_a_plot = True
        elif run_diagnosis:
            make_a_plot = True
        else:
            if image_size>plot_image_size_cut_lower:
                make_a_plot = True

        if make_a_plot or is_bad_result:

            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(f"mask_size = {mask_size}")
            print(f"image_method_unc = {image_method_unc*180./np.pi:0.3f} deg")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            plot_monotel_reconstruction(
                ctapipe_output,
                source.subarray,
                run_id,
                tel_id,
                event,
                image_feature_array,
                star_cam_x,
                star_cam_y,
                image_fit_cam_x,
                image_fit_cam_y,
                image_method_unc*focal_length,
                telescope_type,
                "movie",
            )
            fit_params = [image_fit_arrival, image_fit_impact, image_fit_log_energy]
            sim_movie = movie_simulation(
                telescope_type,
                source.subarray,
                run_id,
                tel_id,
                event,
                fit_params,
                dict_movie_lookup_table,
                dict_movie_eigen_vectors,
                eco_image_1d,
            )
            data_image, data_movie = display_a_movie(
                source.subarray, run_id, tel_id, event, len(eco_image_1d), eco_movie_1d
            )
            make_a_gif(
                ctapipe_output,
                source.subarray,
                run_id,
                tel_id,
                event,
                data_image,
                data_movie,
                sim_movie,
            )
            make_a_eigenmovie_gif(
                ctapipe_output,
                telescope_type,
                source.subarray,
                run_id,
                tel_id,
                event,
                dict_movie_lookup_table,
                dict_movie_eigen_vectors,
                eco_image_1d,
                eco_movie_1d,
            )
        if run_diagnosis and is_bad_result:
            exit()

    avg_tmp_alt = 0.0
    avg_tmp_az = 0.0
    avg_tmp_log_energy = 0.0
    avg_tmp_weight = 0.0
    avg_tmp_err = 0.
    new_list_tel_alt = []
    new_list_tel_az = []
    new_list_tel_log_energy = []
    new_list_tel_weight = []
    new_list_tel_off_angle = []
    new_list_tel_unc = []
    #new_list_tel_chi2 = []
    if len(list_tel_alt) > 0:
        tmp_weight = 0.0
        for tel in range(0, len(list_tel_alt)):
            evt_alt = list_tel_alt[tel]
            evt_az = list_tel_az[tel]
            evt_log_energy = list_tel_log_energy[tel]

            evt_az_2pi = evt_az - 2.0 * np.pi
            if abs(evt_az_2pi - 0.0) < abs(evt_az - 0.0):
                evt_az = evt_az_2pi

            # TS = pow((evt_alt-init_tmp_alt)/rms_tmp_alt,2) + pow((evt_az-init_tmp_az)/rms_tmp_az,2)
            # if TS>1.: continue

            avg_tmp_alt += evt_alt * list_tel_weight[tel]
            avg_tmp_az += evt_az * list_tel_weight[tel]
            avg_tmp_log_energy += evt_log_energy * list_tel_weight[tel]
            avg_tmp_err += 1.
            tmp_weight += list_tel_weight[tel]

            new_list_tel_alt += [evt_alt]
            new_list_tel_az += [evt_az]
            new_list_tel_log_energy += [evt_log_energy]
            new_list_tel_weight += [list_tel_weight[tel]]
            new_list_tel_off_angle += [list_tel_off_angle[tel]]
            new_list_tel_unc += [list_tel_unc[tel]]
            #new_list_tel_chi2 += [list_tel_chi2[tel]]

        avg_tmp_alt = avg_tmp_alt / tmp_weight
        avg_tmp_az = avg_tmp_az / tmp_weight
        avg_tmp_log_energy = avg_tmp_log_energy / tmp_weight
        avg_tmp_err = pow(avg_tmp_err / tmp_weight, 0.5)
        avg_tmp_weight = 1.0 / (avg_tmp_err * avg_tmp_err)

    make_a_plot = False
    if select_run_id != 0:
        make_a_plot = True
    elif run_diagnosis:
        make_a_plot = True
    if make_a_plot:
        plot_template_reconstruction(
            ctapipe_output,
            source,
            list_telescope_type,
            run_id,
            event,
            list_tel_id,
            truth_alt,
            truth_az,
            avg_tmp_alt,
            avg_tmp_az,
            avg_tmp_err,
        )

    list_tel_ana = []
    list_tel_ana += [new_list_tel_log_energy]
    list_tel_ana += [new_list_tel_alt]
    list_tel_ana += [new_list_tel_az]
    list_tel_ana += [new_list_tel_off_angle]
    list_tel_ana += [new_list_tel_unc]
    #list_tel_ana += [new_list_tel_chi2]

    return (
        truth_log_energy,
        avg_tmp_log_energy,
        avg_tmp_alt,
        avg_tmp_az,
        avg_tmp_err,
        list_tel_ana,
    )



def run_multiscopic_analysis(ctapipe_output, list_telescope_type, run_id, source, event, make_a_plot=False):
    event_id = event.index["event_id"]
    ntel = len(event.r0.tel)

    truth_alt = float(event.simulation.shower.alt / u.rad)
    truth_az = float(event.simulation.shower.az / u.rad)

    list_line_x = []
    list_line_y = []
    list_line_angle_err = []
    list_line_intensity = []
    list_line_length = []
    list_line_width = []
    list_line_a = []
    list_line_b = []
    list_line_a_err = []
    list_line_b_err = []
    list_line_w = []
    list_tel_id = []
    list_image_feature = []
    list_pixel_width = []
    list_frac_leakage_intensity = []
    list_sum_noise = []
    list_npix = []

    for tel_idx in range(0, len(list(event.dl0.tel.keys()))):
        tel_id = list(event.dl0.tel.keys())[tel_idx]

        telescope_type = str(source.subarray.tel[tel_id])
        if not telescope_type in list_telescope_type:
            continue

        pixel_width = float(source.subarray.tel[tel_id].camera.geometry.pixel_width[0] / u.m)

        truth_info_array = find_image_truth(
            source, source.subarray, run_id, tel_id, event
        )
        truth_energy = float(truth_info_array[0] / u.TeV)
        truth_core_x = truth_info_array[1]
        truth_core_y = truth_info_array[2]
        # truth_alt = float(truth_info_array[3]/u.rad)
        # truth_az = float(truth_info_array[4]/u.rad)
        truth_height = truth_info_array[5]
        truth_xmax = truth_info_array[6]
        star_cam_x = truth_info_array[7]
        star_cam_y = truth_info_array[8]
        impact_x = truth_info_array[9]
        impact_y = truth_info_array[10]
        focal_length = source.subarray.tel[tel_id].optics.equivalent_focal_length / u.m

        (
            is_edge_image,
            image_feature_array,
            eco_movie_1d,
            eco_image_1d,
            eco_time_1d,
        ) = make_standard_movie(
            ctapipe_output,
            telescope_type,
            source.subarray,
            run_id,
            tel_id,
            event,
            flip=False,
            reposition=False,
        )

        mask_size = image_feature_array[0]
        image_size = image_feature_array[14]
        if mask_size < mask_size_cut:
            #print (f"mask_size = {mask_size}, failed mask_size cut")
            continue
        #if image_size < image_size_cut:
        #    print (f"image_size = {image_size:0.1f}, failed image_size cut")
        #    continue

        if make_a_plot and image_size>plot_image_size_cut_lower and image_size<plot_image_size_cut_upper:
            plot_monotel_reconstruction(
                ctapipe_output,
                source.subarray,
                run_id,
                tel_id,
                event,
                image_feature_array,
                star_cam_x,
                star_cam_y,
                0.,
                0.,
                100.,
                telescope_type,
                "mono",
            )

        list_tel_id += [tel_id]
        list_image_feature += [image_feature_array]

        image_center_x = image_feature_array[1]
        image_center_y = image_feature_array[2]
        angle = image_feature_array[3]
        length = image_feature_array[4]
        width = image_feature_array[5]
        angle_err = image_feature_array[13]
        image_size = image_feature_array[14]
        frac_leakage_intensity = image_feature_array[15]
        sum_noise = image_feature_array[16]
        n_pix = image_feature_array[17]

        line_a = image_feature_array[8]
        line_b = image_feature_array[9]
        line_a_err = image_feature_array[11]
        line_b_err = image_feature_array[12]

        image_center_nom_x, image_center_nom_y = camxy_to_nominal(
            source, source.subarray, run_id, tel_id, image_center_x, 1.*image_center_y
        )

        image_head_nom_x, image_head_nom_y = camxy_to_nominal(
            source,
            source.subarray,
            run_id,
            tel_id,
            image_center_x - 0.01,
            1.*(line_a * (image_center_x - 0.01) + line_b),
        )

        image_tail_nom_x, image_tail_nom_y = camxy_to_nominal(
            source,
            source.subarray,
            run_id,
            tel_id,
            image_center_x + 0.01,
            1.*(line_a * (image_center_x + 0.01) + line_b),
        )

        line_nom_a = (image_tail_nom_y - image_head_nom_y) / (
            image_tail_nom_x - image_head_nom_x
        )
        line_nom_b = image_center_nom_y - line_nom_a * image_center_nom_x
        line_nom_a_err = line_a_err
        line_nom_b_err = line_b_err / focal_length

        list_line_x += [image_center_nom_x]
        list_line_y += [image_center_nom_y]
        list_line_a += [line_nom_a]
        list_line_b += [line_nom_b]
        list_line_a_err += [line_nom_a_err]
        list_line_b_err += [line_nom_b_err]
        list_line_intensity += [image_size]
        list_line_angle_err += [angle_err]
        list_line_length += [length/focal_length]
        list_line_width += [width/focal_length]
        list_pixel_width += [pixel_width/focal_length]
        list_frac_leakage_intensity += [frac_leakage_intensity]
        list_sum_noise += [sum_noise]
        list_npix += [n_pix]

        #list_line_x += [image_center_x]
        #list_line_y += [image_center_y]
        #list_line_a += [line_a]
        #list_line_b += [line_b]
        #list_line_a_err += [line_a_err]
        #list_line_b_err += [line_b_err]
        #list_line_intensity += [image_size]
        #list_line_angle_err += [angle_err]
        #list_line_length += [length]
        #list_line_width += [width]
        #list_pixel_width += [pixel_width]

    xing_alt = 0.0
    xing_az = 0.0
    xing_weight = 0.0
    n_tel = len(list_line_a)
    if len(list_line_a) < 2:
        print (f'Less than 2 images for xing method.')
    else:
        (
            xing_nom_x,
            xing_nom_y,
            xing_err,
        ) = find_intersection_multiple_lines(
            list_tel_id,
            list_line_x,
            list_line_y,
            list_line_a,
            list_line_b,
            list_line_a_err,
            list_line_b_err,
            list_line_intensity,
            list_line_angle_err,
            list_line_length,
            list_line_width,
            list_pixel_width,
            list_frac_leakage_intensity,
            list_sum_noise,
            list_npix,
        )
        print (f"xing_nom_x = {xing_nom_x}, xing_nom_y = {xing_nom_y}")
        if abs(xing_nom_x)<np.pi/2. and abs(xing_nom_y)<np.pi/2.:
            xing_alt, xing_az = nominal_to_altaz(
                source,
                source.subarray,
                run_id,
                list_tel_id[0],
                xing_nom_x * u.rad,
                xing_nom_y * u.rad,
            )
            print (f"xing_alt = {xing_alt}, xing_az = {xing_az}")

        xing_weight = 1.0 / (xing_err * xing_err)

        xing_off_angle = (
            angular_separation(
                truth_az * u.rad, truth_alt * u.rad, xing_az * u.rad, xing_alt * u.rad
            )
            .to(u.rad)
            .value
        )

        is_bad_result = False
        if xing_off_angle * 180.0 / np.pi > 0.5 and xing_err*180./np.pi<0.3 and len(list_tel_id) >= 2:
            is_bad_result = True

        #if xing_off_angle * 180.0 / np.pi <0.05 and xing_err*180./np.pi>1.0:
        #    make_a_plot = True

        if make_a_plot or is_bad_result:
            print("plot xing reconstruction.")
            print (f"xing_off_angle = {xing_off_angle*180./np.pi} deg")
            print (f"xing_err = {xing_err*180./np.pi} deg")
            print (f"truth_alt = {truth_alt:0.3f}, xing_alt = {xing_alt:0.3f}")
            print (f"truth_az = {truth_az:0.3f}, xing_az = {xing_az:0.3f}")
            plot_xing_reconstruction(
                ctapipe_output,
                source,
                list_telescope_type,
                run_id,
                event,
                list_tel_id,
                list_image_feature,
                truth_alt,
                truth_az,
                xing_alt,
                xing_az,
                xing_err,
                "xing",
            )

    return xing_alt, xing_az, xing_weight, n_tel

def event_selection(event,ana_tag):

    n_good_images = 0
    for tel in range(0,len(event.dl1.tel)):
        if not event.dl1.tel[tel].is_valid:
            continue
        #print (f"event.dl1.tel[{tel}] = {event.dl1.tel[tel]}")
        #print (f"event.dl1.tel[{tel}].parameters = {event.dl1.tel[tel].parameters}")
        #exit()
        if event.dl1.tel[tel].parameters['hillas']['intensity']<50.:
            continue
        fov_lat_deg = event.dl1.tel[tel].parameters['hillas']['fov_lat'].to(u.deg).value
        fov_lon_deg = event.dl1.tel[tel].parameters['hillas']['fov_lon'].to(u.deg).value
        fov_rad_deg = pow(fov_lat_deg*fov_lat_deg+fov_lon_deg*fov_lon_deg,0.5)
        if 'freepact' in ana_tag:
            if fov_rad_deg>3.0:
                continue
        n_good_images += 1

    print (f"n_good_images = {n_good_images}")
    if n_good_images<2: 
        return False
    return True

def loop_all_events(ana_tag,training_sample_path, ctapipe_output, list_telescope_type, select_evt=None, save_output=True):
    analysis_result = []
    lookup_table_type = "box3d"

    global run_diagnosis
    global plot_image_size_cut_lower
    global plot_image_size_cut_upper
    global truth_energy_cut_lower
    global truth_energy_cut_upper
    global select_run_id
    global select_event_id
    global weighting_mode
    global use_template

    if not save_output:
        run_diagnosis = True
        #plot_image_size_cut_lower = 2000.
        plot_image_size_cut_lower = 2.*image_size_cut_analysis
        plot_image_size_cut_upper = 1e10
        truth_energy_cut_lower = 0.1
        truth_energy_cut_upper = 1e10
        n_tel_min = 1
        n_tel_max = 10000
        if not select_evt==None:
            select_run_id = select_evt[0]
            select_event_id = select_evt[1]

    if 'yes' in ana_tag:
        use_template = True
    else:
        use_template = False

    if 'zeroth' in ana_tag:
        weighting_mode = 0
    if 'first' in ana_tag:
        weighting_mode = 1
    if 'second' in ana_tag:
        weighting_mode = 2

    print("loading svd pickle data... ")

    dict_movie_eigen_vectors = {}
    dict_image_eigen_vectors = {}
    dict_time_eigen_vectors = {}
    dict_movie_lookup_table = {}
    dict_image_lookup_table = {}
    dict_time_lookup_table = {}
    dict_image_to_physics_model = {}
    for telescope_type in list_telescope_type:

        output_filename = f"{ctapipe_output}/output_machines/movie_fast_conversion_model_{telescope_type}.pkl"
        if not os.path.exists(output_filename):
            print (f"{output_filename} does not exist.")
            dict_image_to_physics_model[f'{telescope_type}'] = None
        else:
            image_to_physics_model_pkl = pickle.load(open(output_filename, "rb"))
            dict_image_to_physics_model[f'{telescope_type}'] = image_to_physics_model_pkl

        output_filename = f"{ctapipe_output}/output_machines/movie_{lookup_table_type}_lookup_table_{telescope_type}.pkl"
        if not os.path.exists(output_filename):
            print (f"{output_filename} does not exist.")
            dict_movie_lookup_table[f'{telescope_type}'] = None
        else:
            movie_lookup_table_pkl = pickle.load(open(output_filename, "rb"))
            dict_movie_lookup_table[f'{telescope_type}'] = movie_lookup_table_pkl
        output_filename = (
            f"{ctapipe_output}/output_machines/movie_eigen_vectors_{telescope_type}.pkl"
        )
        if not os.path.exists(output_filename):
            print (f"{output_filename} does not exist.")
            dict_movie_eigen_vectors[f'{telescope_type}'] = None
        else:
            movie_eigen_vectors_pkl = pickle.load(open(output_filename, "rb"))
            dict_movie_eigen_vectors[f'{telescope_type}'] = movie_eigen_vectors_pkl

        output_filename = f"{ctapipe_output}/output_machines/image_{lookup_table_type}_lookup_table_{telescope_type}.pkl"
        if not os.path.exists(output_filename):
            print (f"{output_filename} does not exist.")
            dict_image_lookup_table[f'{telescope_type}'] = None
        else:
            image_lookup_table_pkl = pickle.load(open(output_filename, "rb"))
            dict_image_lookup_table[f'{telescope_type}'] = image_lookup_table_pkl
        output_filename = (
            f"{ctapipe_output}/output_machines/image_eigen_vectors_{telescope_type}.pkl"
        )
        if not os.path.exists(output_filename):
            print (f"{output_filename} does not exist.")
            dict_image_eigen_vectors[f'{telescope_type}'] = None
        else:
            image_eigen_vectors_pkl = pickle.load(open(output_filename, "rb"))
            dict_image_eigen_vectors[f'{telescope_type}'] = image_eigen_vectors_pkl

        output_filename = f"{ctapipe_output}/output_machines/time_{lookup_table_type}_lookup_table_{telescope_type}.pkl"
        if not os.path.exists(output_filename):
            print (f"{output_filename} does not exist.")
            dict_time_lookup_table[f'{telescope_type}'] = None
        else:
            time_lookup_table_pkl = pickle.load(open(output_filename, "rb"))
            dict_time_lookup_table[f'{telescope_type}'] = time_lookup_table_pkl
        output_filename = (
            f"{ctapipe_output}/output_machines/time_eigen_vectors_{telescope_type}.pkl"
        )
        if not os.path.exists(output_filename):
            print (f"{output_filename} does not exist.")
            dict_time_eigen_vectors[f'{telescope_type}'] = None
        else:
            time_eigen_vectors_pkl = pickle.load(open(output_filename, "rb"))
            dict_time_eigen_vectors[f'{telescope_type}'] = time_eigen_vectors_pkl

    print(f"loading file: {training_sample_path}")
    source = SimTelEventSource(training_sample_path, focal_length_choice="EQUIVALENT")

    list_tel_id = []
    for tel_idx in range(0, source.subarray.n_tels):
        tel_id = source.subarray.tel_ids[tel_idx]
        telescope_type = str(source.subarray.tel[tel_id])
        if not telescope_type in list_telescope_type:
            continue
        list_tel_id += [tel_id]
    new_subarray = source.subarray.select_subarray(list_tel_id)

    # Explore the instrument description
    subarray_table = new_subarray.to_table()
    nlines = len(subarray_table) + 10
    subarray_table.pprint(nlines)
    #print(new_subarray.to_table())

    calib = CameraCalibrator(subarray=source.subarray)
    image_processor = ImageProcessor(subarray=source.subarray)
    shower_processor = ShowerProcessor(subarray=source.subarray)

    ob_keys = source.observation_blocks.keys()
    run_id = list(ob_keys)[0]

    tel_pointing_alt = float(
        source.observation_blocks[run_id].subarray_pointing_lat / u.rad
    )
    tel_pointing_az = float(
        source.observation_blocks[run_id].subarray_pointing_lon / u.rad
    )
    print(f"tel_pointing_alt = {tel_pointing_alt}")
    print(f"tel_pointing_az = {tel_pointing_az}")

    sum_combine_result = []
    sum_hillas_result = []
    sum_xing_result = []
    sum_template_result = []
    sum_truth_result = []

    hillas_score = 0
    xing_score = 0
    n_truth_events = 0

    for event in source:

        if select_run_id != 0:
            if run_id > select_run_id:
                exit()
            if run_id != select_run_id:
                continue

        print(
            "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
        )

        event_id = event.index["event_id"]
        if select_event_id != 0:
            if event_id != select_event_id:
                continue

        print(f"run_id = {run_id}")
        print(f"event_id = {event_id}")

        truth_alt = float(event.simulation.shower.alt / u.rad)
        truth_az = float(event.simulation.shower.az / u.rad)
        truth_energy = event.simulation.shower.energy / u.TeV
        print (f"truth_alt = {truth_alt}, truth_az = {truth_az}, truth_energy = {truth_energy:0.3f} TeV")

        if run_diagnosis:
            if truth_energy < truth_energy_cut_lower: continue
            if truth_energy > truth_energy_cut_upper: continue

        calib(event)  # fills in r1, dl0, and dl1
        image_processor(event)
        shower_processor(event)

        sum_truth_result += [
            [
                truth_energy,
                truth_alt * 180.0 / np.pi,
                truth_az * 180.0 / np.pi,
            ]
        ]
        n_truth_events += 1

        is_good_event = event_selection(event,ana_tag)

        reco_result = event.dl2.stereo.geometry["HillasReconstructor"]
        n_tels = len(reco_result.telescopes)
        print (f"n_tels = {n_tels}")
        if n_tels == 0:
            continue

        average_intensity = reco_result.average_intensity
        if run_diagnosis:
            if average_intensity<plot_image_size_cut_lower: continue
            if average_intensity>plot_image_size_cut_upper: continue
        if run_diagnosis:
            if n_tels>n_tel_max: continue
            if n_tels<n_tel_min: continue

        hillas_alt = 0.0
        hillas_az = 0.0
        hillas_err = 0.0
        hillas_weight = 0.0
        hillas_off_angle = 1e10
        if reco_result.is_valid and is_good_event:
            hillas_alt = reco_result.alt.to(u.rad).value
            hillas_az = reco_result.az.to(u.rad).value
            hillas_az_2pi = hillas_az - 2.0 * np.pi
            if abs(hillas_az_2pi - 0.0) < abs(hillas_az - 0.0):
                hillas_az = hillas_az_2pi
            hillas_alt_err = reco_result.alt_uncert.to(u.rad).value
            hillas_az_err = reco_result.az_uncert.to(u.rad).value
            hillas_err = pow(
                hillas_alt_err * hillas_alt_err + hillas_az_err * hillas_az_err, 0.5
            )
            if hillas_err > 0.01 * np.pi / 180.0:
                hillas_weight = 1.0 / (hillas_err * hillas_err)
            hillas_off_angle = angular_separation(
                truth_az * u.rad, truth_alt * u.rad, reco_result.az, reco_result.alt
            ).to(u.deg).value
            print(
                f"hillas_off_angle = {hillas_off_angle:0.3f} +/- {hillas_err*180./np.pi:0.3f} deg"
            )
            sum_hillas_result += [
                [
                    truth_energy,
                    n_tels,
                    hillas_off_angle,
                    hillas_err * 180.0 / np.pi,
                    reco_result.alt.to(u.deg).value,
                    reco_result.az.to(u.deg).value,
                    run_id,
                    event_id,
                ]
            ]
        else:
            print("hillas reconstruction is not valid.")


        tic_xing = time.perf_counter()
        xing_alt, xing_az, xing_weight, xing_n_tel = run_multiscopic_analysis(
            ctapipe_output, list_telescope_type, run_id, source, event
        )
        toc_xing = time.perf_counter()
        xing_err = 0.0
        xing_off_angle = 1e10

        if xing_weight > 0.0:
            xing_off_angle = angular_separation(
                truth_az * u.rad, truth_alt * u.rad, xing_az * u.rad, xing_alt * u.rad
            ).to(u.deg).value
            xing_err = pow(1.0 / xing_weight, 0.5)
            print(
                f"xing_n_tel = {xing_n_tel}, xing_off_angle = {xing_off_angle:0.3f} +/- {xing_err*180./np.pi:0.3f} deg ({toc_xing-tic_xing:0.1f} sec)"
            )
            sum_xing_result += [
                [
                    truth_energy,
                    xing_n_tel,
                    xing_off_angle,
                    xing_err * 180.0 / np.pi,
                    xing_alt * 180.0 / np.pi,
                    xing_az * 180.0 / np.pi,
                    run_id,
                    event_id,
                ]
            ]

        seed_alt = 0.0
        seed_az = 0.0
        seed_weight = 0.0
        if hillas_weight > xing_weight:
            print("use hillas seed.")
            seed_alt = hillas_alt
            seed_az = hillas_az
            seed_weight = hillas_weight
        else:
            print("use xing seed.")
            seed_alt = xing_alt
            seed_az = xing_az
            seed_weight = xing_weight

        tic_template = time.perf_counter()
        (
            truth_log_energy,
            avg_tmp_log_energy,
            avg_tmp_alt,
            avg_tmp_az,
            avg_tmp_err,
            list_tel_ana,
        ) = run_monoscopic_analysis(
            ctapipe_output,
            list_telescope_type,
            run_id,
            source,
            event,
            dict_image_to_physics_model,
            dict_movie_lookup_table,
            dict_movie_eigen_vectors,
            dict_image_lookup_table,
            dict_image_eigen_vectors,
            dict_time_lookup_table,
            dict_time_eigen_vectors,
            seed_alt,
            seed_az,
            seed_weight,
        )
        toc_template = time.perf_counter()

        avg_tmp_weight = 0.
        if avg_tmp_err > 0.0:
            avg_tmp_weight = 1./(avg_tmp_err*avg_tmp_err)

        tmp_off_angle = angular_separation(
            truth_az * u.rad,
            truth_alt * u.rad,
            avg_tmp_az * u.rad,
            avg_tmp_alt * u.rad,
        )
        print(
            f"tmp_off_angle = {tmp_off_angle.to(u.deg).value:0.3f} +/- {avg_tmp_err*180./np.pi:0.3f} deg ({toc_template-tic_template:0.1f} sec)"
        )
        print (f"truth_alt = {truth_alt:0.3f}, avg_tmp_alt = {avg_tmp_alt:0.3f}")
        print (f"truth_az = {truth_az:0.3f}, avg_tmp_az = {avg_tmp_az:0.3f}")
        print(
            f"truth_log_energy = {pow(10.,truth_log_energy):0.2f} TeV, tmp_log_energy = {pow(10.,avg_tmp_log_energy):0.2f} TeV"
        )
        sum_template_result += [
            [
                truth_energy,
                tmp_off_angle.to(u.deg).value,
                avg_tmp_err * 180.0 / np.pi,
                avg_tmp_alt * 180.0 / np.pi,
                avg_tmp_az * 180.0 / np.pi,
                truth_log_energy,
                avg_tmp_log_energy,
                list_tel_ana,
                toc_template - tic_template,
                run_id,
                event_id,
            ]
        ]

        
        combine_alt = 0.0
        combine_az = 0.0
        combine_weight = 0.0
        combine_err = 0.0
        combine_alt += avg_tmp_alt * avg_tmp_weight
        combine_az += avg_tmp_az * avg_tmp_weight
        combine_weight += avg_tmp_weight
        combine_err += 1.0
        combine_alt += xing_alt * xing_weight
        combine_az += xing_az * xing_weight
        combine_weight += xing_weight
        combine_err += 1.0
        if combine_weight > 0.0:

            combine_alt = combine_alt / combine_weight
            combine_az = combine_az / combine_weight
            combine_err = pow(combine_err / combine_weight, 0.5)

            combine_rms = 0.0
            combine_rms += (pow(avg_tmp_alt-combine_alt,2)+pow(avg_tmp_az-combine_az,2))*avg_tmp_weight
            combine_rms += (pow(xing_alt-combine_alt,2)+pow(xing_az-combine_az,2))*xing_weight
            combine_rms += (pow(hillas_alt-combine_alt,2)+pow(hillas_az-combine_az,2))*hillas_weight
            combine_rms = pow(combine_rms / combine_weight, 0.5)

            combine_err = max(combine_err,combine_rms)

            combine_off_angle = angular_separation(
                truth_az * u.rad,
                truth_alt * u.rad,
                combine_az * u.rad,
                combine_alt * u.rad,
            )
            print(
                f"combine_off_angle = {combine_off_angle.to(u.deg).value:0.3f} +/- {combine_err*180./np.pi:0.3f} deg"
            )
            sum_combine_result += [
                [
                    truth_energy,
                    combine_off_angle.to(u.deg).value, 
                    combine_err * 180.0 / np.pi,
                    run_id,
                    event_id,
                ]
            ]

        #ana_tag = "veritas"
        analysis_result = [
            sum_hillas_result,
            sum_xing_result,
            sum_template_result,
            sum_combine_result,
            sum_truth_result,
        ]
        if save_output:
            output_filename = f"{ctapipe_output}/output_analysis/{ana_tag}_run{run_id}.pkl"
            print(f"writing file to {output_filename}")
            with open(output_filename, "wb") as file:
                pickle.dump(analysis_result, file)

        if xing_weight > 0.0:
            if hillas_off_angle<0.1:
                hillas_score += 1
            if xing_off_angle<0.1:
                xing_score += 1
        print (f"hillas_score = {hillas_score}")
        print (f"xing_score = {xing_score}")
        print (f"n_truth_events = {n_truth_events}")

        if xing_weight > 0.0:
            make_xing_plot = False
            bad_xing = False
            if run_diagnosis:
                make_xing_plot = True
            #if xing_off_angle/hillas_off_angle>5. and xing_off_angle/(hillas_err*180./np.pi)>3.0:
            #if xing_off_angle/hillas_off_angle>5.:
            #if xing_off_angle>0.5 and xing_off_angle/(xing_err*180./np.pi)>5.0:
            if xing_off_angle/(xing_err*180./np.pi)>5.0:
                make_xing_plot = True
                bad_xing = True
            if select_event_id != 0:
                make_xing_plot = True
            if make_xing_plot:
                xing_alt, xing_az, xing_weight, xing_n_tel = run_multiscopic_analysis(
                    ctapipe_output, list_telescope_type, run_id, source, event, make_a_plot=True,
                )
                #exit()
            if run_diagnosis and bad_xing:
                exit()


        print(
            "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
        )
        if select_event_id != 0:
            exit()

