
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
from ctapipe.visualization import ArrayDisplay, CameraDisplay
from traitlets.config import Config

ctapipe_output = os.environ.get("CTAPIPE_OUTPUT_PATH")
subprocess.call(['sh', './clean_plots.sh'])

font = {'family': 'serif', 'color':  'white', 'weight': 'normal', 'size': 10, 'rotation': 0.,}

time_direction_cut = 40.
image_direction_cut = 0.3

def remove_nan_pixels(image_2d):

    num_rows, num_cols = image_2d.shape
    for x_idx in range(0,num_cols):
        for y_idx in range(0,num_rows):
            if math.isnan(image_2d[y_idx,x_idx]): 
                image_2d[y_idx,x_idx] = 0.

def image_translation(input_image_2d, shift_row, shift_col):

    num_rows, num_cols = input_image_2d.shape

    image_shift = np.zeros_like(input_image_2d)
    for x_idx in range(0,num_cols):
        for y_idx in range(0,num_rows):
            if y_idx+shift_row<0: continue
            if x_idx-shift_col<0: continue
            if y_idx+shift_row>=num_rows: continue
            if x_idx-shift_col>=num_cols: continue
            image_shift[y_idx+shift_row,x_idx-shift_col] = input_image_2d[y_idx,x_idx]

    return image_shift

def image_rotation(input_image_2d, angle_rad):

    num_rows, num_cols = input_image_2d.shape

    image_rotate = np.zeros_like(input_image_2d)
    rotation_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],[np.sin(angle_rad), np.cos(angle_rad)]])
    center_col = float(num_cols)/2.
    center_row = float(num_rows)/2.
    for x_idx in range(0,num_cols):
        for y_idx in range(0,num_rows):
            delta_row = float(y_idx - center_row)
            delta_col = float(x_idx - center_col)
            delta_coord = np.array([delta_col,delta_row])
            rot_coord = rotation_matrix @ delta_coord
            rot_row = round(rot_coord[1]+center_row)
            rot_col = round(rot_coord[0]+center_col)
            if rot_row<0: continue
            if rot_row>=num_rows: continue
            if rot_col<0: continue
            if rot_col>=num_cols: continue
            image_rotate[rot_row,rot_col] = input_image_2d[y_idx,x_idx]

    return image_rotate

def fit_image_to_line(geometry,image_input_1d,transpose=False):

    x = []
    y = []
    w = []
    for pix in range(0,len(image_input_1d)):
        if image_input_1d[pix]==0.: continue
        if not transpose:
            x += [float(geometry.pix_x[pix]/u.m)]
            y += [float(geometry.pix_y[pix]/u.m)]
            w += [pow(image_input_1d[pix],1.0)]
        else:
            x += [float(geometry.pix_y[pix]/u.m)]
            y += [float(geometry.pix_x[pix]/u.m)]
            w += [pow(image_input_1d[pix],1.0)]
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
    for pix in range(0,len(image_input_1d)):
        if image_input_1d[pix]==0.: continue
        if not transpose:
            x += [[float(geometry.pix_x[pix]/u.m)-avg_x]]
            y += [[float(geometry.pix_y[pix]/u.m)-avg_y]]
            w += [pow(image_input_1d[pix],1.0)]
        else:
            x += [[float(geometry.pix_y[pix]/u.m)-avg_x]]
            y += [[float(geometry.pix_x[pix]/u.m)-avg_y]]
            w += [pow(image_input_1d[pix],1.0)]
    x = np.array(x)
    y = np.array(y)
    w = np.diag(w)

    # Compute the weighted SVD
    U, S, Vt = np.linalg.svd(w @ x, full_matrices=False)
    # Calculate the weighted pseudo-inverse
    if S[0]==0.:
        return 0., 0., np.inf
    S_pseudo_w = np.diag(1 / S)
    x_pseudo_w = Vt.T @ S_pseudo_w @ U.T
    # Compute the weighted least-squares solution
    A_svd = x_pseudo_w @ (w @ y)
    # Compute chi2
    chi2 = np.linalg.norm((w @ x).dot(A_svd)-(w @ y), 2)/np.trace(w)

    a = float(A_svd[0])
    b = float(avg_y - a*avg_x)

    if chi2==0.:
        return 0., 0., np.inf
    else:
        return a, b, np.trace(w)/chi2

def reset_time(input_image_1d, input_time_1d):

    center_time = 0.
    image_size = 0.
    for pix in range(0,len(input_image_1d)):
        if input_image_1d[pix]==0.: continue
        image_size += input_image_1d[pix]
        center_time += input_time_1d[pix]*input_image_1d[pix]

    if image_size==0.:
        return

    center_time = center_time/image_size
    for pix in range(0,len(input_image_1d)):
        if input_image_1d[pix]==0.: continue
        input_time_1d[pix] += -1.*center_time

    return

def find_image_moments(geometry, input_image_1d, input_time_1d):

    image_center_x = 0.
    image_center_y = 0.
    mask_center_x = 0.
    mask_center_y = 0.
    center_time = 0.
    image_size = 0.
    mask_size = 0.
    for pix in range(0,len(input_image_1d)):
        if input_image_1d[pix]==0.: continue
        mask_size += 1.
        mask_center_x += float(geometry.pix_x[pix]/u.m)
        mask_center_y += float(geometry.pix_y[pix]/u.m)
        image_size += input_image_1d[pix]
        image_center_x += float(geometry.pix_x[pix]/u.m)*input_image_1d[pix]
        image_center_y += float(geometry.pix_y[pix]/u.m)*input_image_1d[pix]
        center_time += input_time_1d[pix]*input_image_1d[pix]

    if image_size==0.:
        return 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.

    mask_center_x = mask_center_x/mask_size
    mask_center_y = mask_center_y/mask_size
    image_center_x = image_center_x/image_size
    image_center_y = image_center_y/image_size
    center_time = center_time/image_size


    #min_distance = 1e10
    #center_pix = 0
    #for pix in range(0,len(input_image_1d)):
    #    if input_image_1d[pix]==0.: continue
    #    distance = pow(pow(float(geometry.pix_x[pix]/u.m)-image_center_x,2) + pow(float(geometry.pix_y[pix]/u.m)-image_center_y,2),0.5)
    #    if min_distance>distance:
    #        min_distance = distance
    #        center_pix = pix
    #center_time = input_time_1d[center_pix]

    cov_xx = 0.
    cov_xy = 0.
    cov_yx = 0.
    cov_yy = 0.
    for pix in range(0,len(input_image_1d)):
        if input_image_1d[pix]==0.: continue
        diff_x = float(geometry.pix_x[pix]/u.m)-image_center_x
        diff_y = float(geometry.pix_y[pix]/u.m)-image_center_y
        weight = input_image_1d[pix]
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

    a, b, w = fit_image_to_line(geometry,input_image_1d)
    aT, bT, wT = fit_image_to_line(geometry,input_image_1d,transpose=True)
    #print (f'angle 1 = {np.arctan(a)}')
    #print (f'angle 2 = {np.arctan(1./aT)}')
    if w<wT:
        a = 1./aT
        b = -bT/aT
        w = wT
    if w==np.inf:
        return 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.
    angle = np.arctan(a)

    truth_angle = np.arctan2(-image_center_y,-image_center_x)
    print (f'angle = {angle}')
    print (f'truth_angle = {truth_angle}')

    rotation_matrix = np.array([[np.cos(-angle), -np.sin(-angle)],[np.sin(-angle), np.cos(-angle)]])
    diff_x = mask_center_x-image_center_x
    diff_y = mask_center_y-image_center_y
    delta_coord = np.array([diff_x,diff_y])
    rot_coord = rotation_matrix @ delta_coord
    direction_of_image = rot_coord[0]*image_size

    direction_of_time = 0.
    for pix in range(0,len(input_image_1d)):
        if input_image_1d[pix]==0.: continue
        diff_x = float(geometry.pix_x[pix]/u.m)-image_center_x
        diff_y = float(geometry.pix_y[pix]/u.m)-image_center_y
        diff_t = input_time_1d[pix]-center_time
        delta_coord = np.array([diff_x,diff_y])
        rot_coord = rotation_matrix @ delta_coord
        if rot_coord[0]==0.: continue
        direction_of_time += rot_coord[0]/abs(rot_coord[0])*diff_t*input_image_1d[pix]

    if abs(direction_of_image)/image_direction_cut<abs(direction_of_time)/time_direction_cut:
        if direction_of_time>0.:
            angle = angle+np.pi
            print (f'change direction.')
    else:
        if direction_of_image>0.:
            angle = angle+np.pi
            print (f'change direction.')
    print (f'new angle = {angle}')

    return image_size, image_center_x, image_center_y, angle, pow(semi_major_sq,0.5), pow(semi_minor_sq,0.5), abs(direction_of_time), abs(direction_of_image), a, b

fig, ax = plt.subplots()
figsize_x = 8.6
figsize_y = 6.4
fig.set_figheight(figsize_y)
fig.set_figwidth(figsize_x)

training_sample_path = "/nevis/ged/data/rshang/sct_40deg_prod3/gamma_40deg_0deg_run1879___cta-prod3-sct_desert-2150m-Paranal-SCT.simtel.gz"

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

for event in source:

    event_id = event.index['event_id']
    #if event_id!=183701: continue

    ntel = len(event.r0.tel)
    
    truth_energy = float(event.simulation.shower.energy/u.TeV)
    truth_core_x = float(event.simulation.shower.core_x/u.m)
    truth_core_y = float(event.simulation.shower.core_y/u.m)
    truth_alt = float(event.simulation.shower.alt/u.rad)
    truth_az = float(event.simulation.shower.az/u.rad)
    truth_height = float(event.simulation.shower.h_first_int/u.m)
    truth_x_max = float(event.simulation.shower.x_max/(u.g/(u.cm*u.cm)))
    print (f'truth_energy = {truth_energy} TeV')
    
    calib(event)  # fills in r1, dl0, and dl1
    #image_processor(event) # Takes DL1/Image data and cleans and parametrizes the images into DL1/parameters. Should be run after CameraCalibrator.
    #shower_processor(event) # Run the stereo event reconstruction on the input events.
    
    for tel_idx in range(0,len(list(event.dl0.tel.keys()))):

        tel_id = list(event.dl0.tel.keys())[tel_idx]
        print ('====================================================================================')
        print (f'event_id = {event_id}, tel_id = {tel_id}')

        geometry = subarray.tel[tel_id].camera.geometry

        dirty_image_1d = event.dl1.tel[tel_id].image
        dirty_image_2d = geometry.image_to_cartesian_representation(dirty_image_1d)
        remove_nan_pixels(dirty_image_2d)

        clean_image_1d = event.dl1.tel[tel_id].image
        clean_time_1d = event.dl1.tel[tel_id].peak_time
        image_mask = tailcuts_clean(geometry,clean_image_1d,boundary_thresh=1,picture_thresh=3,min_number_picture_neighbors=2)
        for pix in range(0,len(image_mask)):
            if not image_mask[pix]:
                clean_image_1d[pix] = 0.
                clean_time_1d[pix] = 0.
        #hillas_params = hillas_parameters(geometry, clean_image_1d)
        #print (f'hillas_params = {hillas_params}')
        #hillas_psi = hillas_params['psi']
        #hillas_x = hillas_params['x']
        #hillas_y = hillas_params['y']

        reset_time(clean_image_1d, clean_time_1d)

        clean_image_2d = geometry.image_to_cartesian_representation(clean_image_1d)
        remove_nan_pixels(clean_image_2d)
        clean_time_2d = geometry.image_to_cartesian_representation(clean_time_1d)
        remove_nan_pixels(clean_time_2d)

        pixel_width = float(geometry.pixel_width[0]/u.m)
        print (f'pixel_width = {pixel_width}')

        image_size, image_center_x, image_center_y, angle, semi_major, semi_minor, time_direction, image_direction, line_a, line_b = find_image_moments(geometry, clean_image_1d, clean_time_1d)
        if image_size==0.:
            continue
        #print (f'image_center_x = {image_center_x}')
        #print (f'image_center_y = {image_center_y}')
        #print (f'angle = {angle}')

        shift_pix_x = image_center_x/pixel_width
        shift_pix_y = image_center_y/pixel_width
        shift_image_2d = image_translation(clean_image_2d, round(float(shift_pix_y)), round(float(shift_pix_x)))
        rotate_image_2d = image_rotation(shift_image_2d, angle*u.rad)

        xmax = max(geometry.pix_x)/u.m
        xmin = min(geometry.pix_x)/u.m
        ymax = max(geometry.pix_y)/u.m
        ymin = min(geometry.pix_y)/u.m

        #fig.clf()
        #axbig = fig.add_subplot()
        #label_x = 'X'
        #label_y = 'Y'
        #axbig.set_xlabel(label_x)
        #axbig.set_ylabel(label_y)
        #im = axbig.imshow(dirty_image_2d,origin='lower',extent=(xmin,xmax,ymin,ymax))
        #cbar = fig.colorbar(im)
        #fig.savefig(f'{ctapipe_output}/output_plots/evt{event_id}_tel{tel_id}_dirty_image.png',bbox_inches='tight')
        #axbig.remove()

        fig.clf()
        axbig = fig.add_subplot()
        label_x = 'X'
        label_y = 'Y'
        axbig.set_xlabel(label_x)
        axbig.set_ylabel(label_y)
        im = axbig.imshow(clean_image_2d,origin='lower',extent=(xmin,xmax,ymin,ymax))
        cbar = fig.colorbar(im)
        axbig.scatter(0., 0., s=90, facecolors='none', edgecolors='r', marker='o')
        if np.cos(angle*u.rad)>0.:
            line_x = np.linspace(image_center_x, xmax, 100)
            line_y = -(line_a*line_x + line_b)
            axbig.plot(line_x,line_y,color='w',alpha=0.3,linestyle='dashed')
        else:
            line_x = np.linspace(xmin, image_center_x, 100)
            line_y = -(line_a*line_x + line_b)
            axbig.plot(line_x,line_y,color='w',alpha=0.3,linestyle='dashed')
        axbig.set_xlim(xmin,xmax)
        axbig.set_ylim(ymin,ymax)
        txt = axbig.text(-0.35, 0.35, 'image size = %0.2e'%(image_size), fontdict=font)
        txt = axbig.text(-0.35, 0.32, 'image direction = %0.2e'%(image_direction), fontdict=font)
        txt = axbig.text(-0.35, 0.29, 'time direction = %0.2e'%(time_direction), fontdict=font)
        image_qual = pow(image_direction/image_direction_cut,2)+pow(time_direction/time_direction_cut,2)
        txt = axbig.text(-0.35, 0.26, 'image quality = %0.2e'%(image_qual), fontdict=font)
        if image_qual<1.:
            txt = axbig.text(-0.35, 0.23, 'bad image!!!', fontdict=font)
        fig.savefig(f'{ctapipe_output}/output_plots/evt{event_id}_tel{tel_id}_clean_image.png',bbox_inches='tight')
        axbig.remove()

        fig.clf()
        axbig = fig.add_subplot()
        label_x = 'X'
        label_y = 'Y'
        axbig.set_xlabel(label_x)
        axbig.set_ylabel(label_y)
        im = axbig.imshow(clean_time_2d,origin='lower',extent=(xmin,xmax,ymin,ymax))
        cbar = fig.colorbar(im)
        axbig.scatter(0., 0., s=90, facecolors='none', edgecolors='r', marker='o')
        fig.savefig(f'{ctapipe_output}/output_plots/evt{event_id}_tel{tel_id}_clean_time.png',bbox_inches='tight')
        axbig.remove()

        #fig.clf()
        #axbig = fig.add_subplot()
        #label_x = 'X'
        #label_y = 'Y'
        #axbig.set_xlabel(label_x)
        #axbig.set_ylabel(label_y)
        #im = axbig.imshow(shift_image_2d,origin='lower',extent=(xmin,xmax,ymin,ymax))
        #cbar = fig.colorbar(im)
        #fig.savefig(f'{ctapipe_output}/output_plots/evt{event_id}_tel{tel_id}_shift_image.png',bbox_inches='tight')
        #axbig.remove()

        #fig.clf()
        #axbig = fig.add_subplot()
        #label_x = 'X'
        #label_y = 'Y'
        #axbig.set_xlabel(label_x)
        #axbig.set_ylabel(label_y)
        #im = axbig.imshow(rotate_image_2d,origin='lower',extent=(xmin,xmax,ymin,ymax))
        #cbar = fig.colorbar(im)
        #fig.savefig(f'{ctapipe_output}/output_plots/evt{event_id}_tel{tel_id}_rotate_image.png',bbox_inches='tight')
        #axbig.remove()

        #waveform = event.dl0.tel[tel_id].waveform
        #n_pix, n_samp = waveform.shape
        #print (f'tel_id = {tel_id}, n_pix = {n_pix}, n_samp = {n_samp}')

        #for pix in range(0,n_pix):
        #    #if image_mask[pix]: continue
        #    if not image_mask[pix]: continue
        #    pix_waveform = waveform[pix,:]
        #    fig.clf()
        #    axbig = fig.add_subplot()
        #    axbig.plot(pix_waveform)
        #    fig.savefig(f'{ctapipe_output}/output_plots/evt{event_id}_tel{tel_id}_pix{pix}_waveform.png',bbox_inches='tight')
        #    axbig.remove()

    #break

