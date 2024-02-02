
import os, sys

import math
import numpy as np
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from matplotlib import pyplot as plt
from matplotlib import colors

from ctapipe.coordinates import CameraFrame, NominalFrame, TelescopeFrame
from ctapipe.image import hillas_parameters, tailcuts_clean

time_direction_cut = 40.
image_direction_cut = 0.2

ctapipe_output = os.environ.get("CTAPIPE_OUTPUT_PATH")

font = {'family': 'serif', 'color':  'white', 'weight': 'normal', 'size': 10, 'rotation': 0.,}

class MyArray3D:

    def __init__(self,x_bins=10,start_x=0.,end_x=10.,y_bins=10,start_y=0.,end_y=10.,z_bins=10,start_z=0.,end_z=10.,overflow=True):
        array_shape = (x_bins,y_bins,z_bins)
        self.delta_x = (end_x-start_x)/float(x_bins)
        self.delta_y = (end_y-start_y)/float(y_bins)
        self.delta_z = (end_z-start_z)/float(z_bins)
        self.xaxis = np.zeros(array_shape[0]+1)
        self.yaxis = np.zeros(array_shape[1]+1)
        self.zaxis = np.zeros(array_shape[2]+1)
        self.waxis = np.zeros(array_shape)
        self.overflow = overflow
        for idx in range(0,len(self.xaxis)):
            self.xaxis[idx] = start_x + idx*self.delta_x
        for idx in range(0,len(self.yaxis)):
            self.yaxis[idx] = start_y + idx*self.delta_y
        for idx in range(0,len(self.zaxis)):
            self.zaxis[idx] = start_z + idx*self.delta_z

    def reset(self):
        for idx_x in range(0,len(self.xaxis)-1):
            for idx_y in range(0,len(self.yaxis)-1):
                for idx_z in range(0,len(self.zaxis)-1):
                    self.waxis[idx_x,idx_y,idx_z] = 0.

    def add(self, add_array, factor=1.):
        for idx_x in range(0,len(self.xaxis)-1):
            for idx_y in range(0,len(self.yaxis)-1):
                for idx_z in range(0,len(self.zaxis)-1):
                    self.waxis[idx_x,idx_y,idx_z] = self.waxis[idx_x,idx_y,idx_z]+add_array.waxis[idx_x,idx_y,idx_z]*factor

    def get_bin(self, value_x, value_y, value_z):
        key_idx_x = -1
        key_idx_y = -1
        key_idx_z = -1
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
            key_idx_x = len(self.xaxis)-2
        if value_y>self.yaxis.max():
            key_idx_y = len(self.yaxis)-2
        if value_z>self.zaxis.max():
            key_idx_z = len(self.zaxis)-2
        return [key_idx_x,key_idx_y,key_idx_z]

    def fill(self, value_x, value_y, value_z, weight=1.):
        key_idx = self.get_bin(value_x, value_y, value_z)
        key_idx_x = key_idx[0]
        key_idx_y = key_idx[1]
        key_idx_z = key_idx[2]
        if key_idx_x==-1: 
            key_idx_x = 0
            if not self.overflow: weight = 0.
        if key_idx_y==-1: 
            key_idx_y = 0
            if not self.overflow: weight = 0.
        if key_idx_z==-1: 
            key_idx_z = 0
            if not self.overflow: weight = 0.
        if key_idx_x==len(self.xaxis): 
            key_idx_x = len(self.xaxis)-2
            if not self.overflow: weight = 0.
        if key_idx_y==len(self.yaxis): 
            key_idx_y = len(self.yaxis)-2
            if not self.overflow: weight = 0.
        if key_idx_z==len(self.zaxis): 
            key_idx_z = len(self.zaxis)-2
            if not self.overflow: weight = 0.
        self.waxis[key_idx_x,key_idx_y,key_idx_z] += 1.*weight
    
    def divide(self, add_array):
        for idx_x in range(0,len(self.xaxis)-1):
            for idx_y in range(0,len(self.yaxis)-1):
                for idx_z in range(0,len(self.zaxis)-1):
                    if add_array.waxis[idx_x,idx_y,idx_z]==0.:
                        self.waxis[idx_x,idx_y,idx_z] = 0.
                    else:
                        self.waxis[idx_x,idx_y,idx_z] = self.waxis[idx_x,idx_y,idx_z]/add_array.waxis[idx_x,idx_y,idx_z]

    def get_bin_center(self, idx_x, idx_y, idx_z):
        return [self.xaxis[idx_x]+0.5*self.delta_x,self.yaxis[idx_y]+0.5*self.delta_y,self.zaxis[idx_z]+0.5*self.delta_z]

    def get_bin_content(self, value_x, value_y, value_z):
        key_idx = self.get_bin(value_x, value_y, value_z)
        key_idx_x = key_idx[0]
        key_idx_y = key_idx[1]
        key_idx_z = key_idx[2]
        if key_idx_x==-1: 
            key_idx_x = 0
        if key_idx_y==-1: 
            key_idx_y = 0
        if key_idx_z==-1: 
            key_idx_z = 0
        if key_idx_x==len(self.xaxis): 
            key_idx_x = len(self.xaxis)-2
        if key_idx_y==len(self.yaxis): 
            key_idx_y = len(self.yaxis)-2
        if key_idx_z==len(self.zaxis): 
            key_idx_z = len(self.zaxis)-2
        return self.waxis[key_idx_x,key_idx_y,key_idx_z]

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

    direction_of_time = direction_of_time/time_direction_cut
    direction_of_image = direction_of_image/image_direction_cut
    if (direction_of_time+direction_of_image)>0.:
        angle = angle+np.pi
        print (f'change direction.')

    print (f'new angle = {angle}')

    return [image_size, image_center_x, image_center_y, angle, pow(semi_major_sq,0.5), pow(semi_minor_sq,0.5), direction_of_time, direction_of_image, a, b]

def find_image_truth(source, subarray, run_id, tel_id, event):

    truth_energy = event.simulation.shower.energy
    truth_core_x = event.simulation.shower.core_x
    truth_core_y = event.simulation.shower.core_y
    truth_alt = event.simulation.shower.alt
    truth_az = event.simulation.shower.az
    truth_height = event.simulation.shower.h_first_int
    truth_x_max = event.simulation.shower.x_max
    print (f'truth_energy = {truth_energy}')

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
    impact_x = float((truth_core_x-tel_x)/u.m)
    impact_y = float((truth_core_y-tel_y)/u.m)

    camera_frame = CameraFrame(
        telescope_pointing=tel_pointing,
        focal_length=focal_length,
    )

    star_cam = star_altaz.transform_to(camera_frame)
    star_cam_x = star_cam.x.to_value(u.m)
    star_cam_y = star_cam.y.to_value(u.m)

    truth_info_array = [truth_energy, truth_core_x, truth_core_y, truth_alt, truth_az, truth_height, truth_x_max, star_cam_x, star_cam_y, impact_x, impact_y]

    return truth_info_array

def make_a_movie(subarray, run_id, tel_id, event):

    #fig, ax = plt.subplots()
    #figsize_x = 8.6
    #figsize_y = 6.4
    #fig.set_figheight(figsize_y)
    #fig.set_figwidth(figsize_x)

    event_id = event.index['event_id']
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

    reset_time(clean_image_1d, clean_time_1d)

    clean_image_2d = geometry.image_to_cartesian_representation(clean_image_1d)
    remove_nan_pixels(clean_image_2d)
    clean_time_2d = geometry.image_to_cartesian_representation(clean_time_1d)
    remove_nan_pixels(clean_time_2d)

    pixel_width = float(geometry.pixel_width[0]/u.m)

    image_moment_array = find_image_moments(geometry, clean_image_1d, clean_time_1d)
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

    image_qual = abs(image_direction+time_direction)

    shift_pix_x = image_center_x/pixel_width
    shift_pix_y = image_center_y/pixel_width
    shift_image_2d = image_translation(clean_image_2d, round(float(shift_pix_y)), round(float(shift_pix_x)))
    rotate_image_2d = image_rotation(shift_image_2d, angle*u.rad)

    waveform = event.dl0.tel[tel_id].waveform
    n_pix, n_samp = waveform.shape

    max_sample = 40
    window_size = 8
    n_windows = 5
    clean_movie_1d = []
    for win in range(0,n_windows):
        clean_movie_1d += [np.zeros_like(clean_image_1d)]
    for pix in range(0,n_pix):
        if not image_mask[pix]: continue # select signal
        for win in range(0,n_windows):
            for sample in range(0,window_size):
                sample_idx = sample + win*window_size
                clean_movie_1d[win][pix] +=  waveform[pix,sample_idx]

    xmax = max(geometry.pix_x)/u.m
    xmin = min(geometry.pix_x)/u.m
    ymax = max(geometry.pix_y)/u.m
    ymin = min(geometry.pix_y)/u.m

    whole_movie_1d = []
    for win in range(0,n_windows):

        clean_movie_2d = geometry.image_to_cartesian_representation(clean_movie_1d[win])
        remove_nan_pixels(clean_movie_2d)

        shift_movie_2d = image_translation(clean_movie_2d, round(float(shift_pix_y)), round(float(shift_pix_x)))
        rotate_movie_2d = image_rotation(shift_movie_2d, angle*u.rad)
        rotate_movie_1d = geometry.image_from_cartesian_representation(rotate_movie_2d)
        whole_movie_1d.extend(rotate_movie_1d)
        print (f'len(whole_movie_1d) = {len(whole_movie_1d)}')

        #if image_qual>100.:
        #    fig.clf()
        #    axbig = fig.add_subplot()
        #    label_x = 'X'
        #    label_y = 'Y'
        #    axbig.set_xlabel(label_x)
        #    axbig.set_ylabel(label_y)
        #    im = axbig.imshow(rotate_movie_2d,origin='lower',extent=(xmin,xmax,ymin,ymax))
        #    cbar = fig.colorbar(im)
        #    fig.savefig(f'{ctapipe_output}/output_plots/evt{event_id}_tel{tel_id}_win{win}_clean_movie.png',bbox_inches='tight')
        #    axbig.remove()

    #fig.clf()
    #axbig = fig.add_subplot()
    #label_x = 'X'
    #label_y = 'Y'
    #axbig.set_xlabel(label_x)
    #axbig.set_ylabel(label_y)
    #im = axbig.imshow(clean_image_2d,origin='lower',extent=(xmin,xmax,ymin,ymax))
    #cbar = fig.colorbar(im)
    #axbig.scatter(0., 0., s=90, facecolors='none', edgecolors='r', marker='o')
    #if np.cos(angle*u.rad)>0.:
    #    line_x = np.linspace(image_center_x, xmax, 100)
    #    line_y = -(line_a*line_x + line_b)
    #    axbig.plot(line_x,line_y,color='w',alpha=0.3,linestyle='dashed')
    #else:
    #    line_x = np.linspace(xmin, image_center_x, 100)
    #    line_y = -(line_a*line_x + line_b)
    #    axbig.plot(line_x,line_y,color='w',alpha=0.3,linestyle='dashed')
    #axbig.set_xlim(xmin,xmax)
    #axbig.set_ylim(ymin,ymax)
    #txt = axbig.text(-0.35, 0.35, 'image size = %0.2e'%(image_size), fontdict=font)
    #txt = axbig.text(-0.35, 0.32, 'image direction = %0.2e'%(image_direction), fontdict=font)
    #txt = axbig.text(-0.35, 0.29, 'time direction = %0.2e'%(time_direction), fontdict=font)
    #txt = axbig.text(-0.35, 0.26, 'image quality = %0.2e'%(image_qual), fontdict=font)
    #if image_qual<1.:
    #    txt = axbig.text(-0.35, 0.23, 'bad image!!!', fontdict=font)
    #fig.savefig(f'{ctapipe_output}/output_plots/evt{event_id}_tel{tel_id}_clean_image.png',bbox_inches='tight')
    #axbig.remove()

    #fig.clf()
    #axbig = fig.add_subplot()
    #label_x = 'X'
    #label_y = 'Y'
    #axbig.set_xlabel(label_x)
    #axbig.set_ylabel(label_y)
    #im = axbig.imshow(clean_time_2d,origin='lower',extent=(xmin,xmax,ymin,ymax))
    #cbar = fig.colorbar(im)
    #axbig.scatter(0., 0., s=90, facecolors='none', edgecolors='r', marker='o')
    #fig.savefig(f'{ctapipe_output}/output_plots/evt{event_id}_tel{tel_id}_clean_time.png',bbox_inches='tight')
    #axbig.remove()

    return image_qual, image_moment_array, whole_movie_1d

