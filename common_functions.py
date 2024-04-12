
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

training_event_select = 2

time_direction_cut = 1.
image_direction_cut = 1.
image_size_cut = 100.
#image_size_cut = 5000.

n_samples_per_window = 1
total_samples = 64
#select_samples = 10
select_samples = 16

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

def image_smooth(input_image_2d,mask_2d):

    num_rows, num_cols = input_image_2d.shape
    center_col = float(num_cols)/2.
    center_row = float(num_rows)/2.

    #mode = 50.
    #kernel_radius = float(num_cols)/mode
    kernel_radius = 2.
    kernel_pix_size = int(kernel_radius)
    gaus_norm = 2.*np.pi*kernel_radius*kernel_radius
    image_kernel = np.zeros_like(input_image_2d)
    for idx_x in range(0,num_cols):
        for idx_y in range(0,num_rows):
            delta_row = float(idx_y - center_row)
            delta_col = float(idx_x - center_col)
            distance = pow(delta_row*delta_row+delta_col*delta_col,0.5)
            pix_content = np.exp(-(distance*distance)/(2.*kernel_radius*kernel_radius))
            image_kernel[idx_y,idx_x] = pix_content/gaus_norm

    image_smooth = np.zeros_like(input_image_2d)
    kernel_norm = np.sum(image_kernel)
    for idx_x1 in range(0,num_cols):
        for idx_y1 in range(0,num_rows):
            image_smooth[idx_y1,idx_x1] = 0.
            if not mask_2d[idx_y1,idx_x1]: continue # select signal
            for idx_x2 in range(idx_x1-2*kernel_pix_size,idx_x1+2*kernel_pix_size):
                for idx_y2 in range(idx_y1-2*kernel_pix_size,idx_y1+2*kernel_pix_size):
                    if idx_x2<0: continue
                    if idx_y2<0: continue
                    if idx_x2>=num_cols: continue
                    if idx_y2>=num_rows: continue
                    if not mask_2d[idx_y2,idx_x2]: continue # select signal
                    old_content = input_image_2d[idx_y2,idx_x2]
                    scale = image_kernel[int(center_row)+idx_y2-idx_y1,int(center_col)+idx_x2-idx_x1]
                    image_smooth[idx_y1,idx_x1] += old_content*scale/kernel_norm

    return image_smooth

def image_cutout(geometry, image_input_1d):

    eco_image_1d = []
    for pix in range(0,len(image_input_1d)):
        x = float(geometry.pix_x[pix]/u.m)
        y = float(geometry.pix_y[pix]/u.m)
        if abs(y)<0.05:
            eco_image_1d += [image_input_1d[pix]]
        else:
            image_input_1d[pix] = 0.
    return eco_image_1d

def image_cutout_restore(geometry, eco_image_1d, origin_image_1d):

    eco_pix = 0
    for pix in range(0,len(origin_image_1d)):
        x = float(geometry.pix_x[pix]/u.m)
        y = float(geometry.pix_y[pix]/u.m)
        if abs(y)<0.05:
            origin_image_1d[pix] = eco_image_1d[eco_pix]
            eco_pix += 1
        else:
            origin_image_1d[pix] = 0.

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
        return 0.

    center_time = center_time/image_size

    for pix in range(0,len(input_image_1d)):
        if input_image_1d[pix]==0.: continue
        input_time_1d[pix] += -1.*center_time

    return center_time

def find_image_moments(geometry, input_image_1d, input_time_1d, star_cam_xy=None):

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
        return [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]

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
        return [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
    angle = np.arctan(a)

    rotation_matrix = np.array([[np.cos(-angle), -np.sin(-angle)],[np.sin(-angle), np.cos(-angle)]])
    diff_x = mask_center_x-image_center_x
    diff_y = mask_center_y-image_center_y
    delta_coord = np.array([diff_x,diff_y])
    rot_coord = rotation_matrix @ delta_coord
    direction_of_image = rot_coord[0]*image_size

    direction_of_time = 0.
    diff_t_norm = 0.
    for pix in range(0,len(input_image_1d)):
        if input_image_1d[pix]==0.: continue
        diff_x = float(geometry.pix_x[pix]/u.m)-image_center_x
        diff_y = float(geometry.pix_y[pix]/u.m)-image_center_y
        diff_t = input_time_1d[pix]-center_time
        delta_coord = np.array([diff_x,diff_y])
        rot_coord = rotation_matrix @ delta_coord
        if rot_coord[0]==0.: continue
        direction_of_time += rot_coord[0]*diff_t*input_image_1d[pix]
        diff_t_norm += diff_t*diff_t
    if diff_t_norm>0.:
        direction_of_time = direction_of_time/pow(diff_t_norm,0.5)

    direction_of_time = direction_of_time/time_direction_cut
    direction_of_image = direction_of_image/image_direction_cut
    #if (direction_of_time+direction_of_image)>0.:
    if (direction_of_image)>0.:
        angle = angle+np.pi
        #print (f'change direction.')

    truth_angle = np.arctan2(-image_center_y,-image_center_x)
    if not star_cam_xy==None:
        truth_angle = np.arctan2(star_cam_xy[1]-image_center_y,star_cam_xy[0]-image_center_x)
    #print (f'angle = {angle}')
    #print (f'truth_angle = {truth_angle}')

    truth_projection = np.cos(truth_angle-angle)

    if not star_cam_xy==None:
        angle = truth_angle

    return [image_size, image_center_x, image_center_y, angle, pow(semi_major_sq,0.5), pow(semi_minor_sq,0.5), direction_of_time, direction_of_image, a, b, truth_projection]

def camxy_to_altaz(source, subarray, run_id, tel_id, star_cam_x, star_cam_y):

    geometry = subarray.tel[tel_id].camera.geometry

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
        x=star_cam_x*u.m,
        y=star_cam_y*u.m,
        frame=camera_frame,
    )
    
    star_altaz = star_cam.transform_to(altaz)
    star_alt = star_altaz.alt.to_value(u.rad)
    star_az = star_altaz.az.to_value(u.rad)

    return star_alt, star_az

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

def movie_simulation(fig, subarray, run_id, tel_id, event, init_params, movie_lookup_table, movie_eigen_vectors):

    event_id = event.index['event_id']
    geometry = subarray.tel[tel_id].camera.geometry

    lightcone, image_moment_array, eco_image_1d, eco_time_1d = make_standard_image(fig, subarray, run_id, tel_id, event)
    n_eco_pix = len(eco_image_1d)

    fit_arrival = init_params[0]
    fit_impact = init_params[1]
    fit_log_energy = init_params[2]

    fit_movie_latent_space = []
    for r in range(0,len(movie_lookup_table)):
        fit_movie_latent_space += [movie_lookup_table[r].get_bin_content(fit_arrival,fit_impact,fit_log_energy)]
    fit_movie_latent_space = np.array(fit_movie_latent_space)

    eco_movie_1d_fit = movie_eigen_vectors.T @ fit_movie_latent_space
    n_windows = int(select_samples/n_samples_per_window)
    sim_eco_image_1d = []
    sim_image_1d = []
    for win in range(0,n_windows):
        sim_eco_image_1d += [np.zeros_like(eco_image_1d)]
        sim_image_1d += [np.zeros_like(event.dl1.tel[tel_id].image)]

    sim_image_2d = []
    for win in range(0,n_windows):
        for pix in range(0,n_eco_pix):
            movie_pix_idx = pix + win*n_eco_pix
            sim_eco_image_1d[win][pix] = eco_movie_1d_fit[movie_pix_idx]
        image_cutout_restore(geometry, sim_eco_image_1d[win], sim_image_1d[win])
        sim_image_2d += [geometry.image_to_cartesian_representation(sim_image_1d[win])]

    xmax = max(geometry.pix_x)/u.m
    xmin = min(geometry.pix_x)/u.m
    ymax = max(geometry.pix_y)/u.m
    ymin = min(geometry.pix_y)/u.m

    image_max = np.max(eco_image_1d[:])

    for win in range(0,n_windows):

        fig.clf()
        axbig = fig.add_subplot()
        label_x = 'X'
        label_y = 'Y'
        axbig.set_xlabel(label_x)
        axbig.set_ylabel(label_y)
        im = axbig.imshow(sim_image_2d[win],origin='lower',extent=(xmin,xmax,ymin,ymax), vmin=0.,vmax=2.*image_max/float(n_windows))
        #im = axbig.imshow(sim_image_2d[win],origin='lower',extent=(xmin,xmax,ymin,ymax), norm=colors.LogNorm())
        cbar = fig.colorbar(im)
        axbig.set_xlim(xmin,xmax)
        axbig.set_ylim(ymin,ymax)
        fig.savefig(f'{ctapipe_output}/output_plots/evt{event_id}_tel{tel_id}_movie{win}_sim.png',bbox_inches='tight')
        axbig.remove()


def image_simulation(fig, subarray, run_id, tel_id, event, init_params, image_lookup_table, image_eigen_vectors, time_lookup_table, time_eigen_vectors):

    event_id = event.index['event_id']
    geometry = subarray.tel[tel_id].camera.geometry

    lightcone, image_moment_array, eco_image_1d, eco_time_1d = make_standard_image(fig, subarray, run_id, tel_id, event)

    fit_arrival = init_params[0]
    fit_impact = init_params[1]
    fit_log_energy = init_params[2]

    fit_image_latent_space = []
    for r in range(0,len(image_lookup_table)):
        fit_image_latent_space += [image_lookup_table[r].get_bin_content(fit_arrival,fit_impact,fit_log_energy)]
    fit_image_latent_space = np.array(fit_image_latent_space)

    sim_image_1d = np.zeros_like(event.dl1.tel[tel_id].image)
    eco_image_1d_fit = image_eigen_vectors.T @ fit_image_latent_space
    image_cutout_restore(geometry, eco_image_1d_fit, sim_image_1d)
    sim_image_2d = geometry.image_to_cartesian_representation(sim_image_1d)

    data_image_1d = np.zeros_like(event.dl1.tel[tel_id].image)
    image_cutout_restore(geometry, eco_image_1d, data_image_1d)
    data_image_2d = geometry.image_to_cartesian_representation(data_image_1d)

    fit_time_latent_space = []
    for r in range(0,len(time_lookup_table)):
        fit_time_latent_space += [time_lookup_table[r].get_bin_content(fit_arrival,fit_impact,fit_log_energy)]
    fit_time_latent_space = np.array(fit_time_latent_space)

    sim_time_1d = np.zeros_like(event.dl1.tel[tel_id].image)
    eco_time_1d_fit = time_eigen_vectors.T @ fit_time_latent_space
    image_cutout_restore(geometry, eco_time_1d_fit, sim_time_1d)
    sim_time_2d = geometry.image_to_cartesian_representation(sim_time_1d)

    data_time_1d = np.zeros_like(event.dl1.tel[tel_id].image)
    image_cutout_restore(geometry, eco_time_1d, data_time_1d)
    data_time_2d = geometry.image_to_cartesian_representation(data_time_1d)

    xmax = max(geometry.pix_x)/u.m
    xmin = min(geometry.pix_x)/u.m
    ymax = max(geometry.pix_y)/u.m
    ymin = min(geometry.pix_y)/u.m

    fig.clf()
    axbig = fig.add_subplot()
    label_x = 'X'
    label_y = 'Y'
    axbig.set_xlabel(label_x)
    axbig.set_ylabel(label_y)
    im = axbig.imshow(data_image_2d,origin='lower',extent=(xmin,xmax,ymin,ymax))
    cbar = fig.colorbar(im)
    axbig.set_xlim(xmin,xmax)
    axbig.set_ylim(ymin,ymax)
    fig.savefig(f'{ctapipe_output}/output_plots/evt{event_id}_tel{tel_id}_image_data.png',bbox_inches='tight')
    axbig.remove()

    fig.clf()
    axbig = fig.add_subplot()
    label_x = 'X'
    label_y = 'Y'
    axbig.set_xlabel(label_x)
    axbig.set_ylabel(label_y)
    im = axbig.imshow(sim_image_2d,origin='lower',extent=(xmin,xmax,ymin,ymax))
    cbar = fig.colorbar(im)
    axbig.set_xlim(xmin,xmax)
    axbig.set_ylim(ymin,ymax)
    fig.savefig(f'{ctapipe_output}/output_plots/evt{event_id}_tel{tel_id}_image_sim.png',bbox_inches='tight')
    axbig.remove()

    fig.clf()
    axbig = fig.add_subplot()
    label_x = 'X'
    label_y = 'Y'
    axbig.set_xlabel(label_x)
    axbig.set_ylabel(label_y)
    im = axbig.imshow(data_time_2d,origin='lower',extent=(xmin,xmax,ymin,ymax))
    cbar = fig.colorbar(im)
    axbig.set_xlim(xmin,xmax)
    axbig.set_ylim(ymin,ymax)
    fig.savefig(f'{ctapipe_output}/output_plots/evt{event_id}_tel{tel_id}_time_data.png',bbox_inches='tight')
    axbig.remove()

    fig.clf()
    axbig = fig.add_subplot()
    label_x = 'X'
    label_y = 'Y'
    axbig.set_xlabel(label_x)
    axbig.set_ylabel(label_y)
    im = axbig.imshow(sim_time_2d,origin='lower',extent=(xmin,xmax,ymin,ymax))
    cbar = fig.colorbar(im)
    axbig.set_xlim(xmin,xmax)
    axbig.set_ylim(ymin,ymax)
    fig.savefig(f'{ctapipe_output}/output_plots/evt{event_id}_tel{tel_id}_time_sim.png',bbox_inches='tight')
    axbig.remove()

    return

def calculate_lightcone(image_direction,time_direction):

    lightcone = 0.
    if (image_direction*image_direction)>0.:
        lightcone = image_direction*time_direction/(image_direction*image_direction)

    print (f'image_direction = {image_direction:0.2f}, time_direction = {time_direction:0.2f}, lightcone = {lightcone:0.2f}')

    return lightcone

def pass_lightcone(lightcone,image_direction):

    if abs(image_direction)<1.: 
        return False
    if lightcone<-0.6:
        return False

    return True

def plot_monotel_reconstruction(fig, subarray, run_id, tel_id, event, image_moment_array, fit_cam_x, fit_cam_y, tag):

    event_id = event.index['event_id']
    geometry = subarray.tel[tel_id].camera.geometry

    dirty_image_1d = event.dl1.tel[tel_id].image
    dirty_image_2d = geometry.image_to_cartesian_representation(dirty_image_1d)
    remove_nan_pixels(dirty_image_2d)

    clean_image_1d = np.zeros_like(event.dl1.tel[tel_id].image)
    clean_time_1d = np.zeros_like(event.dl1.tel[tel_id].peak_time)
    image_mask = tailcuts_clean(geometry,event.dl1.tel[tel_id].image,boundary_thresh=1,picture_thresh=3,min_number_picture_neighbors=2)
    for pix in range(0,len(image_mask)):
        if not image_mask[pix]:
            clean_image_1d[pix] = 0.
            clean_time_1d[pix] = 0.
        else:
            clean_image_1d[pix] = event.dl1.tel[tel_id].image[pix]
            clean_time_1d[pix] = event.dl1.tel[tel_id].peak_time[pix]

    center_time = reset_time(clean_image_1d, clean_time_1d)

    clean_image_2d = geometry.image_to_cartesian_representation(clean_image_1d)
    remove_nan_pixels(clean_image_2d)
    clean_time_2d = geometry.image_to_cartesian_representation(clean_time_1d)
    remove_nan_pixels(clean_time_2d)

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

    lightcone = calculate_lightcone(image_direction,time_direction)

    xmax = max(geometry.pix_x)/u.m
    xmin = min(geometry.pix_x)/u.m
    ymax = max(geometry.pix_y)/u.m
    ymin = min(geometry.pix_y)/u.m

    fig.clf()
    axbig = fig.add_subplot()
    label_x = 'X'
    label_y = 'Y'
    axbig.set_xlabel(label_x)
    axbig.set_ylabel(label_y)
    im = axbig.imshow(clean_image_2d,origin='lower',extent=(xmin,xmax,ymin,ymax))
    cbar = fig.colorbar(im)
    axbig.scatter(0., 0., s=90, facecolors='none', edgecolors='r', marker='o')
    axbig.scatter(fit_cam_x, -fit_cam_y, s=90, facecolors='none', c='r', marker='+')
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
    txt = axbig.text(-0.35, 0.26, 'light cone = %0.2e'%(lightcone), fontdict=font)
    fig.savefig(f'{ctapipe_output}/output_plots/evt{event_id}_tel{tel_id}_clean_image_{tag}.png',bbox_inches='tight')
    axbig.remove()


def make_standard_image(fig, subarray, run_id, tel_id, event, star_cam_xy=None, make_plots=False):

    event_id = event.index['event_id']
    geometry = subarray.tel[tel_id].camera.geometry

    dirty_image_1d = event.dl1.tel[tel_id].image
    dirty_image_2d = geometry.image_to_cartesian_representation(dirty_image_1d)
    remove_nan_pixels(dirty_image_2d)

    clean_image_1d = np.zeros_like(event.dl1.tel[tel_id].image)
    clean_time_1d = np.zeros_like(event.dl1.tel[tel_id].peak_time)
    image_mask = tailcuts_clean(geometry,event.dl1.tel[tel_id].image,boundary_thresh=1,picture_thresh=3,min_number_picture_neighbors=2)
    for pix in range(0,len(image_mask)):
        if not image_mask[pix]:
            clean_image_1d[pix] = 0.
            clean_time_1d[pix] = 0.
        else:
            clean_image_1d[pix] = event.dl1.tel[tel_id].image[pix]
            clean_time_1d[pix] = event.dl1.tel[tel_id].peak_time[pix]

    center_time = reset_time(clean_image_1d, clean_time_1d)

    clean_image_2d = geometry.image_to_cartesian_representation(clean_image_1d)
    remove_nan_pixels(clean_image_2d)
    clean_time_2d = geometry.image_to_cartesian_representation(clean_time_1d)
    remove_nan_pixels(clean_time_2d)

    pixel_width = float(geometry.pixel_width[0]/u.m)

    image_moment_array = find_image_moments(geometry, clean_image_1d, clean_time_1d, star_cam_xy=star_cam_xy)
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
    truth_projection = image_moment_array[10]

    lightcone = calculate_lightcone(image_direction,time_direction)

    shift_pix_x = image_center_x/pixel_width
    shift_pix_y = image_center_y/pixel_width
    shift_image_2d = image_translation(clean_image_2d, round(float(shift_pix_y)), round(float(shift_pix_x)))
    shift_time_2d = image_translation(clean_time_2d, round(float(shift_pix_y)), round(float(shift_pix_x)))
    rotate_image_2d = image_rotation(shift_image_2d, angle*u.rad)
    rotate_time_2d = image_rotation(shift_time_2d, angle*u.rad)
    rotate_image_1d = geometry.image_from_cartesian_representation(rotate_image_2d)
    rotate_time_1d = geometry.image_from_cartesian_representation(rotate_time_2d)
    eco_image_1d = image_cutout(geometry, rotate_image_1d)
    eco_time_1d = image_cutout(geometry, rotate_time_1d)

    return lightcone, image_moment_array, eco_image_1d, eco_time_1d


def display_a_movie(fig, subarray, run_id, tel_id, event, eco_image_size, eco_movie_1d):

    n_windows = int(select_samples/n_samples_per_window)
    eco_image_1d = []
    for win in range(0,n_windows):
        eco_image_1d += [np.zeros(eco_image_size)]

    for win in range(0,n_windows):
        for pix in range(0,eco_image_size):
            entry = pix + win*eco_image_size
            eco_image_1d[win][pix] = eco_movie_1d[entry]

    event_id = event.index['event_id']
    geometry = subarray.tel[tel_id].camera.geometry
    xmax = max(geometry.pix_x)/u.m
    xmin = min(geometry.pix_x)/u.m
    ymax = max(geometry.pix_y)/u.m
    ymin = min(geometry.pix_y)/u.m

    dirty_image_1d = event.dl1.tel[tel_id].image
    image_max = np.max(dirty_image_1d[:])
    image_1d = []
    for win in range(0,n_windows):
        image_1d += [np.zeros_like(dirty_image_1d)]
    
    for win in range(0,n_windows):
        image_cutout_restore(geometry, eco_image_1d[win], image_1d[win])
        image_2d = geometry.image_to_cartesian_representation(image_1d[win])

        fig.clf()
        axbig = fig.add_subplot()
        label_x = 'X'
        label_y = 'Y'
        axbig.set_xlabel(label_x)
        axbig.set_ylabel(label_y)
        im = axbig.imshow(image_2d,origin='lower',extent=(xmin,xmax,ymin,ymax),vmin=0.,vmax=2.*image_max/float(n_windows))
        #im = axbig.imshow(image_2d,origin='lower',extent=(xmin,xmax,ymin,ymax), norm=colors.LogNorm())
        cbar = fig.colorbar(im)
        line_x = np.linspace(xmin, xmax, 100)
        line_y = -(0.*line_x + 0.05)
        axbig.plot(line_x,line_y,color='w',alpha=0.3,linestyle='dashed')
        line_x = np.linspace(xmin, xmax, 100)
        line_y = -(0.*line_x - 0.05)
        axbig.plot(line_x,line_y,color='w',alpha=0.3,linestyle='dashed')
        fig.savefig(f'{ctapipe_output}/output_plots/evt{event_id}_tel{tel_id}_movie{win}_data.png',bbox_inches='tight')
        axbig.remove()

def make_a_movie(fig, subarray, run_id, tel_id, event, star_cam_xy=None, make_plots=False):

    event_id = event.index['event_id']
    geometry = subarray.tel[tel_id].camera.geometry

    dirty_image_1d = event.dl1.tel[tel_id].image
    dirty_image_2d = geometry.image_to_cartesian_representation(dirty_image_1d)
    remove_nan_pixels(dirty_image_2d)

    clean_image_1d = np.zeros_like(event.dl1.tel[tel_id].image)
    clean_time_1d = np.zeros_like(event.dl1.tel[tel_id].peak_time)
    image_mask = tailcuts_clean(geometry,event.dl1.tel[tel_id].image,boundary_thresh=1,picture_thresh=3,min_number_picture_neighbors=2)
    for pix in range(0,len(image_mask)):
        if not image_mask[pix]:
            clean_image_1d[pix] = 0.
            clean_time_1d[pix] = 0.
        else:
            clean_image_1d[pix] = event.dl1.tel[tel_id].image[pix]
            clean_time_1d[pix] = event.dl1.tel[tel_id].peak_time[pix]

    center_time = reset_time(clean_image_1d, clean_time_1d)
    print (f'movie center_time = {center_time}')

    clean_image_2d = geometry.image_to_cartesian_representation(clean_image_1d)
    remove_nan_pixels(clean_image_2d)
    clean_time_2d = geometry.image_to_cartesian_representation(clean_time_1d)
    remove_nan_pixels(clean_time_2d)

    image_max = np.max(clean_image_2d[:,:])

    pixel_width = float(geometry.pixel_width[0]/u.m)

    image_moment_array = find_image_moments(geometry, clean_image_1d, clean_time_1d, star_cam_xy=star_cam_xy)
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
    truth_projection = image_moment_array[10]
    print (f'image_size = {image_size:0.1f}')

    lightcone = calculate_lightcone(image_direction,time_direction)

    whole_movie_1d = []

    shift_pix_x = image_center_x/pixel_width
    shift_pix_y = image_center_y/pixel_width
    #shift_image_2d = image_translation(clean_image_2d, round(float(shift_pix_y)), round(float(shift_pix_x)))
    #rotate_image_2d = image_rotation(shift_image_2d, angle*u.rad)
    #rotate_image_1d = geometry.image_from_cartesian_representation(rotate_image_2d)
    #eco_image_1d = image_cutout(geometry, rotate_image_1d)

    waveform = event.dl0.tel[tel_id].waveform
    n_pix, n_samp = waveform.shape
    print (f'n_pix = {n_pix}, n_samp = {n_samp}')


    #n_windows = 64
    ##n_windows = 16
    #window_size = int(n_samp/n_windows)
    #clean_movie_1d = []
    #for win in range(0,n_windows):
    #    clean_movie_1d += [np.zeros_like(clean_image_1d)]
    #for pix in range(0,n_pix):
    #    if not image_mask[pix]: continue # select signal
    #    for win in range(0,n_windows):
    #        for sample in range(0,window_size):
    #            sample_idx = sample + win*window_size
    #            clean_movie_1d[win][pix] +=  waveform[pix,sample_idx]

    n_windows = int(total_samples/n_samples_per_window)
    clean_movie_1d = []
    for win in range(0,n_windows):
        clean_movie_1d += [np.zeros_like(clean_image_1d)]
    for pix in range(0,n_pix):
        if not image_mask[pix]: continue # select signal
        for win in range(0,n_windows):
            for sample in range(0,n_samples_per_window):
                sample_idx = int(sample + win*n_samples_per_window)
                if sample_idx<0: continue
                if sample_idx>=n_samp: continue
                clean_movie_1d[win][pix] += waveform[pix,sample_idx]

    center_time_window = 0.
    total_weight = 0.
    for win in range(0,n_windows):
        total_weight += np.sum(clean_movie_1d[win][:])
        center_time_window += float(win)*np.sum(clean_movie_1d[win][:])
    center_time_window = round(center_time_window/total_weight)
    print (f'center_time_window = {center_time_window}')
    #center_time_sample = center_time_window*n_samples_per_window

    # better not use this...
    #for win in range(0,n_windows):
    #    movie_mask = tailcuts_clean(geometry,clean_movie_1d[win],boundary_thresh=0.5,picture_thresh=1,min_number_picture_neighbors=2)
    #    for pix in range(0,len(movie_mask)):
    #        if not movie_mask[pix]:
    #            clean_movie_1d[win][pix] = 0.

    n_windows_slim = int(select_samples/n_samples_per_window)
    slim_movie_1d = []
    for win in range(0,n_windows_slim):
        slim_movie_1d += [np.zeros_like(clean_image_1d)]

    for pix in range(0,n_pix):
        for win in range(0,n_windows_slim):
            old_win = int(center_time_window - n_windows_slim/2 + win)
            #old_win = win
            if old_win<0: continue

            #slim_movie_1d[win][pix] = np.log(max(0.,clean_movie_1d[old_win][pix])+1.)
            slim_movie_1d[win][pix] = clean_movie_1d[old_win][pix]

            #for win2 in range(old_win,n_windows):
            #    slim_movie_1d[win][pix] += clean_movie_1d[win2][pix]

    image_max = np.max(slim_movie_1d[:][:])

    xmax = max(geometry.pix_x)/u.m
    xmin = min(geometry.pix_x)/u.m
    ymax = max(geometry.pix_y)/u.m
    ymin = min(geometry.pix_y)/u.m

    whole_movie_1d = []
    for win in range(0,n_windows_slim):

        #movie_mask = tailcuts_clean(geometry,clean_movie_1d[win],boundary_thresh=0.5,picture_thresh=1.0,min_number_picture_neighbors=2)
        #for pix in range(0,len(movie_mask)):
        #    if not movie_mask[pix]:
        #        clean_movie_1d[win][pix] = 0.

        clean_movie_2d = geometry.image_to_cartesian_representation(slim_movie_1d[win])
        remove_nan_pixels(clean_movie_2d)

        #image_mask_2d = geometry.image_to_cartesian_representation(image_mask)
        #smooth_movie_2d = image_smooth(clean_movie_2d,image_mask_2d)
        #shift_movie_2d = image_translation(smooth_movie_2d, round(float(shift_pix_y)), round(float(shift_pix_x)))
        shift_movie_2d = image_translation(clean_movie_2d, round(float(shift_pix_y)), round(float(shift_pix_x)))
        rotate_movie_2d = image_rotation(shift_movie_2d, angle*u.rad)
        rotate_movie_1d = geometry.image_from_cartesian_representation(rotate_movie_2d)
        eco_movie_1d = image_cutout(geometry, rotate_movie_1d)
        eco_movie_2d = geometry.image_to_cartesian_representation(rotate_movie_1d)
        remove_nan_pixels(eco_movie_2d)

        whole_movie_1d.extend(eco_movie_1d)
        #print (f'len(whole_movie_1d) = {len(whole_movie_1d)}')

        #if image_size>image_size_cut and make_plots:

        #    print ('make data movie...')
        #    fig.clf()
        #    axbig = fig.add_subplot()
        #    label_x = 'X'
        #    label_y = 'Y'
        #    axbig.set_xlabel(label_x)
        #    axbig.set_ylabel(label_y)
        #    im = axbig.imshow(rotate_movie_2d,origin='lower',extent=(xmin,xmax,ymin,ymax),vmin=0.,vmax=image_max)
        #    #im = axbig.imshow(rotate_movie_2d,origin='lower',extent=(xmin,xmax,ymin,ymax), norm=colors.LogNorm())
        #    cbar = fig.colorbar(im)
        #    line_x = np.linspace(xmin, xmax, 100)
        #    line_y = -(0.*line_x + 0.05)
        #    axbig.plot(line_x,line_y,color='w',alpha=0.3,linestyle='dashed')
        #    line_x = np.linspace(xmin, xmax, 100)
        #    line_y = -(0.*line_x - 0.05)
        #    axbig.plot(line_x,line_y,color='w',alpha=0.3,linestyle='dashed')
        #    fig.savefig(f'{ctapipe_output}/output_plots/evt{event_id}_tel{tel_id}_movie{win}_data.png',bbox_inches='tight')
        #    axbig.remove()


    if image_size>image_size_cut and make_plots:

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
        txt = axbig.text(-0.35, 0.26, 'light cone = %0.2e'%(lightcone), fontdict=font)
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

    return lightcone, image_moment_array, whole_movie_1d

