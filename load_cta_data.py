import os
import subprocess
import glob

import time
import math
import numpy as np
from astropy import units as u

from ctapipe import utils
from ctapipe.utils.datasets import get_dataset_path
from ctapipe.calib import CameraCalibrator
from ctapipe.image import hillas_parameters, tailcuts_clean, ImageProcessor
from ctapipe.reco import ShowerProcessor
from ctapipe.io import EventSource, SimTelEventSource, HDF5TableWriter
from ctapipe.visualization import CameraDisplay

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

def smooth_map(image_data,xaxis,yaxis,mode):

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
                    image_smooth[idx_y1,idx_x1] += old_content*scale

    return image_smooth

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

def clean_image(image_data,image_mask):

    num_rows, num_cols = image_data.shape
    image_clean = np.zeros_like(image_data)

    for x_idx in range(0,num_cols):
        for y_idx in range(0,num_rows):
            if image_data[y_idx,x_idx]==0.: continue
            if image_mask[y_idx,x_idx]!=1: 
                image_clean[y_idx,x_idx] = 0.
            else:
                image_clean[y_idx,x_idx] = image_data[y_idx,x_idx]

    return image_clean

def find_mask(image_data):

    image_mask = np.zeros_like(image_data)

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
            if significance>5.:
                image_mask[y_idx,x_idx] = 1

    return image_mask

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
            image_rotate[rot_row,rot_col] += input_image_2d[y_idx,x_idx]

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

def load_training_samples(training_sample_path, is_training, min_energy=0.1, max_energy=1000., one_evt=1e10):

    id_list = []
    truth_shower_position_matrix = []
    cam_axes = []
    telesc_position_matrix = []
    big_image_matrix = []
    big_param_matrix = []

    for path in range(0,len(training_sample_path)):
    
        source = SimTelEventSource(training_sample_path[path], focal_length_choice='EQUIVALENT')
    
        # Explore the instrument description
        subarray = source.subarray
        #print (subarray.to_table())
    
        ob_keys = source.observation_blocks.keys()
        run_id = list(ob_keys)[0]
    
        tel_pointing_alt = float(source.observation_blocks[run_id].subarray_pointing_lat/u.rad)
        tel_pointing_az = float(source.observation_blocks[run_id].subarray_pointing_lon/u.rad)
        if tel_pointing_alt>1.*math.pi:
            tel_pointing_alt = tel_pointing_alt-2.*math.pi
        if tel_pointing_az>1.*math.pi:
            tel_pointing_az = tel_pointing_az-2.*math.pi
        
        # Apply some calibration and trace integration
        calib = CameraCalibrator(subarray=subarray)
        image_processor = ImageProcessor(subarray=subarray)
        shower_processor = ShowerProcessor(subarray=subarray)
    
        evt_idx = 0
        for event in source:
        
            event_id = event.index['event_id']
            if one_evt!=1e10:
                if event_id!=one_evt: continue
    
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
            image_processor(event)
            shower_processor(event)
            ranked_tel_key_array = rank_brightest_telescope(event.r0.tel)

            stereo = event.dl2.stereo.geometry["HillasReconstructor"]
            hillas_shower_alt = 0.
            hillas_shower_az = 0.
            hillas_shower_height = 0.
            hillas_shower_core_x = 0.
            hillas_shower_core_y = 0.
            if stereo.is_valid:
                hillas_shower_alt = float(stereo.alt/u.rad)
                hillas_shower_az = float(stereo.az/u.rad)
                hillas_shower_height = float(stereo.h_max/u.m)
                hillas_shower_core_x = float(stereo.core_x/u.m)
                hillas_shower_core_y = float(stereo.core_y/u.m)
                print ('Hillas Reconstruction is successful.')
            else:
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

                dl1tel = event.dl1.tel[tel_key]
                noisy_image = dl1tel.image
        
                # Define a camera geometry
                geom = subarray.tel[tel_key].camera.geometry
                # The CameraGeometry has functions to convert the 1d image arrays to 2d arrays and back to the 1d array
                analysis_image_square = geom.image_to_cartesian_representation(true_image)
                #analysis_image_square = geom.image_to_cartesian_representation(noisy_image)
                #if is_training:
                #    analysis_image_square = geom.image_to_cartesian_representation(true_image)
                num_rows, num_cols = analysis_image_square.shape
                for x_idx in range(0,num_cols):
                    for y_idx in range(0,num_rows):
                        if math.isnan(analysis_image_square[y_idx,x_idx]): analysis_image_square[y_idx,x_idx] = 0.
    
                x_axis, y_axis = get_cam_coord_axes(geom,analysis_image_square)

                #if not is_training:
                #    # image cleaning
                #    analysis_image_smooth = smooth_map(analysis_image_square,x_axis,y_axis,50.)
                #    image_mask = np.zeros_like(analysis_image_smooth)

                #    image_mask = find_mask(analysis_image_smooth)
                #    renormalize_background(analysis_image_smooth,image_mask)
                #    image_mask = find_mask(analysis_image_smooth)

                #    renormalize_background(analysis_image_square,image_mask)
                #    analysis_image_square = clean_image(analysis_image_square,image_mask)

    
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

                analysis_image_recenter = np.zeros_like(analysis_image_square)
                if is_training:
                    shift_x = -evt_cam_x
                    shift_y = -evt_cam_y
                    analysis_image_recenter = image_translation(analysis_image_square, x_axis, y_axis, shift_x, shift_y)
                else:
                    analysis_image_recenter = analysis_image_square

                analysis_image_rotate = np.zeros_like(analysis_image_square)
                if is_training:
                    angle_rad = -np.arctan2(evt_impact_y,evt_impact_x)
                    analysis_image_rotate = image_rotation(analysis_image_recenter, x_axis, y_axis, angle_rad)
                else:
                    analysis_image_rotate = analysis_image_recenter
    
                analysis_image_rotate_1d = geom.image_from_cartesian_representation(analysis_image_rotate)
                
                #fig.clf()
                #axbig = fig.add_subplot()
                #label_x = 'X'
                #label_y = 'Y'
                #axbig.set_xlabel(label_x)
                #axbig.set_ylabel(label_y)
                #axbig.imshow(analysis_image_square,origin='lower')
                #fig.savefig(f'{ctapipe_output}/output_plots/training_image_evt{event_id}_tel{tel_idx}_original.png',bbox_inches='tight')
                #axbig.remove()
    
                #fig.clf()
                #axbig = fig.add_subplot()
                #label_x = 'X'
                #label_y = 'Y'
                #axbig.set_xlabel(label_x)
                #axbig.set_ylabel(label_y)
                #axbig.imshow(analysis_image_recenter,origin='lower')
                #fig.savefig(f'{ctapipe_output}/output_plots/training_image_evt{event_id}_tel{tel_idx}_recenter.png',bbox_inches='tight')
                #axbig.remove()
    
                #fig.clf()
                #axbig = fig.add_subplot()
                #label_x = 'X'
                #label_y = 'Y'
                #axbig.set_xlabel(label_x)
                #axbig.set_ylabel(label_y)
                #axbig.imshow(analysis_image_rotate,origin='lower')
                #fig.savefig(f'{ctapipe_output}/output_plots/training_image_evt{event_id}_tel{tel_idx}_rotate.png',bbox_inches='tight')
                #axbig.remove()
                #exit()
    
                evt_truth_energy = shower_energy
                evt_truth_impact = pow(pow(evt_impact_x,2)+pow(evt_impact_y,2),0.5)
                evt_truth_height = shower_height
                evt_truth_x_max = shower_x_max

                id_list += [[run_id,event_id,tel_key,subarray]]
                telesc_position_matrix += [[tel_pointing_alt,tel_pointing_az,tel_x,tel_y,tel_focal_length]]
                big_image_matrix += [analysis_image_rotate_1d]
                #big_param_matrix += [[evt_truth_energy,1./pow(evt_truth_impact/1000.,1),evt_truth_height]]
                big_param_matrix += [[evt_truth_energy/pow(evt_truth_impact/1000.,1)]]
                truth_shower_position_matrix += [[shower_alt,shower_az,shower_core_x,shower_core_y,evt_truth_energy]]
                cam_axes += [[x_axis,y_axis]]

        
            evt_idx += 1

    return id_list, telesc_position_matrix, truth_shower_position_matrix, cam_axes, big_image_matrix, big_param_matrix


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



