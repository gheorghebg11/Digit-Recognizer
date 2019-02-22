# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 10:48:42 2019
CNN for Digit Recognizer - MNIST - Data Augmentation module
"""

import math
import random
import os
from datetime import datetime
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from skimage import transform

np.set_printoptions(precision=2, suppress=True, threshold=np.nan)

#####################################################
#### Step 1 : Some global var and the Architecture
#####################################################
########## Some initialization, do NOT change
nbr_classes = 10
size_images = (28,28)

datetime_now = str(datetime.now()).replace('-','').replace(':','').replace(' ','')
datetime_now = datetime_now[:datetime_now.find('.')]

########## Paths
data_dir = os.path.join(os.getcwd(), 'data')
#data_dir = os.path.abspath("C:/MLdata/DigitRecdata")

data_path = 'train'

########## Some Variables for the Architecture
# Data augmentation
multiply_by = 1 #by a factor of how much we augment the amount of data

max_rotation_angle = 20
max_scalling_factor = 1.2
max_shift_pixels = 3

# Use all of the data ?
percentage_data_to_use = 100

##### CV
batch_size_cv = 128 # this is for the NN, ensuring that the # of CV examples is a multiple of batch_size_cv
random_split_cv = True
cv_set_fraction = 0.2

# Visualization
nbr_samples_to_visualize = 5

######### Some Functions
def transform_image(img):
	
	angle = random.uniform(-max_rotation_angle, max_rotation_angle)
	scaling_factor = random.uniform(1.0 / max_scalling_factor, max_scalling_factor)
	shift_v = random.randint(-max_shift_pixels, max_shift_pixels)
	shift_h = random.randint(-max_shift_pixels, max_shift_pixels)
	
	# rescale
	rescaled_im = transform.rescale(img.reshape(28,28), scaling_factor, anti_aliasing=True, multichannel =False, mode='constant', preserve_range=True)
	diff_to_28 = rescaled_im.shape[0] - 28

	if diff_to_28 < 0:
		new_im = np.pad(rescaled_im, pad_width=[[-math.ceil(diff_to_28/2),-math.floor(diff_to_28/2)], [-math.ceil(diff_to_28/2),-math.floor(diff_to_28/2)]], mode='edge')
	elif diff_to_28 > 0:
		new_im = rescaled_im[math.ceil(diff_to_28/2) : 28 + math.ceil(diff_to_28/2), math.ceil(diff_to_28/2) : 28 + math.ceil(diff_to_28/2)]
	else:
		new_im = rescaled_im

	# rotate
	new_im = transform.rotate(new_im, angle)

	# shift
	new_im = np.pad(new_im, pad_width=[[max_shift_pixels,max_shift_pixels], [max_shift_pixels,max_shift_pixels]], mode='edge')[max_shift_pixels+shift_v : 28+max_shift_pixels+shift_v, max_shift_pixels+shift_h : 28+max_shift_pixels+shift_h]

	return new_im.reshape(28*28)
	

def augment_data(xy_data, multiply_by=10):
	
	nbr_ex = xy_data.shape[0]
	batch_size = nbr_ex // 5
	while nbr_ex % batch_size != 0:
		batch_size= batch_size -1
		
	print(f'Starting shape is {xy_data.shape}, we augment in batches of {batch_size}')
	
	for k in range(1,multiply_by):
		print(f'Starting to add augmented data nbr {k} / {multiply_by-1}')
		for i in range(nbr_ex // batch_size):
			
			new_data = np.apply_along_axis(transform_image, 1, xy_data.iloc[i*batch_size:(i+1)*batch_size,1:])
			labels = xy_data.iloc[i*batch_size:(i+1)*batch_size,0].values
			xy_data = xy_data.append(pd.DataFrame(np.concatenate([labels.reshape(labels.shape[0],1), new_data], axis=1), columns=xy_data.columns), ignore_index = True)
			print(f'Concat done up to {xy_data.shape[0]} / {nbr_ex*multiply_by}' )

		#rescaled_im = transform.rescale(x_data.drop('label', axis=1).iloc[i].values.reshape(28,28), scaling_factor, anti_aliasing=True, multichannel =False, mode='constant')
	return xy_data

def split_train_test(xy_data, random_split, cv_set_fraction):
	nbr_total_ex = xy_data.shape[0]

	nbr_cv_ex = int(nbr_total_ex*cv_set_fraction)
	nbr_cv_ex -= (nbr_cv_ex % batch_size_cv)

	nbr_train_ex = nbr_total_ex - nbr_cv_ex

	if random_split:
		xy_data = xy_data.sample(frac=1.0)

	return [xy_data.iloc[:nbr_train_ex], xy_data.iloc[nbr_train_ex:]]

# input x_data, y_labels:pandas df
def visualize_samples(nbr_samples_to_visualize, xy_data):
	print(f'\nVisualizing {nbr_samples_to_visualize} image(s) :')

	for i in range(nbr_samples_to_visualize):
		random_sample = random.randint(0,xy_data.shape[0])

		image = np.array(xy_data.iloc[random_sample,1:]).reshape(28,28)

		fig, ax = plt.subplots(1,2, figsize=(15,4))

		ax[0].plot(image.reshape(28*28))
		ax[0].set_title('784 x 1 pixels value')
		ax[1].imshow(image, cmap='gray')
		ax[1].set_title('28 x 28 pixels')

		plt.show()


########### THE ACTION


filename = data_path + '.csv'
print(f'Loading {filename}')
x_csv = pd.read_csv(os.path.join(data_dir,filename))
print(f'Data loaded')


nbr_ex = int(percentage_data_to_use * x_csv.shape[0] / 100)
x_csv = x_csv.iloc[:nbr_ex]

x_train, x_cv = split_train_test(x_csv, random_split_cv, cv_set_fraction)
print(f'\nData split in traint/test with ratio {1-cv_set_fraction}/{cv_set_fraction}')

print(f'\nAugmenting the training data:')
x_train_aug = augment_data(x_train, multiply_by)
print('Training data augmented')

print(f'\nAugmenting the test data:')
x_cv_aug = augment_data(x_cv, multiply_by)
print('Testing data augmented')



model_code = 'x' + str(multiply_by) + '_' + str(percentage_data_to_use) + '%_' + str(max_rotation_angle) + '_' + str(max_shift_pixels)+ '_' + str(max_scalling_factor) + '_'+ str(cv_set_fraction) 

filename_save_train = 'train_' + model_code + '.csv'
print(f'\nSaving {filename_save_train} in progress (could take some time)')
x_train_aug.to_csv(os.path.join(data_dir,filename_save_train), float_format='%.0f', index=False)

filename_save_cv = 'train_' + model_code + '_cv.csv'
print(f'\nSaving {filename_save_cv} in progress (could take some time)')
x_cv_aug.to_csv(os.path.join(data_dir,filename_save_cv), float_format='%.0f', index=False)

filename_save_txt = 'train_' + model_code + '_details.txt'
with open(os.path.join(data_dir,filename_save_txt), 'w') as notepad_file:
	notepad_file.write(f'Data augmented on {datetime.now()}')
	
	notepad_file.write(f'\n- We kept {percentage_data_to_use}% of the starting data.')
	notepad_file.write(f'\n- The data has been split in train/cv, where cv is {cv_set_fraction*100}% of the data.')
	if multiply_by != 1:
		notepad_file.write(f'\n- Both sets have been augmented by a factor of x{multiply_by} with the same parameters:')
		notepad_file.write(f'\n  - Max rotation angle is {max_rotation_angle}')
		notepad_file.write(f'\n  - Max scaling factor is {max_scalling_factor}')
		notepad_file.write(f'\n  - Max shift (vertical and horizontal) is {max_shift_pixels}')

visualize_samples(nbr_samples_to_visualize, x_train_aug)