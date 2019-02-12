# -*- coding: utf-8 -*-
"""
Spyder Editor
CNN for Digit Recognizer - MNIST
"""

import math
import random
import os
from datetime import datetime
import matplotlib.pyplot as plt
from PIL import Image

import tensorflow as tf
import numpy as np
import pandas as pd

tf.reset_default_graph()
np.set_printoptions(precision=2, suppress=True, threshold=np.nan)

#####################################################
#### Step 1 : Some global var and the Architecture
#####################################################
########## Some initialization, do NOT change
nbr_classes = 10
fil_size_stride_pool = []
nbr_neurons = []

datetime_now = str(datetime.now()).replace('-','').replace(':','').replace(' ','')
datetime_now = datetime_now[:datetime_now.find('.')]

########## Paths
load_model = False
loaded_model_name = 'model_20190210215510'

# Dir
if load_model:
	save_dir = os.path.join(os.getcwd(), 'loaded_' + loaded_model_name + '_' + datetime_now)
else:
	save_dir = os.path.join(os.getcwd(), 'model_' + datetime_now)
os.mkdir(save_dir)
load_model_dir = os.path.join(os.getcwd(), loaded_model_name)
data_dir = os.path.join(os.getcwd(), 'data')
external_pics_dir = os.path.join(os.getcwd(), 'pics')

# Path
load_model_path = os.path.join(load_model_dir, 'model.ckpt')
filename_save_model_archi = os.path.join(save_dir, 'model_details.txt')
filename_save_path = os.path.join(save_dir, 'model.ckpt')
tensorboard_path = os.path.join(save_dir)
filters_layer_0_savepath = os.path.join(save_dir, 'filters.png')
filter_evolution_savepath = os.path.join(save_dir, 'filter_evolution.png')
cross_ent_train_savepath = os.path.join(save_dir, 'cross_entropy_train.png')
cross_ent_cv_savepath = os.path.join(save_dir, 'cross_entropy_cv.png')


########## Some Global var for Training/Testing
use_tboard = False
use_cv = True

save_model = False
save_model_properties = True
predict_test_set = False
predict_from_external_pics = False
plot_wrong_classifications_at_end_training = False # change that to the number of batches to do

ask_for_more_training = True

nbr_training_steps =  500
use_epochs_instead_of_it = False
nbr_epochs = 1

########## Visualization
nbr_samples_to_visualize = 0
plot_distrib = False # plot distribution of classes, as well as values of a random pixel

########## Monitoring Learning & Plotting stuff
size_plot_per_filter = 1.5
nbr_filters_per_row = 10

# plot accuracy
freq_plot_acc = 5

# plot cross-ent
freq_plot_cross_ent = freq_plot_acc * 2

# plot filters in layer 0
plot_filters_layer_0 = True
nbr_filters_layer_0 = None # None means that we look at all of them
freq_plot_filters = freq_plot_acc * 10
save_png_filters_layer_0_at_end = True

# plot the evolution of one  filter
plot_filter_evolution = True
freq_plot_filter_evolution = freq_plot_acc * 10
save_png_filter_evolution_at_end = True

########## Some Variables for the Architecture
# Initialization weights
use_Xavier_init = True

# Batch
batch_size = 128 # in [16,32, 64, 128, 256]
shuffle_dataset_every_epoch = True

# Normalization
use_input_normalization = False
use_batch_normalization = True

# Regularization
use_L2reg_in_conv = False
use_L2reg_in_full = True
beta = 0.1 # reg variable [0.01, 0.1]

keep_prob_training = 0.7 #dropout variable, set to 1 to ignore dropout, only on FC-layers

# Optimizer
choices_gd = ['SGD', 'Adam']
optimizer_gd = 'Adam'
learning_rate = 0.1

# The convolution layers (nbr of filters, filter size, (v_stride,h_stride), 2x2maxpool)
fil_size_stride_pool.append([32, [3,3], [1,1], True])

# The full layers (nbr neurons)
nbr_neurons.append(nbr_classes)

# CV
cv_set_fraction = 0.01
is_cv_split_random = False
batch_size_cv = batch_size # so that I don't have to divide the cross-ent by batch_size and not loose precision
cv_eval_nbr_iterations = freq_plot_acc * 10

#####################################################
##### Step 2 - Process Data
#####################################################
# input x_data:pandas df
def preprocess_training_data(x_data, standardize, nbr_samples_to_visualize = 0):

	x_data = x_data / 255

	if standardize:
		x_mean = np.mean(x_data, axis=0)
		x_std = np.std(x_data, axis=0)
		np.savetxt(os.path.join(save_dir, 'x_mean.txt'), x_mean, fmt = '%.8f')
		np.savetxt(os.path.join(save_dir, 'x_std.txt'), x_std, fmt = '%.8f')

		x_data = (x_data - x_mean) / (x_std + 0.01)
		x_data.fillna(0 , inplace=True)

		if nbr_samples_to_visualize!= 0:
			_, ax = plt.subplots(1,2, figsize=(9,4))
			ax[0].imshow(np.array(x_mean).reshape(28,28), cmap='gray')
			ax[0].set_title('The mean image is:')
			ax[1].imshow(np.array(x_std).reshape(28,28), cmap='gray')
			ax[1].set_title('The std-dev image is')
			plt.show()

			for i in range(28*28):
				if x_std[i] == 0 and x_mean[i] != 0:
					print(f'The index {i} has std = 0 but mean != 0.')
	else:
		x_mean = 0
		x_std = 1

	return x_data, x_mean, x_std

def preprocess_cv_test_data(x_data, x_mean, x_std):

	x_data = x_data / 255
	x_data = (x_data - x_mean) / (x_std + 0.01)
	x_data.fillna(0 , inplace=True)
	return x_data

def invert_colors(x_data):

	return x_data

#####################################################
##### Step 3 - Visualize & Explore Data
#####################################################
# input x_data, y_labels:pandas df
def visualize_samples(nbr_samples_to_visualize, x_data, y_labels, is_normalized):
	print(f'Visualizing {nbr_samples_to_visualize} image(s) :')

	for i in range(nbr_samples_to_visualize):
		random_sample = random.randint(0,x_data.shape[0])

		fig, ax = plt.subplots(1,2, figsize=(15,4))

		ax[0].plot(np.array(x_data.iloc[random_sample]).reshape(28*28))
		ax[0].set_title('784 x 1 pixels value')
		ax[1].imshow(np.array(x_data.iloc[random_sample]).reshape(28,28), cmap='gray')
		ax[1].set_title('28 x 28 pixels')

		plt.show()

		if is_normalized:
			print(f'Image {i+1} / {nbr_samples_to_visualize}: The data has be normalized before plotting. The true label is {y_labels.iloc[random_sample].idxmax()}')
		else:
			print(f'Image {i+1} / {nbr_samples_to_visualize}: The true label is {y_labels.iloc[random_sample].idxmax()}')

def plot_distribution_classes(y_labels, type_dataset):
	print(f'Here is the distribution of the classes, for the {type_dataset} set')

	y_distrib = y_labels.sum(axis=0)
	plt.bar(x=y_distrib.index, height=y_distrib, width=0.8)
	plt.show()
	plt.close()

def plot_distribution_per_pixel(x_data, pixel_coord, type_dataset, plot_all = False):
	print(f'Here is the distribution of the intensities of the pixel {pixel_coord}, for the {type_dataset} set')

	if pixel_coord[0] < 0 or pixel_coord[0] >= 27 or pixel_coord[1] < 0 or pixel_coord[1] >= 27:
		new_pixel_coord = [x % 28 for x in pixel_coord]
		print(f'ERROR: there is no pixel with coord {pixel_coord}, we will print {new_pixel_coord}')
		pixel_coord = new_pixel_coord

	plt.hist(x_data[f'pixel{pixel_coord[0]*28 + pixel_coord[1]}'], rwidth = 0.8, log = True)
	plt.show()

	if plot_all:
		_, ax = plt.subplots(28, 28, squeeze= False, figsize=(25,25))
		for k in range(28*28):
			nbr_cols = int(k / 28)
			nbr_row = k % 28
			ax[nbr_cols][nbr_row].hist(x_data[f'pixel{k}'], rwidth = 0.8, log = True)
			ax[nbr_cols][nbr_row].xaxis.set_visible(False)
			ax[nbr_cols][nbr_row].yaxis.set_visible(False)
		plt.show()
	plt.close()

#####################################################
##### Step 4 - Some Preliminary functions for the NN
#####################################################
## Create variables
def weight_variable(shape, use_reg, fan_in):
	stddev = 0.1
	if use_Xavier_init:
		stddev = math.sqrt(2/fan_in)
	initial = tf.truncated_normal(shape, stddev = stddev, name='weight_init')

	regularizer = None
	if use_reg:
		regularizer = tf.contrib.layers.l2_regularizer(scale=beta)

	weights = tf.get_variable(name = 'weights', initializer = initial, dtype=tf.float32, regularizer = regularizer)
	return weights

def bias_variable(shape, init_value=0.1):
	initial = tf.constant(init_value, shape = shape, name='bias_init')
	return tf.get_variable(name = 'bias', initializer = initial, dtype=tf.float32)

## NN layers
def conv_layer(nbr_layer, input, nbr_filters, filter_size, stride, maxpool, use_batch_normalization):
	name_layer = 'conv_layer_' + str(nbr_layer)
	with tf.variable_scope(name_layer):
		fan_in = input.get_shape().as_list()[1] * input.get_shape().as_list()[2]
		W = weight_variable([filter_size[1], filter_size[0], input.get_shape().as_list()[3], nbr_filters], use_L2reg_in_conv, fan_in)
		b = bias_variable([nbr_filters], 0.1)

		total_param = np.prod(W.shape) + np.prod(b.shape)
		print(f'Created convolutional layer {nbr_layer} with {nbr_filters} filters of size {filter_size}, and {total_param} parameters')

		if use_batch_normalization:
			input = tf.layers.batch_normalization(input, training = is_train, name='batch_norm_inconv')
		input = tf.nn.conv2d(input, W, strides = [1,stride[0],stride[1],1], padding ='SAME', name='conv2d') + b
		if maxpool:
			input = tf.nn.max_pool(input, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='maxpool_inconv')
		return tf.nn.relu(input)

def full_layer(nbr_layer, input, nbr_output, use_batch_normalization, activation = True):
	name_layer = 'full_layer_' + str(nbr_layer)
	with tf.variable_scope(name_layer):
		fan_in = input.get_shape().as_list()[1]
		W = weight_variable([input.get_shape().as_list()[1], nbr_output], use_L2reg_in_full, fan_in)
		b = bias_variable([nbr_output], 0.1)

		total_param = np.prod(W.shape) + np.prod(b.shape)
		print(f'Created fully-connec layer {nbr_layer} with {nbr_output} neurons, and {total_param} parameters')

		if use_batch_normalization:
			input = tf.layers.batch_normalization(input, training = is_train)
		if activation:
			return tf.nn.relu(tf.nn.dropout(tf.matmul(input,W) + b, keep_prob = keep_prob))
		else:
			return tf.nn.dropout(tf.matmul(input,W) + b, keep_prob = keep_prob)

## Create batch
def extract_batch(x_data, y_data, batch_nbr, batch_size):
	nbr_training_ex = x_data.shape[0]
	batch_start = batch_nbr * batch_size % nbr_training_ex
	batch_end = (batch_nbr + 1) * batch_size % nbr_training_ex

	if batch_start < batch_end:
		return [np.array(x_data[batch_start:batch_end]).reshape(batch_size, 28,28,1), y_data[batch_start:batch_end]]
	else:
		#print(f'Beginining of a new epoch, batch_end is {batch_end}, hopefully creating the batches is done right...')
		if batch_end == 0:
			return [np.array(x_data[batch_start:]).reshape(batch_size, 28,28,1), y_data[batch_start:]]
		else:
			return [np.vstack([x_data[batch_start:], x_data[:batch_end]]).reshape(batch_size, 28,28,1), pd.concat([y_data.iloc[batch_start:], y_data.iloc[:batch_end]], ignore_index = True) ]

def shuffle_dataset(x_data, y_label, random_seed = np.random.randint(100000)):
	return [x_data.sample(frac=1.0, random_state=random_seed), y_label.sample(frac=1.0, random_state=random_seed)]

def split_train_test(x_data, y_label, random_split, cv_set_fraction = 0.2):
	nbr_total_ex = x_data.shape[0]

	nbr_cv_ex = int(nbr_total_ex*cv_set_fraction)
	nbr_cv_ex -= (nbr_cv_ex % batch_size_cv)

	nbr_train_ex = nbr_total_ex - nbr_cv_ex

	if random_split:
		x_data = x_data.sample(frac=1.0)
		y_label = y_label.sample(frac=1.0)

	return [x_data[:nbr_train_ex], y_label[:nbr_train_ex], x_data[nbr_train_ex:], y_label[nbr_train_ex:]]

def extract_batch_test(x_data, batch_nbr, batch_size):
	nbr_test_ex = x_data.shape[0]
	batch_start = batch_nbr * batch_size % nbr_test_ex
	batch_end = (batch_nbr + 1) * batch_size % nbr_test_ex

	if batch_start < batch_end:
		return np.array(x_data[batch_start:batch_end]).reshape(batch_size, 28,28,1)
	else:
		if batch_end == 0:
			return np.array(x_data[batch_start:]).reshape(batch_size, 28,28,1)
		else:
			return np.vstack([x_data[batch_start:], x_data[:batch_end]]).reshape(batch_size, 28,28,1)

# Plotting some filters from the CNN
def plot_filters(iteration_step, nbr_conv_layer=0, nbr_filters=None, save = False):

	if nbr_conv_layer >= len(fil_size_stride_pool):
		print(f'ERROR: There is no conv layer {nbr_conv_layer} (there are only {len(fil_size_stride_pool)} conv layers, starting at 0), so we cant plot it')
	else:
		weights = sess.run('conv_layers/conv_layer_' + str(nbr_conv_layer) +'/weights:0')

		if nbr_filters == None: # if given None, then print all the filters
			nbr_filters = weights.shape[3]

		if nbr_filters > weights.shape[3]: # need at least one line of filters (else could change the number of filters per line to print too)
			print('ERROR in printing the filters, you want to print {nbr_filters}, but layer {nbr_conv_layer} has only {weights.shape[3]}.')
			nbr_filters = weights.shape[3]

		global nbr_filters_layer_0
		nbr_filters_layer_0 = nbr_filters
		nbr_cols = math.ceil(nbr_filters / nbr_filters_per_row)
		filters_size = fil_size_stride_pool[nbr_conv_layer][1][0], fil_size_stride_pool[nbr_conv_layer][1][1]

		fig, ax = plt.subplots(nbr_cols, nbr_filters_per_row, squeeze= False, figsize=(size_plot_per_filter*nbr_filters_per_row, size_plot_per_filter*nbr_cols))
		for k in range(nbr_filters):
			nbr_col = int(k / nbr_filters_per_row)
			nbr_row = k % nbr_filters_per_row
			ax[nbr_col][nbr_row].imshow(weights.reshape(fil_size_stride_pool[nbr_conv_layer][0],1, filters_size[0], filters_size[1])[k][0], cmap='gray')
		plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
		fig.suptitle(f'Printing {nbr_filters} filters of size {filters_size} from the conv layer {nbr_conv_layer} (after {iteration_step} steps)', va='top', fontsize=16)

		if save:
			plt.savefig(fname = filters_layer_0_savepath)
			plt.close() # close it so it doesn't print at the end
			print(f'\nSaved {nbr_filters} filters of size {filters_size} from the conv layer {nbr_conv_layer}')
		else:
			plt.show()


# Plotting the evolution of a filtera from the CNN
def plot_filter_evolution(iteration_step, filter_layer=0, filter_nbr=0, save = False):

	weights = sess.run('conv_layers/conv_layer_' + str(filter_layer) +'/weights:0')
	if save == False:
		filter_history.append(weights.reshape(fil_size_stride_pool[filter_layer][0], 1, fil_size_stride_pool[filter_layer][1][0], fil_size_stride_pool[filter_layer][1][1])[filter_nbr][0])

	nbr_rows = math.ceil(len(filter_history) / nbr_filters_per_row)

	fig, ax = plt.subplots(nbr_rows, nbr_filters_per_row, squeeze = False, figsize=(size_plot_per_filter*nbr_filters_per_row, size_plot_per_filter*nbr_rows))
	for k in range(len(filter_history)):
		nbr_col = int(k / nbr_filters_per_row)
		nbr_row = k % nbr_filters_per_row
		ax[nbr_col][nbr_row].imshow(filter_history[k], cmap='gray')
		ax[nbr_col][nbr_row].set_title(f'iteration {iteration_step* (k+1) // len(filter_history)}')
	plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
	#fig.suptitle(f'The evolution of filter {filter_nbr} in layer {filter_layer} over {len(filter_history)} periods of time ({iteration_step} iterations each)', va='top', fontsize=16) #, y=0.92+ math.exp(-nbr_rows)

	if save and len(filter_history) != 0:
		plt.savefig(fname = filter_evolution_savepath)
		plt.close() # close it so it doesn't print at the end
		print(f'\nSaved the evolution of filter {filter_nbr} in layer {filter_layer} over {len(filter_history)} periods of time ({iteration_step} iterations each)')
	else:
		plt.show()


#####################################################
##### Step 5 - The NN - Architecture, Error Loss, Reg, Opti, Training, etc
#####################################################

### Create the PLaceholders
with tf.name_scope('input'):
	X = tf.placeholder(tf.float32, shape = [None, 28, 28, 1], name='var_x')
	Y = tf.placeholder(tf.float32, shape = [None, 10], name='var_y')
	#y_test = tf.placeholder(tf.float32, shape = [None, nbr_classes], name='var_y_test')

with tf.name_scope('some_var_for_training'):
	keep_prob = tf.placeholder(tf.float32, name ='keep_prob_dropout')
	is_train = tf.placeholder(tf.bool, name='is_train')

### The NN
with tf.variable_scope('conv_layers'):
	conv = [X]
	nbr_conv_layers = len(fil_size_stride_pool)
	for i in range(nbr_conv_layers):
		conv.append(conv_layer(i, conv[i], fil_size_stride_pool[i][0], fil_size_stride_pool[i][1], fil_size_stride_pool[i][2], fil_size_stride_pool[i][3], use_batch_normalization))
	full = [tf.layers.flatten(conv[nbr_conv_layers], name='flattened_layer')]

with tf.variable_scope('full_layers'):
	nbr_hidden_layers = len(nbr_neurons)
	for i in range(nbr_hidden_layers - 1):
		full.append(full_layer(i, full[i], nbr_neurons[i], use_batch_normalization))
	full_last = full_layer(nbr_hidden_layers - 1, full[nbr_hidden_layers-1], nbr_neurons[nbr_hidden_layers -1], use_batch_normalization, activation = False)

with tf.name_scope('regularization'):
	reg_term = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

with tf.name_scope('loss_error'):
	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = Y, logits = full_last)) + sum(reg_term)

with tf.name_scope('training'):
	optimizer = tf.train.GradientDescentOptimizer(learning_rate)
	if optimizer_gd == 'Adam':
		optimizer = tf.train.AdamOptimizer()
	elif optimizer_gd == 'SGD':
		optimizer = tf.train.GradientDescentOptimizer(learning_rate)

	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) # needed in order for batch normalization to its job before training
	with tf.control_dependencies(update_ops):
		train_step = optimizer.minimize(cross_entropy)

with tf.name_scope('perf_metrics'):
	prediction_class = tf.argmax(tf.nn.softmax(full_last), axis = 1)
	nbr_correct_predictions = tf.reduce_sum(tf.cast(tf.equal(prediction_class, tf.cast(tf.argmax(Y, axis = 1), tf.int64)), tf.uint8))

with tf.name_scope('summaries'):
	tf.summary.scalar('cross_entropy', cross_entropy)
	tf.summary.scalar('nbr_correct_pred', nbr_correct_predictions)
	merged = tf.summary.merge_all()

with tf.name_scope('model_saver'):
	saver = tf.train.Saver()

#####################################################
##### Step 6 - Run the NN
#####################################################

### Loading, Preprocessing and Visualizing
if load_model == False:

	# Loading the data
	print('Loading the data')
	x_csv = pd.read_csv(os.path.join(data_dir,'train.csv'))
	y_train = pd.get_dummies(x_csv['label'])
	x_train = x_csv.drop('label', axis=1)
	print('Data loaded')

	# Preprocesing the data
	if use_cv:
		x_train, y_train, x_cv, y_cv = split_train_test(x_train, y_train, is_cv_split_random, cv_set_fraction)

	x_train, x_mean, x_std = preprocess_training_data(x_train, use_input_normalization, nbr_samples_to_visualize)
	nbr_training_ex = x_train.shape[0]

	if use_cv:
		x_cv = preprocess_cv_test_data(x_cv, x_mean, x_std)
		nbr_cv_ex = x_cv.shape[0]

	# Visualization
	if nbr_samples_to_visualize!= 0:
		visualize_samples(nbr_samples_to_visualize, x_train, y_train, use_input_normalization)

	if plot_distrib:
		plot_distribution_classes(y_train, 'train')
		plot_distribution_classes(y_cv, 'cross-validation')
		plot_distribution_per_pixel(x_train, (10,10), 'train')

elif load_model == True:

	# Load x_mean and x_std
	if use_input_normalization:
		x_mean = np.loadtxt(fname = os.path.join(load_model_dir, 'x_mean.txt'), dtype=np.float32)
		x_std = np.loadtxt(fname = os.path.join(load_model_dir, 'x_std.txt'), dtype=np.float32)
	else:
		x_mean = 0
		x_std = 1

### Running the Session
with tf.Session() as sess:

	# PREDICTING FROM OUTSIDE IMAGES
	if load_model == True:
		print('\n------------------------------------------------\n')
		saver.restore(sess, load_model_path)
		print(f'\nSystem loaded from {load_model_path}\n')

		if predict_from_external_pics:
			x_data = []

			for filename in os.listdir(external_pics_dir):
				if filename[-3:] in ['png', 'bmp', 'jpg']:
					#img = plt.imread(os.path.join(external_pics_dir,filename))
					#x_data.append(transform.resize(img, output_shape = [28, 28],  anti_aliasing=True, mode='reflect'))
					img = Image.open(os.path.join(external_pics_dir,filename)).convert("L").resize((28,28))
					x_data.append(np.asarray(img))

			x_data = pd.DataFrame((np.array(x_data) * 255).reshape(-1,28*28))
			x_data = preprocess_cv_test_data(x_data, x_mean, x_std)
			nbr_images_to_predict = x_data.shape[0]

			print_prediction_class = sess.run(prediction_class, feed_dict={X:extract_batch_test(x_data, 0, nbr_images_to_predict), keep_prob:1.0, is_train:False})

			for i in range(nbr_images_to_predict):
				fig, ax = plt.subplots(1,2, figsize=(15,4))
				ax[0].plot(np.array(x_data.iloc[i]).reshape(28*28))
				ax[0].set_title('784 x 1 pixels value')
				ax[1].imshow(np.array(x_data.iloc[i]).reshape(28,28), cmap='gray')
				ax[1].set_title('28 x 28 pixels')
				plt.show()
				print(f'The system predicted the number {print_prediction_class[i]}')

		if predict_test_set:
			print('\nTesting begins')
			x_csv_test = pd.read_csv(os.path.join(data_dir,'test.csv'))
			x_test = preprocess_cv_test_data(x_csv_test, x_mean, x_std)
			y_submission = pd.read_csv(os.path.join(data_dir,'sample_submission.csv'))

			nbr_test_ex = x_test.shape[0]
			batch_size_test = 100 # doesn't really matter, just a matter of balance for being able to print partial resutls
			if nbr_test_ex % batch_size_test:
				print('The size of the batch for the test set is not a divisor of the whole set, fix that to avoid issues at the end of the set')

			# Do the prediction
			for i in range(nbr_test_ex // batch_size_test):
				print_prediction_class_test = sess.run(prediction_class, feed_dict={X:extract_batch_test(x_test, i, batch_size_test), keep_prob:1.0, is_train:False})
				if (i+1) % 20 == 0:
					print(f'Finishing test batch {i+1} / {nbr_test_ex / batch_size_test}')

				for idx,value in enumerate(print_prediction_class_test):
					y_submission['Label'].at[i*batch_size_test + idx] = value

			# Save the prediction
			y_submission.to_csv(os.path.join(save_dir,'submission_' +  datetime_now + '.csv'), index=False)

	elif load_model == False:
		# LOADING
		print('\nTensorFlow session running...')
		sess.run(tf.global_variables_initializer())
		print('Global var loaded')
		if use_tboard:
			train_writer = tf.summary.FileWriter(tensorboard_path ,sess.graph)
			print('Summary Writer Created')
		if use_epochs_instead_of_it:
			nbr_training_steps = nbr_epochs * nbr_training_ex // batch_size

		# SOME GLOBAL VAR FOR PLOTTING
		filter_history = []
		cross_ent_history_train = []
		cross_ent_history_cv = []
		keep_training = True

		# TRAINING
		for i in range(nbr_training_steps):
			epoch = i*batch_size / nbr_training_ex

			# If we arrive at the end of an epoch: Shuffle + ask for more ?
			if (i*batch_size % nbr_training_ex) < batch_size and shuffle_dataset_every_epoch:
				# Ask for more training
				if ask_for_more_training and i > 0:
					while "the answer is invalid":
						reply = str(input('Continue Training ?' + ' (y/n): ')).lower().strip()
						if reply[:1] == 'n':
							keep_training = False
							break
						elif reply[:1] == 'y':
							print('Training Resumed.\n')
							break
						else:
							print('Entry not valid. ')
				if keep_training == False:
					break

				# Shuffle
				x_train,y_train = shuffle_dataset(x_train,y_train)
				print(f'\nBeginning of epoch {int(epoch)}, data set shuffled')

			# Training
			_, summary, print_cross_entropy, print_nbr_correct_pred = sess.run([train_step, merged, cross_entropy, nbr_correct_predictions], feed_dict={(X,Y):extract_batch(x_train, y_train, i, batch_size), keep_prob:keep_prob_training, is_train:True})
			if (i+1) % freq_plot_acc == 0:
				print(f'Epoch {epoch:2.2f} ({i+1}th step) with accuracy {print_nbr_correct_pred} / {batch_size} and cross-ent {print_cross_entropy:5.8f}')
				cross_ent_history_train.append(print_cross_entropy)
			if use_tboard:
				train_writer.add_summary(summary, i)

			if (i+1) % freq_plot_cross_ent == 0:
				plt.figure(figsize=(15,4))
				#plt.ylim(0,0.7)
				plt.plot(cross_ent_history_train, label = 'cross-ent training')
				plt.legend(loc ='upper right')
				plt.title(f'cross-entropy  after {i+1} iterations (epoch {epoch}) of training')
				if cross_ent_history_train[0] > cross_ent_history_train[-1]*5:
					plt.yscale('log')
				plt.show()


			# Plotting Filters from a layer
			if (i+1) % freq_plot_filters == 0 and plot_filters_layer_0:
				plot_filters(i+1, nbr_filters=nbr_filters_layer_0)

			# Plotting the evolution of a Filter
			if (i+1) % freq_plot_filter_evolution == 0 and plot_filter_evolution:
				plot_filter_evolution(i+1)

			# CV
			if use_cv and (i+1) % cv_eval_nbr_iterations == 0:
				print(f'\nPerforming CV after {i+1} iterations, with {nbr_cv_ex} examples (batches of {batch_size_cv})')
				print_cv_nbr_correct_pred_total = 0
				print_cv_cross_entropy_total = 0
				freq_print_cv_info = 10

				if nbr_cv_ex % batch_size_cv != 0:
					print(f'The cv_batch_size does not divide the number of cv_examples, the remainder ones will be ignored')

				for j in range(nbr_cv_ex // batch_size_cv):
					print_cv_cross_entropy, print_cv_nbr_correct_pred = sess.run([cross_entropy, nbr_correct_predictions], feed_dict={(X,Y):extract_batch(x_cv, y_cv, j, batch_size_cv), keep_prob:1.0, is_train:False})

					print_cv_cross_entropy_total += print_cv_cross_entropy
					print_cv_nbr_correct_pred_total += print_cv_nbr_correct_pred

					if (j+1) % freq_print_cv_info == 0:
						print(f'Finishing cv batch {j+1} / {nbr_cv_ex // batch_size_cv} with accuracy {print_cv_nbr_correct_pred} / {batch_size_cv} and average cross-ent {print_cv_cross_entropy_total / (j+1):5.8f}')

				cross_ent_history_cv.append(print_cv_cross_entropy_total/ (nbr_cv_ex // batch_size_cv))
				print(f'CV accuracy avg {print_cv_nbr_correct_pred_total} / {nbr_cv_ex} = {print_cv_nbr_correct_pred_total / nbr_cv_ex:0.4f} and cross-ent avg {cross_ent_history_cv[-1]:5.8f}\n')
				plt.figure(figsize=(15,4))
				plt.plot(cross_ent_history_cv, label = 'cross-ent CV')
				plt.legend(loc ='upper right')
				if cross_ent_history_cv[0] > cross_ent_history_cv[-1]*5:
					plt.yscale('log')
				plt.title(f'cross-entropy for CV set after {i+1} iterations (epoch {epoch}) of training')
				plt.show()


		# END OF TRAINING
		print(f'\n--------------------------------------------------------\nTraining finished after {epoch:2.2f} epochs and {nbr_training_steps} training steps.\n--------------------------------------------------------')
		#os.makedirs(save_dir) #automatically created by tboard
		print(f'\nSaving model in {save_dir}')

		# SAVE FILTERS & Cross-Entropy
		# Training Cross-Entropy
		plt.figure(figsize=(15,4))
		plt.plot(cross_ent_history_train, label = 'cross-ent training')
		plt.legend(loc ='upper right')
		plt.title(f'cross-entropy  after {nbr_training_steps} iterations (epoch {epoch}) of training')
		if cross_ent_history_train[0] > cross_ent_history_train[-1]*5:
			plt.yscale('log')
		plt.savefig(fname = cross_ent_train_savepath)
		plt.close() # close it so it doesn't print at the end
		print(f'\nSaved graph for the training cross-entropy')

		# CV Cross-Entropy
		plt.figure(figsize=(15,4))
		plt.plot(cross_ent_history_cv, label = 'cross-ent CV')
		plt.legend(loc ='upper right')
		plt.title(f'cross-entropy after {nbr_training_steps} iterations (epoch {epoch}) of training')
		if cross_ent_history_cv[0] > cross_ent_history_cv[-1]*5:
			plt.yscale('log')
		plt.savefig(fname = cross_ent_cv_savepath)
		plt.close() # close it so it doesn't print at the end
		print(f'\nSaved graph for the CV cross-entropy')

		# Filters
		if save_png_filters_layer_0_at_end:
			plot_filters(i+1, nbr_filters=nbr_filters_layer_0, save = True)

		if save_png_filter_evolution_at_end:
			plot_filter_evolution(i+1, save = True)

		plt.close()
		# PRINT MISCLASSIFIED
		if plot_wrong_classifications_at_end_training:
			print_prediction_class = sess.run(prediction_class, feed_dict={X:extract_batch(x_train, y_train, nbr_training_steps, batch_size)[0], keep_prob:1.0, is_train:False})
			correct_labels = extract_batch( x_train, y_train, nbr_training_steps, batch_size)[1].idxmax(axis =1)

			predictions_match = (print_prediction_class == correct_labels)

			wrong_predictions = [correct_labels.index[idx] for idx, pred in enumerate(predictions_match) if pred == False]
			nbr_wrong_pred = len(wrong_predictions)

			print(f'\nHere are the {nbr_wrong_pred} wrong predictions out of {batch_size} after training through {batch_size*nbr_training_steps} images, {batch_size*nbr_training_steps / x_train.shape[0]:2.2f} epochs')

			if nbr_wrong_pred != 0:
				for i in range(nbr_wrong_pred):
					plt.imshow(np.array(x_train.iloc[wrong_predictions[i]]).reshape(28,28), cmap='gray')
					print(f'\nWrong prediction nbr {i+1} / {nbr_wrong_pred} : true label is {y_train.iloc[wrong_predictions[i]].idxmax()}, predicted as {print_prediction_class[wrong_predictions[i] % batch_size]}')
					plt.show()

		# SAVE THE MODEL
		if save_model:
			save_path = saver.save(sess, filename_save_path)
			print('\nModel saved')

		# SAVE PROPERTIES OF THE MODEL
		if save_model_properties:
			with open(filename_save_model_archi, 'w') as notepad_file:
				notepad_file.write(f'Model from {datetime.now()}, trained on {i*batch_size} examples for {nbr_training_steps} iterations ({epoch} epochs).')
				notepad_file.write(f'\n\nThe weights were initialized using Xavie initialization : {use_Xavier_init}.')
				notepad_file.write(f'\n\nTraining: \n {nbr_training_steps} iterations with {optimizer_gd} (rate if SGD {learning_rate}) \
						\n batches of {batch_size} images with shuffle every epoch : {shuffle_dataset_every_epoch} \
						\n images standardized : {use_input_normalization} \
						\n use batch normalization : {use_batch_normalization}')
				notepad_file.write(f'\n\nArchitecture: {len(fil_size_stride_pool)} conv layers followed by {len(nbr_neurons)} fully connected layers, all have ReLU activation.\n')
				notepad_file.write('More precisely and in this order the hidden layers are:')
				for j in range(len(fil_size_stride_pool)):
					notepad_file.write(f'\n - conv layer {j+1} with {fil_size_stride_pool[j][0]} filters of size {fil_size_stride_pool[j][1]} with stride {fil_size_stride_pool[j][2]}, 2x2Maxpool : {fil_size_stride_pool[j][3]} and L2reg : {use_L2reg_in_conv} (with coeff {beta})')
				for j in range(len(nbr_neurons)):
					notepad_file.write(f'\n - fully connected layer {j+1} with {nbr_neurons[j]} neurons, dropout with rate {keep_prob_training} and L2reg : {use_L2reg_in_full} (with coeff {beta})')
				notepad_file.write(f'\n\nLast Cross-entropy: {print_cross_entropy} and Last Accuracy {print_nbr_correct_pred} / {batch_size}.')
				if save_model:
					model_filepath = os.path.join(save_dir, 'model.ckpt.data-00000-of-00001')
					notepad_file.write(f'\n\nThe size of this model is : {os.path.getsize(model_filepath) >> 20} Mb.')
				notepad_file.write('\n\nSUBMIT SCORE: ')

			print('\nModel Properties saved')


		# PREDICT ON TEST SET
		if predict_test_set:
			print('\nTesting begins')
			x_csv_test = pd.read_csv(os.path.join(data_dir,'test.csv'))
			x_test = preprocess_cv_test_data(x_csv_test, x_mean, x_std)
			y_submission = pd.read_csv(os.path.join(data_dir,'sample_submission.csv'))

			nbr_test_ex = x_test.shape[0]
			batch_size_test = 100 # doesn't really matter, just a matter of balance for being able to print partial resutls
			if nbr_test_ex % batch_size_test:
				print('The size of the batch for the test set is not a divisor of the whole set, fix that to avoid issues at the end of the set')

			# Do the prediction
			for i in range(nbr_test_ex // batch_size_test):
				print_prediction_class_test = sess.run(prediction_class, feed_dict={X:extract_batch_test(x_test, i, batch_size_test), keep_prob:1.0, is_train:False})
				if (i+1) % 20 == 0:
					print(f'Finishing test batch {i+1} / {nbr_test_ex / batch_size_test}')

				for idx,value in enumerate(print_prediction_class_test):
					y_submission['Label'].at[i*batch_size_test + idx] = value

			# Save the prediction
			y_submission.to_csv(os.path.join(save_dir,'submission_' +  datetime_now + '.csv'), index=False)


### To DO :
# change acvtivation
