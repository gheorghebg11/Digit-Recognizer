# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 11:33:21 2019

@author: Bogdan
"""

import os
from datetime import datetime

import numpy as np
import pandas as pd

np.set_printoptions(precision=2, suppress=True, threshold=np.nan)

datetime_now = str(datetime.now()).replace('-','').replace(':','').replace(' ','')
datetime_now = datetime_now[:datetime_now.find('.')]

########## Paths
data_dir = os.path.join(os.getcwd(), 'data')
save_dir = os.path.join(os.getcwd(), 'ensemble_' + datetime_now)
os.mkdir(save_dir)
save_path_sub = os.path.join(save_dir,'submission_' +  datetime_now + '.csv')
save_path_all = os.path.join(save_dir,'all_pred_' +  datetime_now + '.csv')
#############

# SET here the name of the predictions to use
model_names = ['20190212122848', '20190212123001']

# Load sample
y_submission = pd.read_csv(os.path.join(data_dir,'sample_submission.csv'))
nbr_images = len(y_submission.index)
print(f'There are {nbr_images} test images and {len(model_names)} models')

# Load the submissions
for model in model_names:
	submission_path = os.path.join(os.getcwd(), 'model_' + model, 'submission_' +  model + '.csv')
	submission = pd.read_csv(submission_path)['Label'].rename(columns={'Label':model})
	y_submission = pd.concat([y_submission, submission], axis = 1).rename(columns={0:model})
print('Csv concatenated, we now pick the most common value for each image\n')

# Pick the most common value
nbr_images_overten = nbr_images // 10
for idx, row in y_submission.iterrows():
	y_submission.iloc[idx]['Label'] = row.drop(['ImageId']).value_counts().index[0]
	if (idx + 1) % nbr_images_overten == 0:
		print(f'{(idx + 1) // nbr_images_overten:2.0f} / 10 of the job done')

# Save the prediction
y_submission[['ImageId', 'Label']].to_csv(save_path_sub, index=False)
y_submission.to_csv(save_path_all, index=False)
print(f'\nFile saved in {save_path_sub}')