# -*- coding: utf-8 -*-

#
# * Copyright (c) 2009-2016. Authors: see NOTICE file.
# *
# * Licensed under the Apache License, Version 2.0 (the "License");
# * you may not use this file except in compliance with the License.
# * You may obtain a copy of the License at
# *
# *      http://www.apache.org/licenses/LICENSE-2.0
# *
# * Unless required by applicable law or agreed to in writing, software
# * distributed under the License is distributed on an "AS IS" BASIS,
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# * See the License for the specific language governing permissions and
# * limitations under the License.
# */

__author__ = "Marganne Louis <louis.marganne@student.uliege.be>"


from cytomine.models import ImageInstanceCollection, Job, AttachedFileCollection, Annotation, AnnotationCollection
from cytomine import CytomineJob

from tensorflow.keras.models import load_model

from shapely.geometry import Point

import tensorflow as tf
import numpy as np
import cv2 as cv

from utils import *

import joblib
import glob
import sys
import os


def find_by_attribute(att_fil, attr, val):
	return next(iter([i for i in att_fil if hasattr(i, attr) and getattr(i, attr) == val]), None)

def main(argv):
	with CytomineJob.from_cli(argv) as cj:
		cj.job.update(status=Job.RUNNING, progress=0, status_comment="Initialization of the prediction phase...")

		## 1. Create working directories on the machine:
		# - WORKING_PATH/images: store input images
		# - WORKING_PATH/rescaled: store rescaled version of images
		# - WORKING_PATH/in: store output from the model to use

		base_path = "{}".format(os.getenv("HOME"))
		working_path = os.path.join(base_path, str(cj.job.id))
		images_path = os.path.join(working_path, 'images/')
		rescaled_path = os.path.join(working_path, 'rescaled/')
		rescaled_images_path = os.path.join(rescaled_path, 'images/')
		in_path = os.path.join(working_path, 'in/')

		if not os.path.exists(working_path):
			os.makedirs(working_path)
			os.makedirs(images_path)
			os.makedirs(rescaled_path)
			os.makedirs(rescaled_images_path)
			os.makedirs(in_path)


		## 2. Parse input data
		# Select list of images corresponding to input
		images = ImageInstanceCollection().fetch_with_filter("project", cj.parameters.cytomine_id_project)
		image_id_to_object = {image.id : image for image in images}
		if cj.parameters.images_to_predict == 'all':
			pred_images = images
		else:
			images_ids = [int(image_id) for image_id in cj.parameters.images_to_predict.split(',')]
			pred_images = [image_id_to_object[image_id] for image_id in images_ids]


		# Fetch data from the trained model
		tr_model_job = Job().fetch(cj.parameters.model_to_use)
		attached_files = AttachedFileCollection(tr_model_job).fetch()
		tr_model = find_by_attribute(attached_files, 'filename', '%d_model.hdf5' % cj.parameters.model_to_use)
		tr_model_filepath = in_path + '%d_model.hdf5' % cj.parameters.model_to_use
		tr_model.download(tr_model_filepath, override=True)
		tr_parameters = find_by_attribute(attached_files, 'filename', '%d_parameters.joblib' % cj.parameters.model_to_use)
		tr_parameters_filepath = in_path + '%d_parameters.joblib' % cj.parameters.model_to_use
		tr_parameters.download(tr_parameters_filepath, override=True)

		# Load fetched data
		model = load_model(tr_model_filepath)
		parameters_hash = joblib.load(tr_parameters_filepath)


		## 3. Download the images
		cj.job.update(progress=5, statusComment='Downloading images...')

		for image in pred_images:
			image.download(dest_pattern=images_path+'%d.tif' % image.id)


		## 4. Apply rescale to input
		cj.job.update(progress=50, statusComment='Rescaling images...')

		org_images = glob.glob(images_path+'*.tif')
		for i in range(len(org_images)):
			image = cv.imread(org_images[i], cv.IMREAD_UNCHANGED)
			im_name = os.path.basename(org_images[i])[:-4]
			re_img, _ = rescale_pad_img(image, None, 256)
			cv.imwrite(rescaled_images_path+im_name+'.png', re_img)


		## 5. Construct testing set with tensorflow
		test_images = glob.glob(rescaled_images_path+'*.png')

		test_ds = tf.data.Dataset.from_tensor_slices((test_images, None))
		test_ds = test_ds.map(lambda image, lm: parse_data(image, lm), num_parallel_calls=tf.data.experimental.AUTOTUNE)
		test_ds = test_ds.map(lambda image, lm: normalize(image, lm), num_parallel_calls=tf.data.experimental.AUTOTUNE)
		test_ds = test_ds.batch(parameters_hash['model_batch_size'])

		
		## 6. Predict landmarks
		cj.job.update(progress=80, statusComment='Predicting landmarks...')

		# Predict using the model
		preds = model.predict(test_ds)

		# Upscale prediction to original size
		pred_landmarks = []
		for i in range(len(test_images)):
			org_img = cv.imread(org_images[i])
			pred_mask = preds[i]
			pred_mask = np.reshape(pred_mask, newshape=(256,256, parameters_hash['N']))

			lm_list = []
			for j in range(parameters_hash['N']):
				x, y =  maskToKeypoints(pred_mask[:, :, j])
				lm_list.append((x, y))
			
			pred_lm = np.array(lm_list)
			up_size = org_img.shape[:2]
			up_lmks = up_lm(pred_lm, 256, up_size)
			pred_landmarks.append(up_lmks)


		## 7. Save landmarks as annotations in Cytomine
		annotation_collection = AnnotationCollection()
		for i, image in enumerate(pred_images):
			for j in range(parameters_hash['N']):
				lm = Point(pred_landmarks[i][j][0], image.height - pred_landmarks[i][j][1])
				annotation_collection.append(Annotation(location=lm.wkt, id_image=image.id, id_terms=[parameters_hash['cytomine_id_terms'][j]], id_project=cj.parameters.cytomine_id_project))
		annotation_collection.save()

		cj.job.update(status=Job.TERMINATED, progress=100, statusComment='Job terminated.')


if __name__ == '__main__':
	main(sys.argv[1:])