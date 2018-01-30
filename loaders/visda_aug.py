import glob
import os
import sys
from random import randint

import cv2
import numpy as np
import torch
import yaml
from torch.utils import data

import util.visda_helper as visda

from util.setup import load_args
args = load_args(os.getcwd())
paths = args.paths 

root_dir = paths["data_train_path"]
sys.path.append(paths["project_path"])


class VisDaAugDataset(data.Dataset):

	def __init__(self, im_size=visda.shape):
		self.image_fnlist = glob.glob(os.path.join(root_dir, "images", "*.png"))
		self.label_fnlist = [fn.replace("images", "annotations") for fn in self.image_fnlist]
		self.num_images = len(self.image_fnlist)

		self.num_classes = visda.num_classes
		self.img_mean = visda.img_mean
		self.img_stdev = visda.img_stdev

		self.class_weights = torch.FloatTensor(visda.class_weights)

		self.img_size = im_size
		self.default_size = visda.shape

	def __getitem__(self, index):

		img_fn = self.image_fnlist[index]
		lbl_fn = self.label_fnlist[index]

		img = cv2.imread(img_fn)
		lbl = cv2.imread(lbl_fn)

		size = (self.img_size[1], self.img_size[0])
		img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
		lbl = cv2.resize(lbl, size, interpolation=cv2.INTER_NEAREST)

		lbl = transform_labels(lbl)

		img = img - self.img_mean
		img /= self.img_stdev

		images = []
		labels = []

		scale_factors = [1.0, 1.2, 1.4, 1.6]
		for factor in scale_factors:
			scale_size = (int(factor * size[0]), int(factor * size[1]))

			a = cv2.resize(img, scale_size, interpolation=cv2.INTER_AREA)
			b = cv2.resize(lbl, scale_size, interpolation=cv2.INTER_NEAREST)

			startx = randint(0,scale_size[1] - size[1])
			starty = randint(0,scale_size[0] - size[0])

			endx = startx + size[1]
			endy = starty + size[0]

			a = a[startx:endx, starty:endy]
			b = b[startx:endx, starty:endy]

			images.append(a)
			labels.append(b)

		images = np.stack(images, axis=0).transpose((0, 3, 1, 2))
		labels = np.stack(labels, axis=0)

		images = torch.from_numpy(images).type(torch.FloatTensor)
		labels = torch.from_numpy(labels).type(torch.LongTensor)

		return images, labels

	def __len__(self):
		return self.num_images


## Helper Functions: ##

def transform_labels(lbl_img):
	out = np.zeros((lbl_img.shape[0], lbl_img.shape[1]), dtype=np.uint8)

	for lbl in visda.labels:
		if lbl.trainId in visda.ignore_labels: continue
		out[np.where(np.all(lbl_img == lbl.color, axis=-1))] = lbl.trainId

	return out
