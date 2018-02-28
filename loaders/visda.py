import glob
import os
import sys
import random

import cv2
import numpy as np
import torch
import yaml

from torch.utils import data
from torchvision import transforms

import util.visda_helper as visda

from util.setup import load_args
args = load_args(os.getcwd())
paths = args.paths

root_dir = paths["data_train_path"]


class VisDaDataset(data.Dataset):

	def __init__(self, im_size=visda.shape, samples=None):
		self.image_fnlist = glob.glob(os.path.join(root_dir, "images", "*.png"))

		if samples is not None:
			self.image_fnlist = random.sample(self.image_fnlist, samples)

		self.label_fnlist = [fn.replace("images", "annotations") for fn in self.image_fnlist]

		self.num_classes = visda.num_classes - 1
		self.img_mean = visda.img_mean / 255.0
		self.img_stdev = visda.img_stdev / 255.0

		class_weights = -1 * np.log(np.array(visda.class_weights))
		class_weights /= np.max(class_weights)
		class_weights = class_weights[1:]
		self.class_weights = torch.FloatTensor(class_weights)

		self.length = len(self.image_fnlist)

		self.img_size = im_size
		self.default_size = visda.shape

		self.norm = transforms.Normalize(mean=self.img_mean, std=self.img_stdev)

	def __getitem__(self, index):
		img_fn = self.image_fnlist[index]
		lbl_fn = self.label_fnlist[index]

		img = cv2.imread(img_fn)
		lbl = cv2.imread(lbl_fn)

		size = (self.img_size[1], self.img_size[0])
		img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
		lbl = cv2.resize(lbl, size, interpolation=cv2.INTER_NEAREST)

		lbl = self.transform_labels(lbl)
		img = img.astype(np.float32) / 255.0

		img = torch.from_numpy(img).permute(2, 0, 1).type(torch.FloatTensor)
		lbl = torch.from_numpy(lbl).type(torch.LongTensor)

		img = self.norm(img)

		return img, lbl

	def transform_labels(self, lbl_img):
		out = np.zeros((lbl_img.shape[0], lbl_img.shape[1]))

		for lbl in visda.labels:
			c = lbl.trainId-1 if lbl.trainId != 0 else 255
			out[np.where(np.all(lbl_img == lbl.color, axis=-1))] = c

		return out

	def __len__(self):
		return self.length

