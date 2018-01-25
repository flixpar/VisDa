import glob
import os
import sys

import cv2
import numpy as np
import torch
import yaml
from torch.utils import data

from PIL import Image
from torchvision import transforms

import util.visda_helper as visda

from util.setup import load_args
args = load_args(os.getcwd())
paths = args.paths 
root_dir = paths["data_train_path"]


class VisDaDataset(data.Dataset):

	def __init__(self, im_size=visda.shape, mode="train"):
		if mode == "train":
			self.image_fnlist = glob.glob(os.path.join(root_dir, "images", "*.png"))
			self.label_fnlist = [fn.replace("images", "annotations") for fn in self.image_fnlist]
		else:
			self.image_fnlist = glob.glob(os.path.join(root_dir, "eval", "images", "*.png"))
			self.label_fnlist = [fn.replace("images", "annotations") for fn in self.image_fnlist]

		self.num_classes = visda.num_classes
		self.img_mean = visda.img_mean
		self.img_stdev = visda.img_stdev

		self.class_weights = torch.FloatTensor(visda.class_weights)

		self.num_images = len(self.image_fnlist)
		self.img_size = im_size
		self.default_size = visda.shape

	def __getitem__(self, index):
		img_fn = self.image_fnlist[index]
		lbl_fn = self.label_fnlist[index]

		img = Image.open(img_fn)
		lbl = Image.open(lbl_fn)

		img.resize(self.img_size, resample=Image.BICUBIC)
		lbl.resize(self.img_size, resample=Image.NEAREST)

		img = np.asarray(img)
		lbl = np.asarray(lbl)

		lbl = transform_labels(lbl)

		img = img - self.img_mean
		img /= self.img_stdev

		img = torch.from_numpy(img).permute(2, 0, 1).type(torch.FloatTensor)
		lbl = torch.from_numpy(lbl).type(torch.LongTensor)

		return (img, lbl)

	def __len__(self):
		return self.num_images

	def get_original(self, index):
		img_fn = self.image_fnlist[index]
		lbl_fn = self.label_fnlist[index]

		img = cv2.imread(img_fn)
		lbl = cv2.imread(lbl_fn)

		lbl = transform_labels(lbl)

		return (img, lbl)

## Helper Functions: ##

def transform_labels(lbl_img):
	out = np.zeros((lbl_img.shape[0], lbl_img.shape[1]))

	for lbl in visda.labels:
		color = np.flip(lbl.color, 0)
		if lbl.trainId in visda.ignore_labels: continue
		out[np.where(np.all(lbl_img == color, axis=-1))] = lbl.trainId

	return out
