import glob
import multiprocessing as mp
import os
import sys

import cv2
import numpy as np
import torch
import yaml
from torch.utils import data

PROCESSORS = 8

paths_file = os.path.abspath(os.path.join(os.getcwd(), os.pardir, "paths.yaml"))
paths = yaml.load(open(paths_file, 'r'))

root_dir = paths["data_train_path"]
sys.path.append(paths["project_path"])

import utils.visda_helper as visda


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

		self.size = len(self.image_fnlist)
		self.img_size = im_size

	def __getitem__(self, index):
		img_fn = self.image_fnlist[index]
		lbl_fn = self.label_fnlist[index]

		img = cv2.imread(img_fn)
		lbl = cv2.imread(lbl_fn)

		size = (self.img_size[1], self.img_size[0])
		img = cv2.resize(img, size, cv2.INTER_AREA)
		lbl = cv2.resize(lbl, size, cv2.INTER_NEAREST)

		lbl = transform_labels(lbl)

		img = img - self.img_mean
		img /= self.img_stdev

		img = torch.from_numpy(img).permute(2, 0, 1).type(torch.FloatTensor)
		lbl = torch.from_numpy(lbl).type(torch.LongTensor)

		return (img, lbl)

	def __len__(self):
		return self.size


class EagerVisDaDataset(data.Dataset):

	def __init__(self, im_size=visda.shape, mode="eval"):
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

		self.size = len(self.image_fnlist)
		self.img_size = im_size
		self.shape = im_size

		pool = mp.Pool(PROCESSORS)
		self.data = pool.starmap(self.load_img, zip(self.image_fnlist, self.label_fnlist))

	def load_img(self, img_fn, lbl_fn):

		img = cv2.imread(img_fn)
		lbl = cv2.imread(lbl_fn)

		size = (self.shape[1], self.shape[0])
		img = cv2.resize(img, size, cv2.INTER_AREA)
		lbl = cv2.resize(lbl, size, cv2.INTER_NEAREST)

		lbl = transform_labels(lbl)

		img = img - self.img_mean
		img /= self.img_stdev

		img = torch.from_numpy(img).permute(2, 0, 1).type(torch.FloatTensor)
		lbl = torch.from_numpy(lbl).type(torch.LongTensor)

		return (img, lbl)

	def __getitem__(self, index):
		return self.data[index]

	def __len__(self):
		return self.size


## Helper Functions: ##

def transform_labels(lbl_img):
	out = np.zeros((lbl_img.shape[0], lbl_img.shape[1]))

	for lbl in visda.labels:
		if lbl.trainId in visda.ignore_labels: continue
		out[np.where(np.all(lbl_img == lbl.color, axis=-1))] = lbl.trainId

	return out
