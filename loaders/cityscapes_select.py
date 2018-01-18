import os
import sys
import glob
import yaml
import numpy as np
import torch
import cv2
from torch.utils import data

paths_file = os.path.join(os.getcwd(), "paths.yaml")
paths = yaml.load(open(paths_file, 'r'))

data_dir = paths["cityscapes_val_path"]
sys.path.append(paths["project_path"])

import util.cityscapes_helper as cityscapes


class CityscapesSelectDataset(data.Dataset):

	def __init__(self, im_size=cityscapes.shape):
		self.image_fnlist = glob.glob(os.path.join(data_dir, "img*.png"))
		self.label_fnlist = [fn.replace("img", "lbl") for fn in self.image_fnlist]

		self.num_classes = cityscapes.num_classes
		self.img_mean = cityscapes.img_mean
		self.img_stdev = cityscapes.img_stdev

		self.size = len(self.image_fnlist)
		self.img_size = im_size
		self.default_size = cityscapes.shape

	def __getitem__(self, index):
		img_fn = self.image_fnlist[index]
		lbl_fn = self.label_fnlist[index]

		src_img = cv2.imread(img_fn)
		src_lbl = cv2.imread(lbl_fn, 0)

		size = (self.img_size[1], self.img_size[0])
		img = cv2.resize(src_img, size, interpolation=cv2.INTER_AREA)
		lbl = cv2.resize(src_lbl, size, interpolation=cv2.INTER_NEAREST)

		lbl = self.transform_labels(lbl)

		img = img - self.img_mean
		img /= self.img_stdev

		img = torch.from_numpy(img).permute(2, 0, 1).type(torch.FloatTensor)
		lbl = torch.from_numpy(lbl).type(torch.LongTensor)

		return (img, lbl), (src_img, src_lbl)

	def __len__(self):
		return self.size

	def transform_labels(self, lbl):
		out = np.zeros((lbl.shape[0], lbl.shape[1]))

		for i in range(self.num_classes):
			n = cityscapes.trainId2label[i].id
			out[lbl == n] = i

		return out

	def get_original(self, index):
		img_fn = self.image_fnlist[index]
		lbl_fn = self.label_fnlist[index]

		img = cv2.imread(img_fn)
		lbl = cv2.imread(lbl_fn, 0)

		lbl = self.transform_labels(lbl)

		return (img, lbl)
