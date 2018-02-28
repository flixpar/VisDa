import os
import sys
import glob
import yaml
import numpy as np
import torch
import cv2
from torch.utils import data
from skimage.exposure import equalize_adapthist, rescale_intensity
import skimage

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from util.setup import load_args
args = load_args(os.getcwd())
paths = args.paths

data_dir = paths["cityscapes_val_path"]

import util.cityscapes_helper as cityscapes


class CityscapesSelectDataset(data.Dataset):

	def __init__(self, im_size=cityscapes.shape, n_samples=30):
		self.image_fnlist = sorted(glob.glob(os.path.join(data_dir, "*_img.png")))
		self.label_fnlist = [fn.replace("img", "lbl") for fn in self.image_fnlist]

		self.num_classes = cityscapes.num_classes
		self.img_mean = cityscapes.img_mean
		self.img_stdev = cityscapes.img_stdev

		self.size = len(self.image_fnlist)
		self.img_size = im_size
		self.default_size = cityscapes.shape

		self.n_samples = n_samples

	def __getitem__(self, index):
		img_fn = self.image_fnlist[index]
		lbl_fn = self.label_fnlist[index]

		name = img_fn.split('/')[-1].split('_')[0]

		src_img = cv2.imread(img_fn)
		src_lbl = cv2.imread(lbl_fn, 0)

		src_img = self.enhance_contrast(src_img)
		src_lbl = self.transform_labels(src_lbl)

		size = (self.img_size[1], self.img_size[0])
		img = cv2.resize(src_img.copy(), size, interpolation=cv2.INTER_AREA)
		lbl = cv2.resize(src_lbl.copy(), size, interpolation=cv2.INTER_NEAREST)

		img = img - self.img_mean
		img /= self.img_stdev

		img = torch.from_numpy(img).permute(2, 0, 1).type(torch.FloatTensor)
		lbl = torch.from_numpy(lbl).type(torch.LongTensor)

		return (img, lbl), (src_img, src_lbl), name

	def __len__(self):
		return self.n_samples

	def transform_labels(self, lbl):
		out = np.zeros((lbl.shape[0], lbl.shape[1]))

		for l in cityscapes.labels:
			c = l.trainId-1 if l.trainId != 0 else 255
			out[lbl == l.id] = c

		return out

	def get_original(self, index):
		img_fn = self.image_fnlist[index]
		lbl_fn = self.label_fnlist[index]

		img = cv2.imread(img_fn)
		lbl = cv2.imread(lbl_fn, 0)

		lbl = self.transform_labels(lbl)

		return (img, lbl)

	def enhance_contrast(self, img):
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		img = equalize_adapthist(img)
		img = rescale_intensity(img, out_range='uint8').astype(np.uint8)
		img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
		return img
