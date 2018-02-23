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
sys.path.append(paths["project_path"])

import util.cityscapes_helper as cityscapes


class TestDataset(data.Dataset):

	def __init__(self, im_size=cityscapes.shape, n_samples=None, mode="bgr"):
		self.image_fnlist = sorted(glob.glob(os.path.join(data_dir, "*_img.png")))
		
		self.img_mean = 0
		self.img_stdev = 0
		self.img_mode = mode

		self.size = len(self.image_fnlist)
		self.img_size = im_size
		self.default_size = cityscapes.shape

		self.n_samples = n_samples

		self.out_size = (self.img_size[1], self.img_size[0])

	def __getitem__(self, index):
		img_fn = self.image_fnlist[index]
		lbl_fn = self.label_fnlist[index]

		name = img_fn.split('/')[-1].split('_')[0]

		img = cv2.imread(img_fn)
		img = self.enhance_contrast(img)
		img = cv2.resize(img, self.out_size, interpolation=cv2.INTER_AREA)

		if self.img_mode == "rgb":
			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			img = img - np.flip(self.img_mean, 0)
			img /= np.flip(self.img_stdev, 0)
		else:
			img = img - self.img_mean
			img /= self.img_stdev

		img = torch.from_numpy(img).permute(2, 0, 1).type(torch.FloatTensor)

		return img, name

	def __len__(self):
		return self.n_samples

	def reverse_labeling(self, lbl):
		out = np.zeros((lbl.shape[0], lbl.shape[1]))

		for i in range(self.num_classes):
			n = cityscapes.trainId2label[i].id
			out[lbl == n] = i

		return out

	def enhance_contrast(self, img):
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		img = equalize_adapthist(img)
		img = rescale_intensity(img, out_range='uint8').astype(np.uint8)
		img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
		return img
