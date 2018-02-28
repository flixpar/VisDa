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
from torchvision import transforms

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from util.setup import load_args
args = load_args(os.getcwd())
paths = args.paths

data_dir = paths["cityscapes_path"]

import util.cityscapes_helper as cityscapes


class CityscapesDataset(data.Dataset):

	def __init__(self, im_size=cityscapes.shape, mode="train"):
		self.image_fnlist = glob.glob(os.path.join(paths["cityscapes_path"], "images", mode, "**", "*.png"), recursive=True)
		self.label_fnlist = [fn.replace("images", "annotations").replace("leftImg8bit", "gtFine_labelIds") for fn in self.image_fnlist]

		self.num_classes = cityscapes.num_classes - 1
		self.img_mean = cityscapes.img_mean / 255.0
		self.img_stdev = cityscapes.img_stdev / 255.0

		self.length = len(self.image_fnlist)

		self.img_size = im_size
		self.default_size = cityscapes.shape

		class_weights = [
			0.11586621, 0.3235678,  0.19765419, 0.0357135, 0.05368808, 0.14305691,
			0.06162343, 0.01132036, 0.00601009, 0.00239826, 0.01094606, 0.00802482,
			0.01207127, 0.00225619, 0.00184278, 0.00483001, 0.00307986, 0.00130785,
			0.00064916, 0.00409318
		]
		class_weights = -1 * np.log(np.array(class_weights))
		class_weights /= np.max(class_weights)
		class_weights = class_weights[1:]
		self.class_weights = torch.FloatTensor(class_weights)

		self.norm = transforms.Normalize(mean=self.img_mean, std=self.img_stdev)

	def __getitem__(self, index):
		img_fn = self.image_fnlist[index]
		lbl_fn = self.label_fnlist[index]

		src_img = cv2.imread(img_fn)
		src_lbl = cv2.imread(lbl_fn, 0)

		src_img = self.enhance_contrast(src_img)
		src_lbl = self.transform_labels(src_lbl)

		size = (self.img_size[1], self.img_size[0])
		img = cv2.resize(src_img.copy(), size, interpolation=cv2.INTER_AREA)
		lbl = cv2.resize(src_lbl.copy(), size, interpolation=cv2.INTER_NEAREST)

		img = img.astype(np.float32) / 255.0

		img = torch.from_numpy(img).permute(2,0,1).type(torch.FloatTensor)
		lbl = torch.from_numpy(lbl).type(torch.LongTensor)

		img = self.norm(img)

		return img, lbl

	def __len__(self):
		return self.length

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
		img = equalize_adapthist(img)
		img = rescale_intensity(img, out_range='uint8')
		return img
