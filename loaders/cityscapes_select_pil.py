import torch
import torchvision.transforms.functional as transforms
from torch.utils import data

from PIL import Image

import cv2
import numpy as np

import skimage
from skimage.exposure import equalize_adapthist, rescale_intensity

import os
import glob
import yaml
from pathlib import Path

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import util.cityscapes_helper as cityscapes

from util.setup import load_args
args = load_args(os.getcwd())
paths = args.paths


class CityscapesSelectDataset(data.Dataset):

	def __init__(self, im_size=cityscapes.shape, n_samples=30):
		self.image_fnlist = sorted(glob.glob(os.path.join(paths["cityscapes_val_path"], "*_img.png")))
		self.label_fnlist = [fn.replace("img", "lbl") for fn in self.image_fnlist]

		self.num_classes = cityscapes.num_classes
		self.img_mean = np.flip(cityscapes.img_mean, 0) / 255
		self.img_stdev = np.flip(cityscapes.img_stdev, 0) / 255

		self.size = len(self.image_fnlist)
		self.img_size = im_size
		self.default_size = cityscapes.shape

		self.n_samples = n_samples

	def __getitem__(self, index):
		img_fn = self.image_fnlist[index]
		lbl_fn = self.label_fnlist[index]

		name = img_fn.split('/')[-1].split('_')[0]

		src_img = Image.open(img_fn)
		src_lbl = Image.open(lbl_fn)

		src_img = self.enhance_contrast(src_img)
		src_lbl = self.transform_labels(src_lbl)

		size = (self.img_size[1], self.img_size[0])
		img = src_img.resize(size, resample=Image.LANCZOS)
		lbl = src_lbl.resize(size, resample=Image.NEAREST)

		img = transforms.to_tensor(img)
		lbl = transforms.to_tensor(lbl)

		img = transforms.normalize(img, self.img_mean, self.img_stdev)
		lbl = torch.squeeze(lbl.type(torch.LongTensor))

		src_img = np.asarray(src_img)
		src_lbl = np.asarray(src_lbl)

		return (img, lbl), (src_img, src_lbl), name

	def __len__(self):
		return self.n_samples

	def transform_labels(self, lbl):
		lbl = np.array(lbl)
		out = np.zeros((lbl.shape[0], lbl.shape[1]), dtype=np.uint8)

		for i in range(self.num_classes):
			n = cityscapes.trainId2label[i].id
			out[lbl == n] = i

		return Image.fromarray(out)

	def get_original(self, index):
		img_fn = self.image_fnlist[index]
		lbl_fn = self.label_fnlist[index]

		img = cv2.imread(img_fn)
		lbl = cv2.imread(lbl_fn, 0)

		lbl = self.transform_labels(lbl)

		return img, lbl

	def enhance_contrast(self, img):
		img = np.array(img)
		img = equalize_adapthist(img)
		img = rescale_intensity(img, out_range='uint8').astype(np.uint8)
		img = Image.fromarray(img)
		return img
