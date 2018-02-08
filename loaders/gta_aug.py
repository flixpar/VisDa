import torch
import torchvision.transforms.functional as transforms
from torch.utils import data

import cv2
import numpy as np
from PIL import Image

import os
import glob
import random
import yaml

from skimage import transform

import util.visda_helper as visda
from util.setup import load_args

args = load_args(os.getcwd())
paths = args.paths


class GTA5AugDataset(data.Dataset):

	def __init__(self, im_size=visda.shape, batch_size=4, samples=None):

		image_path = os.path.join(paths["data_train_path"], "images", "*.png")
		self.image_fnlist = glob.glob(image_path)
		if samples is not None:
			self.image_fnlist = random.sample(self.image_fnlist, samples)
		self.label_fnlist = [fn.replace("images", "annotations") for fn in self.image_fnlist]
		self.num_images = len(self.image_fnlist)

		self.num_classes = visda.num_classes
		self.img_mean = np.flip(visda.img_mean, 0) / 255
		self.img_stdev = np.flip(visda.img_stdev, 0) / 255

		class_weights = -1 * np.log(np.array(visda.class_weights))
		class_weights /= np.max(class_weights)
		self.class_weights = torch.FloatTensor(class_weights)

		self.img_size = im_size
		self.default_size = visda.shape
		self.batch_size = batch_size

		self.transforms = ["scale"]
		self.scale_factors = [0.80, 1.00, 1.20, 1.40, 1.60]
		self.aug_factor = len(self.scale_factors)

	def __getitem__(self, index):

		block_num = index // (self.aug_factor * self.batch_size)
		block_pos = index %  (self.aug_factor * self.batch_size)

		block_img_num = block_pos % self.batch_size
		img_num = (block_num * self.batch_size) + block_img_num

		aug_pos = block_pos // self.batch_size
		scale_factor = self.scale_factors[aug_pos]

		img_fn = self.image_fnlist[img_num]
		lbl_fn = self.label_fnlist[img_num]

		img = Image.open(img_fn)
		lbl = cv2.imread(lbl_fn)

		img, lbl = self.scale(img, lbl, scale_factor)
		lbl = self.transform_labels(lbl)
		
		img = transforms.to_tensor(img)
		lbl = torch.from_numpy(lbl).type(torch.LongTensor)

		img = transforms.normalize(img, self.img_mean, self.img_stdev)

		return img, lbl

	def transform_labels(self, lbl_img):
		out = np.zeros((lbl_img.shape[0], lbl_img.shape[1]), dtype=np.uint8)

		for lbl in visda.labels:
			if lbl.trainId in visda.ignore_labels: continue
			out[np.where(np.all(lbl_img == lbl.color, axis=-1))] = lbl.trainId

		return out

	def scale(self, img, lbl, factor):

		scale_size = (int(factor * self.img_size[1]), int(factor * self.img_size[0]))
		crop_size  = (self.img_size[1], self.img_size[0])

		img = img.resize(scale_size, resample=Image.LANCZOS)
		lbl = cv2.resize(lbl, scale_size, interpolation=cv2.INTER_NEAREST)

		if factor > 1.0:

			startx = random.randint(0, scale_size[1] - crop_size[1])
			starty = random.randint(0, scale_size[0] - crop_size[0])

			img = transforms.crop(img, startx, starty, crop_size[1], crop_size[0])
			lbl = lbl[startx:endx, starty:endy]

		if factor < 1.0:

			dh = crop_size[1] - scale_size[1]
			dw = crop_size[0] - scale_size[0]

			top = dh//2 if dh%2==0 else (dh//2)+1
			bottom = dh//2

			left = dw//2 if dw%2==0 else (dw//2)+1
			right = dw//2

			img = transforms.pad(img, (left, top, right, bottom))
			lbl = cv2.copyMakeBorder(lbl, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT, value=0)

		return img, lbl

	def reshuffle(self):
		self.image_fnlist = random.shuffle(self.image_fnlist)
		self.label_fnlist = [fn.replace("images", "annotations") for fn in self.image_fnlist]

	def __len__(self):
		return self.num_images * self.aug_factor

