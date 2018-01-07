import os
import glob
import numpy as np
import torch
import cv2
from torch.utils import data

import multiprocessing as mp
PROCESSORS = 8

root_dir = "/media/data/train"


class EagerVisDaDataset(data.Dataset):

	num_classes = 27
	ignore_labels = [0]

	shape = (1052, 1914)

	img_mean = np.array([108.56263368194266, 111.92560322135374, 113.01417537462997])
	img_stdev = 60

	labels = [
			(0, 0, 0), (20, 20, 20), (0, 74, 111), (81, 0, 81), (128, 64, 128),
			(232, 35, 244), (70, 70, 70), (156, 102, 102), (153, 153, 190),
			(180, 165, 180), (100, 100, 150), (90, 120, 150), (153, 153, 153), (30, 170, 250),
			(0, 220, 220), (35, 142, 107), (152, 251, 152), (180, 130, 70), (60, 20, 220), (0, 0, 255),
			(70, 0, 0), (100, 60, 0), (110, 0, 0), (100, 80, 0), (230, 0, 0), (32, 11, 119), (142, 0, 0)
	]

	names = [
		'unlabeled', 'static', 'dynamic', 'ground', 'road',
		'sidewalk', 'building', 'wall', 'fence', 'guard rail', 'bridge', 'tunnel',
		'pole', 'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider',
		'truck', 'bus', 'trailer', 'train', 'motorcycle', 'bicycle', 'license plate'
	]

	class_weights = torch.FloatTensor([
		0.07938154591035282, 0.0016825634303498946, 0.017052185958826162, 0.0027889041507607516,
		0.3215191428487277, 0.08374657046653439, 0.16931354687873518, 0.01830169146084464, 0.006177791836249105,
		0.00029597960970296897, 0.008290651722829744, 0.0006005538561464518, 0.010523853930535492, 0.0013540632950341603,
		0.0008573018347492742, 0.0751681507220292, 0.020420402719133743, 0.13652687513867484, 0.0036909182443827914,
		0.00029019466378333284, 0.0, 0.012004842302352221, 0.00360455521521489, 5.8482905122366766e-05,
		0.000549278449228592, 0.0002823316110694899, 5.228239786719245e-05, 0.025465338440762618
	])

	def __init__(self, im_size=shape, mode="train"):
		if mode == "train":
			self.image_fnlist = glob.glob(os.path.join(root_dir, "images", "*.png"))
			self.label_fnlist = [fn.replace("images", "annotations") for fn in self.image_fnlist]
		else:
			self.image_fnlist = glob.glob(os.path.join(root_dir, "eval", "images", "*.png"))
			self.label_fnlist = [fn.replace("images", "annotations") for fn in self.image_fnlist]

		self.size = len(self.image_fnlist)
		self.img_size = im_size
		self.shape = im_size

		pool = mp.Pool(PROCESSORS)
		self.data = pool.starmap(self.load_img, zip(self.image_fnlist, self.label_fnlist))
	
	def load_img(self, img_fn, lbl_fn):

		img = cv2.imread(img_fn)
		lbl = cv2.imread(lbl_fn)

		if (img.shape[0] != lbl.shape[0] or img.shape[1] != lbl.shape[1]):
			return self.__getitem__(index+1)

		if (lbl.shape != self.shape):
			size = (self.shape[1], self.shape[0])
			img = cv2.resize(img, size, cv2.INTER_LINEAR)
			lbl = cv2.resize(lbl, size, cv2.INTER_NEAREST)

		lbl = self.transform_labels(lbl)

		img = img - self.img_mean
		img /= self.img_stdev

		img = torch.from_numpy(img).permute(2, 0, 1).type(torch.FloatTensor)
		lbl = torch.from_numpy(lbl).type(torch.LongTensor)

		return (img, lbl)

	def __getitem__(self, index):
		return self.data[index]

	def __len__(self):
		return self.size

	def transform_labels(self, lbl):
		out = np.zeros((lbl.shape[0], lbl.shape[1]))
		for i, col in enumerate(self.labels):
			if i in self.ignore_labels: continue
			out[np.where(np.all(lbl == col, axis=-1))] = i
		return out
