import os
import glob
import numpy as np
import torch
import cv2
from torch.utils import data

root_dir = "/home/flixpar/data/train"


class VisDaDataset(data.Dataset):
	num_classes = 35
	class_weights = torch.ones(num_classes)
	ignore_labels = [0, 1, 2, 3]
	shape = (1052, 1914)
	#shape = (526, 957)

	labels = [(0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (20, 20, 20), (111, 74, 0), (81, 0, 81), (128, 64, 128),
	          (244, 35, 232), (250, 170, 160), (230, 150, 140), (70, 70, 70), (102, 102, 156), (190, 153, 153),
	          (180, 165, 180), (150, 100, 100), (150, 120, 90), (153, 153, 153), (153, 153, 153), (250, 170, 30),
	          (220, 220, 0), (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60), (255, 0, 0), (0, 0, 142),
	          (0, 0, 70), (0, 60, 100), (0, 0, 90), (0, 0, 110), (0, 80, 100), (0, 0, 230), (119, 11, 32), (0, 0, 142)]

	names = ['unlabeled', 'ego vehicle', 'rectification border', 'out of roi', 'static', 'dynamic', 'ground', 'road',
	         'sidewalk', 'parking', 'rail track', 'building', 'wall', 'fence', 'guard rail', 'bridge', 'tunnel',
	         'pole', 'polegroup', 'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider',
	         'car', 'truck', 'bus', 'caravan', 'trailer', 'train', 'motorcycle', 'bicycle', 'license plate']

	def __init__(self):
		self.image_fnlist = glob.glob(os.path.join(root_dir, "images", "*.png"))
		self.label_fnlist = [fn.replace("images", "annotations") for fn in self.image_fnlist]
		# self.image_fnlist = sorted(glob.glob(os.path.join(root_dir, "images", "*.png")))
		# self.label_fnlist = sorted(glob.glob(os.path.join(root_dir, "annotations", "*.png")))

		self.size = len(self.image_fnlist)
		#self.img_size = cv2.imread(self.image_fnlist[0]).shape[0:2]
		self.img_size = self.shape

	def __getitem__(self, index):
		img_fn = self.image_fnlist[index]
		lbl_fn = self.label_fnlist[index]

		img = cv2.imread(img_fn)
		lbl = cv2.imread(lbl_fn, 0)

		if (img.shape[0] != lbl.shape[0] or img.shape[1] != lbl.shape[1]):
			return self.__getitem__(index+1)

		#assert (img.shape[0] == lbl.shape[0])
		#assert (img.shape[1] == lbl.shape[1])

		if (lbl.shape != self.shape):
			size = (self.shape[1], self.shape[0])
			img = cv2.resize(img, size, cv2.INTER_LINEAR)
			lbl = cv2.resize(lbl, size, cv2.INTER_NEAREST)

		lbl = self.transform_labels(lbl)

		img = torch.from_numpy(img).permute(2, 0, 1).type(torch.FloatTensor)
		lbl = torch.from_numpy(lbl).type(torch.LongTensor)

		img = img / 255.0

		return (img, lbl)

	def __len__(self):
		return self.size

	def transform_labels(self, lbl):
		out = np.zeros((lbl.shape[0], lbl.shape[1]))
		for i, col in enumerate(self.labels):
			out[lbl == col] = i
		return out
