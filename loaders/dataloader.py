import os
import glob
import numpy as np
import torch
import cv2
from torch.utils import data

root_dir = "/media/data/train"


class VisDaDataset(data.Dataset):

	num_classes = 35
	ignore_labels = [0, 1, 2, 3]

	shape = (1052, 1914)

	img_mean = np.array([108.56263368194266, 111.92560322135374, 113.01417537462997])
	img_stdev = 60

	# labels = [(0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (20, 20, 20), (111, 74, 0), (81, 0, 81), (128, 64, 128),
	#           (244, 35, 232), (250, 170, 160), (230, 150, 140), (70, 70, 70), (102, 102, 156), (190, 153, 153),
	#           (180, 165, 180), (150, 100, 100), (150, 120, 90), (153, 153, 153), (153, 153, 153), (250, 170, 30),
	#           (220, 220, 0), (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60), (255, 0, 0), (0, 0, 142),
	#           (0, 0, 70), (0, 60, 100), (0, 0, 90), (0, 0, 110), (0, 80, 100), (0, 0, 230), (119, 11, 32), (0, 0, 142)]

	labels = [(0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (20, 20, 20), (0, 74, 111), (81, 0, 81), (128, 64, 128),
			(232, 35, 244), (160, 170, 250), (140, 150, 230), (70, 70, 70), (156, 102, 102), (153, 153, 190),
			(180, 165, 180), (100, 100, 150), (90, 120, 150), (153, 153, 153), (153, 153, 153), (30, 170, 250),
			(0, 220, 220), (35, 142, 107), (152, 251, 152), (180, 130, 70), (60, 20, 220), (0, 0, 255), (142, 0, 0),
			(70, 0, 0), (100, 60, 0), (90, 0, 0), (110, 0, 0), (100, 80, 0), (230, 0, 0), (32, 11, 119), (142, 0, 0)]

	names = ['unlabeled', 'ego vehicle', 'rectification border', 'out of roi', 'static', 'dynamic', 'ground', 'road',
	         'sidewalk', 'parking', 'rail track', 'building', 'wall', 'fence', 'guard rail', 'bridge', 'tunnel',
	         'pole', 'polegroup', 'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider',
	         'car', 'truck', 'bus', 'caravan', 'trailer', 'train', 'motorcycle', 'bicycle', 'license plate']

	# class_weights = torch.FloatTensor([0.471072493982, 0.0, 0.0, 0.0, 0.00181448729946, 0.0, 0.00267729106253, 0.324546718887,
	# 		0.0, 0.0, 0.0, 0.167350940922, 0.0, 0.0, 0.000255553958685, 0.0, 0.0, 0.0, 0.0106366173936,
	# 		0.0, 0.0, 0.0, 0.0216458964943, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
	
	class_weights = torch.FloatTensor([
		0.07938154591035282, 0.0, 0.0, 0.0, 0.0016825634303498946, 0.017052185958826162, 0.0027889041507607516,
		0.3215191428487277, 0.08374657046653439, 0.0, 0.0, 0.16931354687873518, 0.01830169146084464, 0.006177791836249105,
		0.00029597960970296897, 0.008290651722829744, 0.0006005538561464518, 0.0, 0.010523853930535492, 0.0013540632950341603,
		0.0008573018347492742, 0.0751681507220292, 0.020420402719133743, 0.13652687513867484, 0.0036909182443827914,
		0.00029019466378333284, 0.0, 0.012004842302352221, 0.00360455521521489, 0.0, 5.8482905122366766e-05,
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

	def __getitem__(self, index):
		img_fn = self.image_fnlist[index]
		lbl_fn = self.label_fnlist[index]

		img = cv2.imread(img_fn)
		lbl = cv2.imread(lbl_fn)

		if (img.shape[0] != lbl.shape[0] or img.shape[1] != lbl.shape[1]):
			return self.__getitem__(index+1)

		if (lbl.shape != self.img_size):
			size = (self.img_size[1], self.img_size[0])
			img = cv2.resize(img, size, cv2.INTER_LINEAR)
			lbl = cv2.resize(lbl, size, cv2.INTER_NEAREST)

		lbl = self.transform_labels(lbl)

		img = img - self.img_mean
		img /= self.img_stdev

		img = torch.from_numpy(img).permute(2, 0, 1).type(torch.FloatTensor)
		lbl = torch.from_numpy(lbl).type(torch.LongTensor)

		return (img, lbl)

	def __len__(self):
		return self.size

	def transform_labels(self, lbl):
		out = np.zeros((lbl.shape[0], lbl.shape[1]))
		for i, col in enumerate(self.labels):
			if i in self.ignore_labels: continue
			out[np.where(np.all(lbl == col, axis=-1))] = i
		return out
