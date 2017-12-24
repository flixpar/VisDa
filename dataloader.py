import os
import glob
import numpy as np
import torch
import cv2
from torch.utils import data

root_dir = ""


class VisDaDataLoader(data.Dataset):

	num_classes = 0
	class_weights = torch.ones(num_classes)
	ignore_labels = []

	def __init__(self):

		self.image_fnlist = glob.glob(os.path.join(root_dir, "images", "*.png"))
		self.label_fnlist = [fn.replace("images", "annotations") for fn in self.image_fnlist]
		# self.image_fnlist = sorted(glob.glob(os.path.join(root_dir, "images", "*.png")))
		# self.label_fnlist = sorted(glob.glob(os.path.join(root_dir, "annotations", "*.png")))

		self.size = len(self.image_fnlist)
		self.img_size = cv2.imread(self.image_fnlist[0]).size()[0]

	def __getitem__(self, index):

		img_fn = self.image_fnlist[index]
		lbl_fn = self.label_fnlist[index]

		img = cv2.imread(img_fn)
		lbl = cv2.imread(lbl_fn, 0)

		return (img, lbl)

	def __len__(self):
		return self.size
