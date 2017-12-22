import os
import glob.glob
import numpy as np
import torch
import cv2
from torch.utils import data

num_classes = 0
ignore_label = 0

root_dir = ""

class DataLoader(data.Dataset):

	def __init__(self):

		self.image_fnlist = sort(glob(os.path.join(root_dir, "images", "*.png")))
		self.label_fnlist = sort(glob(os.path.join(root_dir, "annotations", "*.png")))

		self.size = len(self.image_fnlist)

	def __getitem__(self, index):
		
		img_fn = self.image_fnlist[index]
		lbl_fn = self.label_fnlist[index]

		img = cv2.imread(img_fn)
		lbl = cv2.imread(lbl_fn, 0)

		return (img, lbl)

	def __len__(self):
		return self.size