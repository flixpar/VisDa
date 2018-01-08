import torch
from torch import nn
from torch import optim

import os
import cv2
import numpy as np

class Namespace:
	def __init__(self, **kwargs):
		self.__dict__.update(kwargs)
	def print(self):
		for key, val in sorted(self.__dict__.items()):
			print("{}:\t{}".format(key, val))

def poly_lr_scheduler(optimizer, init_lr, it, lr_decay_iter=1, max_iter=100, power=0.9):
	if it % lr_decay_iter or it > max_iter:
		return optimizer

	lr = init_lr*(1 - it/max_iter)**power
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr

def reverse_img_norm(image):
	img_mean = np.array([108.56263368194266, 111.92560322135374, 113.01417537462997])
	img_stdev = 60
	image = image.transpose(1, 2, 0)
	image *= img_stdev
	image += img_mean
	image = image.astype(np.uint8)
	return image

def recolor(lbl):
	labels = [
		(0,0,0), (128,64,128), (70,70,70), (180,130,70), (232,35,244),
		(35,142,107), (142,0,0), (152,251,152), (156,102,102), (70,0,0),
		(153,153,153), (153,153,190), (60,20,220), (100,60,0), (30,170,250),
		(0,220,220), (100,80,0), (0,0,255), (230,0,0), (32,11,119)
	]

	out = np.zeros((lbl.shape[0], lbl.shape[1], 3))
	for i in range(len(labels)):
		out[lbl==i] = labels[i]
	return out

def save_img(img, name, num, out_path, is_lbl=False):
	fn = "{}_{}.png".format(num, name)
	path = os.path.join(out_path, fn)
	if is_lbl: img = recolor(img)
	cv2.imwrite(path, img)
