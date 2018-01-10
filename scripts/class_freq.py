import os
from glob import glob

import multiprocessing as mp
import itertools as it

import cv2
import numpy as np

import random

base_path = "/media/data/train/annotations"
sample_freq = 0.2
processes = 8

# labels = [(0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (20, 20, 20), (111, 74, 0), (81, 0, 81), (128, 64, 128),
# 	(244, 35, 232), (250, 170, 160), (230, 150, 140), (70, 70, 70), (102, 102, 156), (190, 153, 153),
# 	(180, 165, 180), (150, 100, 100), (150, 120, 90), (153, 153, 153), (153, 153, 153), (250, 170, 30),
# 	(220, 220, 0), (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60), (255, 0, 0), (0, 0, 142),
# 	(0, 0, 70), (0, 60, 100), (0, 0, 90), (0, 0, 110), (0, 80, 100), (0, 0, 230), (119, 11, 32), (0, 0, 142)]
labels = [
	(0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (20, 20, 20), (0, 74, 111), (81, 0, 81), (128, 64, 128),
	(232, 35, 244), (160, 170, 250), (140, 150, 230), (70, 70, 70), (156, 102, 102), (153, 153, 190),
	(180, 165, 180), (100, 100, 150), (90, 120, 150), (153, 153, 153), (153, 153, 153), (30, 170, 250),
	(0, 220, 220), (35, 142, 107), (152, 251, 152), (180, 130, 70), (60, 20, 220), (0, 0, 255), (142, 0, 0),
	(70, 0, 0), (100, 60, 0), (90, 0, 0), (110, 0, 0), (100, 80, 0), (230, 0, 0), (32, 11, 119), (142, 0, 0)
]
ignore_labels = [0, 1, 2, 3]

image_paths = glob(os.path.join(base_path, "*.png"))
num_imgs = len(image_paths)
num_samples = int(num_imgs * sample_freq)
image_paths = random.sample(image_paths, num_samples)

def main():

	print("Processing {} images...".format(num_samples))

	pool = mp.Pool(processes)
	data = pool.map(calc_stats, image_paths)

	all_counts = np.asarray(data)
	counts = np.sum(all_counts, axis=0)

	print("counts: {}".format(counts))
	print("sum: {}".format(np.sum(counts)))

def calc_stats(img_path):
	img = cv2.imread(img_path)
	img = transform_labels(img)
	unique, counts = np.unique(img, return_counts=True)

	full_counts = [0]*35
	for ind, c in zip(unique, counts):
		full_counts[ind] = c

	return full_counts

def transform_labels(lbl):
	out = np.zeros((lbl.shape[0], lbl.shape[1]))
	for i, col in enumerate(labels):
		if i in ignore_labels: continue
		out[np.where(np.all(lbl == col, axis=-1))] = i
	return out.astype(int)

if __name__ == "__main__":
	main()

