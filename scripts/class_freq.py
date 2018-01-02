import os
from glob import glob

import multiprocessing as mp
import itertools as it

import cv2
import numpy as np

import random

base_path = "/home/flixpar/data/VisDa/train/annotations"
sample_freq = 0.3
processes = 5

labels = [(0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (20, 20, 20), (111, 74, 0), (81, 0, 81), (128, 64, 128),
	          (244, 35, 232), (250, 170, 160), (230, 150, 140), (70, 70, 70), (102, 102, 156), (190, 153, 153),
	          (180, 165, 180), (150, 100, 100), (150, 120, 90), (153, 153, 153), (153, 153, 153), (250, 170, 30),
	          (220, 220, 0), (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60), (255, 0, 0), (0, 0, 142),
	          (0, 0, 70), (0, 60, 100), (0, 0, 90), (0, 0, 110), (0, 80, 100), (0, 0, 230), (119, 11, 32), (0, 0, 142)]


image_paths = glob(os.path.join(base_path, "*.png"))
img_ids = random.sample(len(image_paths), len(image_paths)*sample_freq)
image_paths = [pth if i in img_ids for (i, pth) in enumerate(image_paths)]

pool = mp.Pool(processes)
data = pool.map(calc_stats, image_paths)

all_counts = np.array()
for counts in data:
	np.append(all_counts, counts)
counts = np.sum(all_counts, axis=1)

print("counts: {}".format(counts))

def calc_stats(img_path):
	img = cv2.imread(img_path,0)
	unique, counts = np.unique(img, return_counts=True, axis=2)
	full_counts = [counts[i] if i in unique else 0 for i in range(len(labels))]
	return full_counts
