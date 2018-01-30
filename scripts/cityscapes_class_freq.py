import os
from glob import glob

import multiprocessing as mp
import itertools as it

import cv2
import numpy as np

import random

import cityscapes_helper as cityscapes_util

base_path = "/media/data/cityscapes/annotations/train/"
sample_freq = 0.3
processes = 8

image_paths = glob(os.path.join(base_path, "**", "*gtFine_labelIds.png"), recursive=True)
num_imgs = len(image_paths)
num_samples = int(num_imgs * sample_freq)
image_paths = random.sample(image_paths, num_samples)

num_classes=20

def main():

	print("Processing {} of {} images...".format(num_samples, num_imgs))

	pool = mp.Pool(processes)
	data = pool.map(calc_stats, image_paths)

	all_counts = np.asarray(data)
	counts = np.sum(all_counts, axis=0)

	print("counts: {}".format(counts))
	print("sum: {}".format(np.sum(counts)))

	print("weights: {}".format(counts / np.sum(counts)))

def calc_stats(img_path):
	img = cv2.imread(img_path, 0)
	# print(np.unique(img))
	img = transform_labels(img)
	# print(np.unique(img))
	unique, counts = np.unique(img, return_counts=True)

	full_counts = [0]*22
	for ind, c in zip(unique, counts):
		full_counts[ind] = c

	return full_counts

def transform_labels(lbl):
	out = np.zeros((lbl.shape[0], lbl.shape[1]))

	for i in range(num_classes):
		n = cityscapes_util.trainId2label[i].id
		out[lbl == n] = i

	return out.astype(np.uint8)

if __name__ == "__main__":
	main()

