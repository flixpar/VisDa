import os
from glob import glob

import multiprocessing as mp
import itertools as it

import cv2
import numpy as np
from skimage.exposure import equalize_adapthist, rescale_intensity

import random

base_path = "/media/data/cityscapes/"
sample_freq = 0.3
processes = 6

image_paths = glob.glob(os.path.join(base_path, "images", "val", "**", "*.png"), recursive=True)
num_imgs = len(image_paths)
num_samples = int(num_imgs * sample_freq)
image_paths = random.sample(image_paths, num_samples)

print("Sampling {} images...".format(num_samples))

def main():
	pool = mp.Pool(processes)
	data = pool.map(calc_stats, image_paths)

	mean = np.mean(data)
	stdev = np.std(data)

	print("Means: {}".format(mean))
	print("Standard deviations: {}".format(stdev))

def calc_stats(img_path):
	img = cv2.imread(img_path)
	img = enhance_contrast(img)
	mean = np.mean(img, axis=(0,1))
	return mean

def enhance_contrast(img):
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	img = equalize_adapthist(img)
	img = rescale_intensity(img, out_range='uint8').astype(np.uint8)
	img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
	return img

if __name__ == "__main__":
	main()

