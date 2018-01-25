import os
from glob import glob

import multiprocessing as mp
import itertools as it

import cv2
import numpy as np

import random

base_path = "/media/data/train/images"
sample_freq = 0.1
processes = 6

image_paths = glob(os.path.join(base_path, "*.png"))
num_imgs = len(image_paths)
num_samples = int(num_imgs * sample_freq)
image_paths = random.sample(image_paths, num_samples)

print("Sampling {} of {} images...".format(num_samples, num_imgs))

def main():
    pool = mp.Pool(processes)
    data = pool.map(calc_stats, image_paths)

    means, stdevs = zip(*data)
    mean = np.mean(means, axis=0)
    stdev = np.mean(stdevs, axis=0)

    print("Means: {}".format(mean))
    print("Standard deviations: {}".format(stdev))

def calc_stats(img_path):
    img = cv2.imread(img_path)
    mean = np.mean(img, axis=(0,1))
    stdev = np.std(img, axis=(0,1))
    return mean, stdev

if __name__ == "__main__":
    main()

