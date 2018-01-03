import os
from glob import glob

import multiprocessing as mp
import itertools as it

import cv2
import numpy as np

import random

base_path = "/home/flixpar/data/train/images"
sample_freq = 0.2
processes = 4

image_paths = glob(os.path.join(base_path, "*.png"))
num_imgs = len(image_paths)
num_samples = int(num_imgs * sample_freq)
image_paths = random.sample(image_paths, num_samples)

print("Sampling {} images...".format(num_samples))

def main():
    pool = mp.Pool(processes)
    data = pool.map(calc_stats, image_paths)

    avgs, stdevs = zip(*data)
    avgs_0, avgs_1, avgs_2 = zip(*avgs)
    stds_0, stds_1, stds_2 = zip(*stdevs)

    avg_0 = sum(avgs_0)/len(avgs_0)
    avg_1 = sum(avgs_1)/len(avgs_1)
    avg_2 = sum(avgs_2)/len(avgs_2)

    stds_0 = sum(stds_0)/len(stds_0)
    stds_1 = sum(stds_1)/len(stds_1)
    stds_2 = sum(stds_2)/len(stds_2)

    mean = (avg_0, avg_1, avg_2)
    stdev = (stds_0, stds_1, stds_2)

    print("Means: {}".format(mean))
    print("Standard deviations: {}".format(stdev))

def calc_stats(img_path):
    img = cv2.imread(img_path)
    mean = np.mean(img, axis=(0,1))
    stdev = np.std(img, axis=(0,1))
    return (mean, stdev)

if __name__ == "__main__":
    main()

