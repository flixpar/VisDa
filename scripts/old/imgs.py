import os
import glob
import cv2
import random

path = "/home/flixpar/data/train/images"
filelist = glob.glob(os.path.join(path, "*.png"))

indecies = random.sample(range(len(filelist)), 50)

total_min = 255
total_max = 0
avg = 0

for i in indecies:
	img = cv2.imread(filelist[i])
	img_min = img.min()
	img_max = img.max()
	img_mean = img.mean(axis=0)

	total_min = min(total_min, img_min)
	total_max = max(total_max, img_max)
	avg += img_mean

print("min:")
print(total_min)
print("max:")
print(total_max)
print("avg:")
print(avg/50)
