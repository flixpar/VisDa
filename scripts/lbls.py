import os
import glob
import cv2
import random
import numpy as np

path = "/home/flixpar/data/train/annotations"
filelist = glob.glob(os.path.join(path, "*.png"))

ind = random.sample(range(len(filelist)), 75)

unique = set()

for i in ind:
	img = cv2.imread(filelist[i])
	#un = np.unique(img, axis=0)
	#print(un)
	#print(set( tuple(v) for m2d in img for v in m2d ))
	unique = unique | set( tuple(v) for m2d in img for v in m2d )

print(unique)
