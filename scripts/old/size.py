import cv2
import glob
import os

base_dir = "/home/flixpar/data/train/annotations"
filelist = sorted(glob.glob(os.path.join(base_dir, "*.png")))

good = 0
bad = 0

total = len(filelist)

for i, fn in enumerate(filelist):
    img = cv2.imread(fn, 0)
    if (img.shape != (1052, 1914)):
        print(fn)
        bad += 1
    else:
        good += 1

    if (i % 100 == 0):
        print("good: {}, bad: {}, checked: {}/{}".format(good, bad, good+bad, total))
