import os
from glob import glob
import random

num_samples = 50
base_path = "/media/data/train"
dest_path = "/media/data/train/eval"

img_fnlist = glob(os.path.join(base_path, "images", "*.png"))
img_fnlist = random.sample(img_fnlist, num_samples)

for img_fn in img_fnlist:
	img_dest = img_fn.replace("images", "eval/images")
	os.rename(img_fn, img_dest)
	lbl_fn = img_fn.replace("images", "annotations")
	lbl_dest = lbl_fn.replace("annotations", "eval/annotations")
	os.rename(lbl_fn, lbl_dest)
