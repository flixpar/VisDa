import os
from glob import glob
import random
from shutil import copyfile

num_samples = 200
src_path = "/media/data/cityscapes/"
dest_path = "/home/flixpar/eval/"

image_fnlist = glob.glob(os.path.join(src_path, "images", "val", "**", "*.png"), recursive=True)
image_fnlist = random.sample(image_fnlist, num_samples)
label_fnlist = [fn.replace("images", "annotations").replace("leftImg8bit", "gtFine_labelIds") for fn in self.image_fnlist]

for i, (img_fn, lbl_fn) in enumerate(zip(image_fnlist, label_fnlist)):
	print("{} --> {}".format(img_fn, i))

	img_dest = os.path.join(dest_path, "{}_img.png".format(i))
	copyfile(img_fn, img_dest)
	lbl_dest = os.path.join(dest_path, "{}_lbl.png".format(i))
	copyfile(lbl_fn, lbl_dest)
