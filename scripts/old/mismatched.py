import cv2
import os
import glob
import tqdm

base_dir = "/home/flixpar/data/train"
img_list = glob.glob(os.path.join(base_dir, "images", "*.png"))
lbl_list = [fn.replace("images", "annotations") for fn in img_list]

names_file = open("log.txt", "w")

for img_fn, lbl_fn in tqdm.tqdm(zip(img_list, lbl_list), total=len(img_list)):
    img = cv2.imread(img_fn, 0)
    lbl = cv2.imread(lbl_fn, 0)
    if (img.shape != lbl.shape):
        tqdm.tqdm.write(img_fn)
        names_file.write(img_fn)
        names_file.write("\n")

names_file.close()
