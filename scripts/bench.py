import cv2
from PIL import Image
from timeit import default_timer as timer
import numpy as np

filename = "img.jpg"

cv_time = 0
pil_time = 0

for i in range(1000):

	if i % 2 == 0:

		start = timer()
		img = cv2.imread(filename)
		mean = np.mean(img)
		end = timer()

		cv_time += (end - start)

	else:

		start = timer()
		img = np.asarray(Image.open(filename))
		mean = np.mean(img)
		end = timer()

		pil_time += (end - start)


cv_time /= 1000
pil_time /= 1000

print(cv_time)
print(pil_time)