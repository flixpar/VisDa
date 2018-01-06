import torch
from torch import nn
from torch.utils import data
from torch.autograd import Variable
torch.backends.cudnn.benchmark = True

import numpy as np
import yaml
import os
import cv2

from models.gcn import GCN
from loaders.dataloader import VisDaDataset
from util.metrics import scores, miou, class_iou, print_scores
from util.util import Namespace

import torch.nn.functional as F
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import compute_unary, create_pairwise_bilateral, create_pairwise_gaussian, unary_from_softmax


# config:
config_path = "/home/flixpar/VisDa/config.yaml"
args = Namespace(**yaml.load(open(config_path, 'r')))

samples = 5
epoch = 1

save_path = "/home/flixpar/VisDa/saves/gcn-{}.pth".format(epoch)
out_path = "/home/flixpar/VisDa/pred/"

dataset = VisDaDataset(im_size=args.img_size)
dataloader = data.DataLoader(dataset, batch_size=1, shuffle=True)

model = GCN(dataset.num_classes, dataset.img_size, k=args.K).cuda()
model.load_state_dict(torch.load(save_path))
model.eval()

def main():

	iou = 0

	print("Starting predictions...")
	for i, (image, truth) in enumerate(dataloader):

		if i == samples:
			break

		pred = predict(image)
		# pred = pred_crf(image)
		gt = np.squeeze(truth.cpu().numpy())

		save_anno(pred, i, gt=False)
		save_anno(gt, i, gt=True)

		iou += miou(gt, pred, dataset.num_classes)
		# score, class_iou = scores(gt, pred, dataset.num_classes)
		# print_scores(i+1, score, class_iou)

	iou /= samples
	print("Mean IOU: {}".format(iou))

def predict(img):

	img = Variable(img.cuda())
	output = model(img)
	pred = np.squeeze(output.data.max(1)[1].cpu().numpy())

	return pred

def pred_crf(img):

	img = Variable(img.cuda())
	output = model(img)
	pred = F.log_softmax(output, dim=1)

	image = np.squeeze(img.data.cpu().numpy())
	pred = np.squeeze(pred.data.cpu().numpy())

	d = dcrf.DenseCRF2D(image.shape[1], image.shape[2], 35)

	unary = unary_from_softmax(np.ascontiguousarray(pred))
	unary = np.ascontiguousarray(unary)
	d.setUnaryEnergy(unary)

	# This potential penalizes small pieces of segmentation that are
	# spatially isolated -- enforces more spatially consistent segmentations
	d.addPairwiseGaussian(sxy=(3,3), compat=3, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)

	# This creates the color-dependent features --
	# because the segmentation that we get from CNN are too coarse
	# and we can use local color features to refine them
	d.addPairwiseBilateral(sxy=(80,80), srgb=(13,13,13), rgbim=reverse_img_norm(image),
		compat=10, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
	
	Q = d.inference(5)
	res = np.argmax(Q, axis=0).reshape((image.shape[1], image.shape[2]))
	return res

def reverse_img_norm(image):
	img_mean = np.array([108.56263368194266, 111.92560322135374, 113.01417537462997])
	img_stdev = 60
	image = image.transpose(2, 1, 0)
	image *= img_stdev
	image += img_mean
	image = image.astype(np.uint8)
	image = np.ascontiguousarray(image)
	return image

def recolor(lbl):
	labels = [(0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (20, 20, 20), (111, 74, 0), (81, 0, 81), (128, 64, 128),
	          (244, 35, 232), (250, 170, 160), (230, 150, 140), (70, 70, 70), (102, 102, 156), (190, 153, 153),
	          (180, 165, 180), (150, 100, 100), (150, 120, 90), (153, 153, 153), (153, 153, 153), (250, 170, 30),
	          (220, 220, 0), (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60), (255, 0, 0), (0, 0, 142),
	          (0, 0, 70), (0, 60, 100), (0, 0, 90), (0, 0, 110), (0, 80, 100), (0, 0, 230), (119, 11, 32), (0, 0, 142)]

	out = np.zeros((lbl.shape[0], lbl.shape[1], 3))
	for i in range(len(labels)):
		out[lbl==i] = labels[i]
	return out

def save_anno(lbl, num, gt=True):
	fn = "gt{}.png".format(num) if gt else "pred{}.png".format(num)
	path = os.path.join(out_path, fn)
	col = recolor(lbl)
	cv2.imwrite(path, col)

if __name__ == "__main__":
	main()
