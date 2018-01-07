import torch
from torch import nn
from torch.utils import data
from torch.autograd import Variable
torch.backends.cudnn.benchmark = True

import numpy as np
import yaml
import os
import cv2
np.seterr(divide='ignore', invalid='ignore')

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
epoch = 5

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

		# pred = predict(image)
		pred = pred_crf(image)
		gt = np.squeeze(truth.cpu().numpy())

		save_anno(pred, i, gt=False)
		save_anno(gt, i, gt=True)

		iou += miou(gt, pred, dataset.num_classes)
		print_scores(gt, pred, dataset.num_classes, i+1)

	iou /= samples
	print()
	print()
	print("Mean IOU: {}".format(iou))

def predict(img):

	img = Variable(img.cuda())
	output = model(img)
	pred = np.squeeze(output.data.max(1)[1].cpu().numpy())

	return pred

def pred_crf(img):

	# initial prediction
	img = Variable(img.cuda())
	output = model(img)
	pred = F.log_softmax(output, dim=1)

	# reformat outputs
	img = np.squeeze(img.data.cpu().numpy())
	img = reverse_img_norm(img)
	pred = np.squeeze(pred.data.cpu().numpy())

	# init vars
	num_cls = pred.shape[0]
	scale = 0.6
	clip = 1e-9

	# init crf
	d = dcrf.DenseCRF2D(img.shape[0], img.shape[1], n_labels)

	# create unary
	uniform = np.ones(pred.shape) / num_cls
	U = scale * pred + (1 - scale) * uniform
	U = np.clip(U, clip, 1.0)
	U = -np.log(U).reshape([num_cls, -1]).astype(np.float32)

	d.setUnaryEnergy(U)

	# create pairwise
	d.addPairwiseGaussian(sxy=(3,3), compat=3, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
	d.addPairwiseBilateral(sxy=(80,80), srgb=(13,13,13), rgbim=img, compat=10, kernel=dcrf.DIAG_KERNEL,
		normalization=dcrf.NORMALIZE_SYMMETRIC)

	# inference
	Q = d.inference(5)
	res = np.argmax(Q, axis=0).reshape((img.shape[0], img.shape[1]))

	return res

def reverse_img_norm(image):
	img_mean = np.array([108.56263368194266, 111.92560322135374, 113.01417537462997])
	img_stdev = 60
	image = image.transpose(2, 1, 0)
	image *= img_stdev
	image += img_mean
	image = image.astype(np.uint8)
	return image

def recolor(lbl):
	labels = [(0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (20, 20, 20), (0, 74, 111), (81, 0, 81), (128, 64, 128),
			(232, 35, 244), (160, 170, 250), (140, 150, 230), (70, 70, 70), (156, 102, 102), (153, 153, 190),
			(180, 165, 180), (100, 100, 150), (90, 120, 150), (153, 153, 153), (153, 153, 153), (30, 170, 250),
			(0, 220, 220), (35, 142, 107), (152, 251, 152), (180, 130, 70), (60, 20, 220), (0, 0, 255), (142, 0, 0),
			(70, 0, 0), (100, 60, 0), (90, 0, 0), (110, 0, 0), (100, 80, 0), (230, 0, 0), (32, 11, 119), (142, 0, 0)]

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
