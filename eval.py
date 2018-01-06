import torch
from torch import nn
from torch.utils import data
from torch.autograd import Variable
torch.backends.cudnn.benchmark = True

import numpy as np
import yaml
import os

from models.gcn import GCN
from loaders.dataloader import VisDaDataset
from util.metrics import scores, miou, class_iou, print_scores
from util.util import Namespace

import torch.nn.functional as F
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import compute_unary, create_pairwise_bilateral, create_pairwise_gaussian, softmax_to_unary


# config:
config_path = "/home/flixpar/VisDa/config.yaml"
args = Namespace(**yaml.load(open(config_path, 'r')))

samples = 50
epoch = 8

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
		gt = np.squeeze(Variable(truth).data.cpu().numpy())

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
	pred = np.squeeze(output.cpu().numpy())
	pred = F.log_softmax(inputs, dim=0)

	image = img.cpu().numpy()
	pred = pred.cpu().numpy()

	d = dcrf.DenseCRF2D(image.shape[0], image.shape[1], 35)

	unary = softmax_to_unary(pred)
	unary = np.ascontiguousarray(unary)
	d.setUnaryEnergy(unary)

	# This potential penalizes small pieces of segmentation that are
	# spatially isolated -- enforces more spatially consistent segmentations
	d.addPairwiseGaussian(sxy=(3,3), compat=3, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)

	# This creates the color-dependent features --
	# because the segmentation that we get from CNN are too coarse
	# and we can use local color features to refine them
	d.addPairwiseBilateral(sxy=(80,80), srgb=(13,13,13), rgbim=image, compat=10,
		kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
	
	Q = d.inference(5)
	res = np.argmax(Q, axis=0).reshape((image.shape[0], image.shape[1]))

	return res


if __name__ == "__main__":
	main()
