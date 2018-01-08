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
from models.unet import UNet
from loaders.dataloader import VisDaDataset
from util.metrics import scores, miou, class_iou, print_scores
from util.util import *

import torch.nn.functional as F
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import create_pairwise_bilateral, create_pairwise_gaussian


# config:
config_path = "/home/flixpar/VisDa/config.yaml"
args = Namespace(**yaml.load(open(config_path, 'r')))
args.img_size = (int(args.scale_factor*args.default_img_size[0]), int(args.scale_factor * args.default_img_size[1]))

samples = 30
epoch = 10

gcn_save_path = "/home/flixpar/VisDa/saves_gcn/gcn-{}.pth".format(epoch)
unet_save_path = "/home/flixpar/VisDa/saves_unet/unet-{}.pth".format(epoch)
out_path = "/home/flixpar/VisDa/pred/"

dataset = VisDaDataset(im_size=args.img_size, mode="eval")
dataloader = data.DataLoader(dataset, batch_size=1, shuffle=True)

gcn_model = GCN(dataset.num_classes, dataset.img_size, k=args.K).cuda()
unet_model = UNet(dataset.num_classes).cuda()

gcn_model.load_state_dict(torch.load(gcn_save_path))
gcn_model.eval()

unet_model.load_state_dict(torch.load(unet_save_path))
unet_model.eval()

def main():

	iou = 0

	print("Starting predictions...")
	for i, (image, truth) in enumerate(dataloader):

		if i == samples:
			break

		pred = predict(image)
		gt = np.squeeze(truth.cpu().numpy())

		iou += miou(gt, pred, dataset.num_classes)
		cls_iou = cls_iou + class_iou(gt, pred, dataset.num_classes)

	iou /= samples
	cls_iou /= samples
	print()
	print("Mean IOU w/ CRF & Ensemble: {}".format(ioucrf))
	print("Class IOU w/ CRF & Ensemble: {}".format(cls_iou))

def predict(img):

	# GCN prediction
	img = Variable(img.cuda())
	output_gcn = gcn_model(img)
	output_unet = F.softmax(output, dim=1)

	# GCN prediction
	output_unet = unet_model(img)
	output_unet = F.softmax(output, dim=1)

	pred = (output_gcn + output_unet) / 2

	# reformat outputs
	img = np.squeeze(img.data.cpu().numpy())
	img = reverse_img_norm(img)
	pred = np.squeeze(pred.data.cpu().numpy())

	# init vars
	num_cls = pred.shape[0]
	scale = 0.97
	clip = 1e-9

	# init crf
	d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], num_cls)

	# create unary
	uniform = np.ones(pred.shape) / num_cls
	U = scale * pred + (1 - scale) * uniform
	U = np.clip(U, clip, 1.0)
	U = -np.log(U).reshape([num_cls, -1]).astype(np.float32)

	d.setUnaryEnergy(U)

	# create pairwise
	d.addPairwiseGaussian(sxy=(3,3), compat=3, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
	d.addPairwiseBilateral(sxy=(40,40), srgb=(15,15,15), rgbim=np.ascontiguousarray(img), compat=10, kernel=dcrf.DIAG_KERNEL,
		normalization=dcrf.NORMALIZE_SYMMETRIC)

	# inference
	Q = d.inference(4)
	res = np.argmax(Q, axis=0).reshape((img.shape[0], img.shape[1]))

	return res


if __name__ == "__main__":
	main()
