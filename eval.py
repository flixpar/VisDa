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

samples = 5
epoch = 10

save_path = "/home/flixpar/VisDa/saves/gcn-{}.pth".format(epoch)
out_path = "/home/flixpar/VisDa/pred/"

dataset = VisDaDataset(im_size=args.img_size, mode="eval")
dataloader = data.DataLoader(dataset, batch_size=1, shuffle=True)

if args.model=="GCN": model = GCN(dataset.num_classes, dataset.img_size, k=args.K).cuda()
elif args.model=="UNet": model = UNet(dataset.num_classes).cuda()
else: raise ValueError("Invalid model arg.")

model.load_state_dict(torch.load(save_path))
model.eval()

def main():

	iou = 0

	print("Starting predictions...")
	for i, (image, truth) in enumerate(dataloader):

		if i == samples:
			break

		pred = predict(image)
		predcrf = pred_crf(image)
		gt = np.squeeze(truth.cpu().numpy())

		save_img(predcrf, "predcrf", i, out_path, is_lbl=True)
		save_img(pred, "pred", i, out_path, is_lbl=True)
		save_img(gt, "gt", i, out_path, is_lbl=True)
		save_img(reverse_img_norm(np.squeeze(image.cpu().numpy())), "src", i, out_path, is_lbl=False)

		iou += miou(gt, predcrf, dataset.num_classes)
		print_scores(gt, predcrf, dataset.num_classes, i+1)

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
	scale = 0.85
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
	d.addPairwiseBilateral(sxy=(80,80), srgb=(13,13,13), rgbim=np.ascontiguousarray(img), compat=10, kernel=dcrf.DIAG_KERNEL,
		normalization=dcrf.NORMALIZE_SYMMETRIC)

	# inference
	Q = d.inference(5)
	res = np.argmax(Q, axis=0).reshape((img.shape[0], img.shape[1]))

	return res


if __name__ == "__main__":
	main()
