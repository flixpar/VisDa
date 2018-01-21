import torch
from torch import nn
from torch.utils import data
from torch.autograd import Variable
import torch.nn.functional as F
torch.backends.cudnn.benchmark = True

import pydensecrf.densecrf as dcrf

import os
import yaml
import time
from tqdm import tqdm

import numpy as np
np.seterr(divide='ignore', invalid='ignore')

from models.gcn import GCN
from models.unet import UNet

from loaders.visda import VisDaDataset
from loaders.cityscapes import CityscapesDataset
from loaders.cityscapes_select import CityscapesSelectDataset
from loaders.eval_dataloader import EvalDataloader

import util.cityscapes_helper as cityscapes
from util.metrics import calc_miou, calc_class_iou, confusion_matrix
from util.util import *
from util.scores import Scorer
from util.setup import *


class Evaluator:

	def __init__(self, dataset, samples=30, metrics=["miou", "cls_iou"], crf=True, standalone=False, save_pred=False, per_image=False, pred_path=None):
		self.n_samples = samples
		self.metrics = metrics
		self.use_crf = crf
		self.standalone = standalone
		self.save_pred = save_pred
		self.per_image = per_image
		self.dataset = dataset
		self.img_size = self.dataset.img_size

		self.dataloader = data.DataLoader(self.dataset, batch_size=1, shuffle=False, num_workers=6)

	def eval(self, model):
		model.eval()

		scores = Scorer(self.metrics)

		loader = self.dataloader
		if self.standalone: loader = tqdm(loader, total=self.n_samples)

		for i, l in enumerate(loader):
			l = list(l)

			# unpack values from loader
			image 		= l[0]
			image_full 	= l[2]
			gt 			= l[3]
			name 		= l[4] if len(l)>4 else None

			# convert inputs as needed
			image_full = np.squeeze(image_full.cpu().numpy())
			gt = np.squeeze(gt.cpu().numpy())

			# inference on image
			pred = self.predict(model, image)
			pred = self.upsample(pred)

			# refine prediction with CRF
			if self.use_crf:
				if self.save_pred: alt = np.argmax(pred.copy(), axis=0)
				pred = self.refine(pred, image_full)

			# compute output prediction
			pred = np.argmax(pred, axis=0)

			# evaluate
			scores.update(gt, pred)

			# display metrics
			if self.standalone and self.per_image:
				self.display_latest(name)

			# save output images
			if self.save_pred:
				path = os.path.join(paths["project_path"], "pred")
				if self.use_crf:
					save_set(image_full, gt, alt, pred, i+1, path)
				else:
					save_set(image_full, gt, pred, None, i+1, path)
		

		model.train()
		return scores.final_scores()

	def predict(self, model, img):
		img = Variable(img.cuda())
		output = model(img)
		pred = F.softmax(output, dim=1).cpu()
		return pred

	def upsample(self, img):
		size = self.dataset.default_size
		upsampler = nn.Upsample(size=size, mode='bilinear')
		out = upsampler(img)
		out = np.squeeze(out.data.cpu().numpy())
		return out

	def display_latest(self, num, name, scores):
		if not (self.per_image and self.standalone):
			return

		tqdm.write("Image {}".format(num+1))
		if name is not None: tqdm.write("Name: {}".format(name))
		tqdm.write(scores.latest_to_string())
		tqdm.write("\n")

	def refine(self, pred, img):

		# init vars
		num_cls = pred.shape[0]
		trust = 0.98
		clip = 1e-8

		# init crf
		d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], num_cls)

		# create unary
		uniform = np.ones(pred.shape) / num_cls
		U = (trust * pred) + ((1 - trust) * uniform)
		U = np.clip(U, clip, 1.0)
		U = -np.log(U).reshape([num_cls, -1]).astype(np.float32)

		d.setUnaryEnergy(U)

		# create pairwise
		d.addPairwiseGaussian(sxy=(3,3), compat=3, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
		d.addPairwiseBilateral(sxy=(40,40), srgb=(15,15,15), rgbim=np.ascontiguousarray(img), compat=10, kernel=dcrf.DIAG_KERNEL,
			normalization=dcrf.NORMALIZE_SYMMETRIC)

		# inference
		Q = d.inference(4)
		res = np.array(Q).reshape((num_cls, img.shape[0], img.shape[1]))

		return res


if __name__ == "__main__":

	args = load_args(os.getcwd())

	trained_epochs = args.eval["epochs"]
	save_path = os.path.join(paths["project_path"], "saves{}".format(args.eval["version"]), "{}-{}.pth".format(args.model, trained_epochs))

	print()
	print("size:\t{}".format(args.img_size))
	print("scale factor:\t{}".format(args.scale_factor))
	print("batch size:\t{}".format(args.batch_size))
	print("K:\t{}".format(args.K))
	print()
	print("version:\t{}".format(args.eval["version"]))
	print("epochs: \t{}".format(args.eval["epochs"]))
	print("samples:\t{}".format(args.eval["samples"]))
	print("used CRF:\t{}".format(args.eval["crf"]))
	print()

	dataset = CityscapesSelectDataset(im_size=img_size, n_samples=self.n_samples)

	if args.model=="GCN": model = GCN(cityscapes.num_classes, args.img_size, k=args.K).cuda()
	elif args.model=="UNet": model = UNet(cityscapes.num_classes).cuda()
	else: raise ValueError("Invalid model arg.")
	model.load_state_dict(torch.load(save_path))

	start = time.time()

	evaluator = Evaluator(dataset, mode=args.eval["mode"], samples=args.eval["samples"], crf=args.eval["crf"],
		metrics=["miou","cls_iou","classmatch"], standalone=True, save_pred=args.eval["save_pred"], per_image=args.eval["per_image"])
	iou, cls_iou, matches = evaluator.eval(model)

	end = time.time()

	print("Took {} seconds.".format(end-start))
	print()
	print("Mean IOU: {}".format(iou))
	print()
	print("Mean class IOU:")
	for i in cls_iou:
		print(i)
	print()
	print("Matches:")
	print(matches)
	print()

