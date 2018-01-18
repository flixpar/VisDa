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
from loaders.eval_dataloader import EvalDataloader

import util.cityscapes_helper as cityscapes
from util.metrics import miou, class_iou, confusion_matrix
from util.util import *

args = Namespace(**yaml.load(open(os.path.join(os.getcwd(), "config.yaml"), 'r')))
args.img_size = (int(args.scale_factor*args.default_img_size[0]), int(args.scale_factor * args.default_img_size[1]))

paths = yaml.load(open(os.path.join(os.getcwd(), "paths.yaml"), 'r'))


class Evaluator:

	def __init__(self, mode="val", samples=30, metrics=["miou", "cls_iou"], crf=True, standalone=False, save_pred=False, per_image=False):
		self.n_samples = samples
		self.metrics = metrics
		self.use_crf = crf
		self.standalone = standalone
		self.save_pred = save_pred
		self.per_image = per_image

		if mode == "val":
			self.dataset = VisDaDataset(im_size=args.img_size)
		elif mode == "cityscapes":
			self.dataset = CityscapesDataset(im_size=args.img_size)
		else:
			raise ValueError("Invalid mode.")

		self.dataloader = data.DataLoader(EvalDataloader(self.dataset, self.n_samples), batch_size=1, shuffle=True, num_workers=6)

	def eval(self, model):
		model.eval()

		if "miou" in self.metrics:
			iou = []
		if "cls_iou" in self.metrics:
			cls_iou = np.zeros(self.dataset.num_classes)
		if "cfm" in self.metrics:
			cfm = np.zeros((self.dataset.num_classes, self.dataset.num_classes))
		if "classmatch" in self.metrics:
			good = np.zeros((self.n_samples, self.dataset.num_classes), dtype=np.bool)

		loader = self.dataloader
		if self.standalone: loader = tqdm(loader, total=self.n_samples)

		for i, ((image, _), (image_full, gt)) in enumerate(loader):

			image_full = np.squeeze(image_full.cpu().numpy())
			gt = np.squeeze(gt.cpu().numpy())

			pred = self.predict(model, image)
			pred = self.upsample(pred)

			if self.use_crf:
				pred_alt = np.argmax(pred.copy(), axis=0)
				pred = self.refine(pred, image_full)
			pred = np.argmax(pred, axis=0)

			if self.save_pred:
				path = os.path.join(paths["project_path"], "pred")
				if self.use_crf:
					save_set(image_full, gt, pred_alt, pred, i+1, path)
				else:
					save_set(image_full, gt, pred, None, i+1, path)

			img_clsiou = class_iou(gt, pred, self.dataset.num_classes)
			img_miou = np.nanmean(img_clsiou)
			img_clsiou = np.nan_to_num(img_clsiou)
			img_cfm = confusion_matrix(gt.flatten(), pred.flatten(), self.dataset.num_classes, normalize=False)

			if "miou" in self.metrics:
				iou.append(img_miou)
			if "cls_iou" in self.metrics:
				cls_iou = cls_iou + img_clsiou
			if "cfm" in self.metrics:
				cfm = cfm + img_cfm
			if "classmatch" in self.metrics:
				k = 0.01 * pred.size
				a,b = np.unique(pred, return_counts=True)
				p = dict(zip(a,b))
				g = np.unique(gt)
				matches = [((i in g) == (i in p.keys() and p[i]>k)) for i in range(self.dataset.num_classes)]
				good[i] = matches


			if self.per_image and self.standalone:
				tqdm.write("Image {}".format(i+1))
				tqdm.write("mIOU: {}".format(miou(gt, pred, self.dataset.num_classes, ignore_zero=False)))
				tqdm.write("class IOU: {}".format(list(class_iou(gt, pred, self.dataset.num_classes))))
				tqdm.write("matches: {}".format((list(good[i]))))
				tqdm.write("\n")

		
		res = []
		if "miou" in self.metrics:
			meaniou = np.asarray(iou).mean()
			stdeviou = np.asarray(iou).std()
			res.append((meaniou, stdeviou))
		if "cls_iou" in self.metrics:
			cls_iou /= self.n_samples
			res.append(cls_iou)
		if "cfm" in self.metrics:
			cfm = cfm.astype('float') / cfm.sum(axis=1)[:, np.newaxis]
			res.append(cfm)
		if "classmatch" in self.metrics:
			res.append(good)

		model.train()
		return tuple(res)

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

	def refine(self, pred, img):

		# init vars
		num_cls = pred.shape[0]
		scale = 0.97
		clip = 1e-8

		# init crf
		d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], num_cls)

		# create unary
		uniform = np.ones(pred.shape) / num_cls
		U = (scale * pred) + ((1 - scale) * uniform)
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

	eval_args = yaml.load(open(os.path.join(os.getcwd(), "eval.yaml"), 'r'))
	args = Namespace(**yaml.load(open(os.path.join(paths["project_path"], "saves{}".format(eval_args["version"]), "config.yaml"), 'r')))

	trained_epochs = eval_args["epochs"]
	save_path = os.path.join(paths["project_path"], "saves{}".format(eval_args["version"]), "{}-{}.pth".format(args.model, trained_epochs))

	print()
	print("size:\t{}".format(args.img_size))
	print("scale factor:\t{}".format(args.scale_factor))
	print("batch size:\t{}".format(args.batch_size))
	print("K:\t{}".format(args.K))
	print()
	print("version:\t{}".format(eval_args["version"]))
	print("epochs: \t{}".format(eval_args["epochs"]))
	print("samples:\t{}".format(eval_args["samples"]))
	print("used CRF:\t{}".format(eval_args["crf"]))
	print()

	if args.model=="GCN": model = GCN(cityscapes.num_classes, args.img_size, k=args.K).cuda()
	elif args.model=="UNet": model = UNet(cityscapes.num_classes).cuda()
	else: raise ValueError("Invalid model arg.")
	model.load_state_dict(torch.load(save_path))

	start = time.time()

	evaluator = Evaluator(mode=eval_args["mode"], samples=eval_args["samples"], crf=eval_args["crf"],
		metrics=["miou","cls_iou","classmatch"], standalone=True, save_pred=eval_args["save_pred"], per_image=eval_args["per_image"])
	iou, cls_iou, matches = evaluator.eval(model)

	end = time.time()

	print("Took {} seconds.".format(end-start))
	print()
	print("Mean IOU: {}".format(iou))
	print("Mean class IOU:")
	for i in cls_iou:
		print(i)
	print(matches)
	print()

