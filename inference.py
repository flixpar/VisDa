import torch
from torch import nn
from torch.utils import data
from torch.autograd import Variable
import torch.nn.functional as F
torch.backends.cudnn.benchmark = True

import os
import yaml
from tqdm import tqdm

import numpy as np
np.seterr(divide='ignore', invalid='ignore')

from models.gcn import GCN
from models.gcn_densenet import GCN_DENSENET
from models.gcn_deconv import GCN_DECONV
from models.gcn_psp import GCN_PSP
from models.gcn_comb import GCN_COMBINED
from models.gcn_resnext import GCN_RESNEXT
from models.pspnet import PSPNet
from models.unet import UNet

from loaders.test import TestDataset

from util.util import *
from util.setup import *


class Evaluator:

	def __init__(self, dataset):
		self.dataset = dataset
		self.img_size = self.dataset.img_size

		self.dataloader = data.DataLoader(self.dataset, batch_size=1, shuffle=False, num_workers=6)

	def eval(self, model):
		model.eval()
		save_path = os.path.join(os.getcwd(), "pred")

		loader = tqdm(self.dataloader, total=self.n_samples)
		for i, l in enumerate(loader):
			l = list(l)

			# unpack values from loader
			image		= l[0]
			name		= l[1][0]

			# augment with flip
			img_flip = self.flip_tensor(image, 3)

			# prepare data
			img = Variable(img.cuda(), volatile=True)
			img_flip = Variable(img_flip.cuda(), volatile=True)

			# inference on original
			output = model(img)
			pred = F.softmax(output, dim=1).cpu()
			pred = self.upsample(pred)

			# inference on flipped
			output = model(img_flip)
			pred_flip = F.softmax(output, dim=1).cpu()
			pred_flip = self.upsample(pred_flip)

			# merge predictions
			pred_flip = np.flip(pred_flip, 2)
			pred = (pred/pred.mean()) + (pred_flip/pred_flip.mean())

			# compute output prediction
			pred = np.argmax(pred, axis=0)

			# save output image
			pred_out = self.recolor(pred)
			cv2.imwrite(pred_out, name)

	def upsample(self, img):
		size = self.dataset.default_size
		upsampler = nn.Upsample(size=size, mode='bilinear')
		out = upsampler(img)
		out = np.squeeze(out.data.cpu().numpy())
		return out

	def flip_tensor(self, x, dim):
		dim = x.dim() + dim if dim < 0 else dim
		inds = tuple(slice(None, None) if i != dim else x.new(torch.arange(x.size(i)-1, -1, -1).tolist()).long() for i in range(x.dim()))
		return x[inds]

	def recolor(self, lbl):
		out = np.zeros((lbl.shape[0], lbl.shape[1], 3))
		for label in visda.labels:
			out[lbl==label.trainId] = label.color
		return out


if __name__ == "__main__":

	version = yaml.load(open("eval.yaml",'r'))["version"]
	args = load_args(os.path.join(os.getcwd(), "saves{}".format(version)), eval_path=os.getcwd())

	trained_epochs = args.eval["epochs"]
	save_path = os.path.join(args.paths["project_path"], "saves{}".format(args.eval["version"]), "{}-{}.pth".format(args.model, trained_epochs))

	dataset = TestDataset()

	if args.model=="GCN":				model = GCN(cityscapes.num_classes, args.img_size, k=args.K).cuda()
	elif args.model=="GCN_DENSENET":	model = GCN_DENSENET(cityscapes.num_classes, args.img_size, k=args.K).cuda()
	elif args.model=="GCN_DECONV":		model = GCN_DECONV(cityscapes.num_classes, args.img_size, k=args.K).cuda()
	elif args.model=="GCN_PSP":			model = GCN_PSP(cityscapes.num_classes, args.img_size, k=args.K).cuda()
	elif args.model=="GCN_COMB":		model = GCN_COMBINED(cityscapes.num_classes, args.img_size).cuda()
	elif args.model=="GCN_RESNEXT":		model = GCN_RESNEXT(dataset.num_classes, k=self.args.K).cuda()
	elif args.model=="UNet":			model = UNet(cityscapes.num_classes).cuda()
	elif args.model=="PSPNet":			model = PSPNet(cityscapes.num_classes, args.img_size).cuda()
	else: raise ValueError("Invalid model arg.")
	model.load_state_dict(torch.load(save_path))

	evaluator = Evaluator(dataset)
	evaluator.eval(model)

