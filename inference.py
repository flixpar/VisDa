import torch
from torch import nn
from torch.utils import data
from torch.autograd import Variable
torch.backends.cudnn.benchmark = True
from gcn import GCN
import numpy as np

from dataloader import VisDaDataset
from metrics import scores

def predict(img):

	img = Variable(img.cuda())
	output = model(img)
	pred = np.squeeze(output.data.max(1)[1].cpu().numpy())

	return pred

def print_scores(pred, gt, i=0):
	if i: print("### Image {} ###".format(i))
	score, class_iou = scores(gt, pred, dataset.num_classes)
	for key, val in score.items():
		print("{}{}".format(key, val))
	for key, val in class_iou.items():
		if not np.isnan(val):
			print("{}:\t{}".format(key, val))
	print()