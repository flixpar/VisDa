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

if __name__ == "__main__":
	main()
