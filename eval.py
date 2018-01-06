import torch
from torch import nn
from torch.utils import data
from torch.autograd import Variable
torch.backends.cudnn.benchmark = True
from gcn import GCN
import numpy as np
import yaml

from dataloader import VisDaDataset
from metrics import scores

from inference import predict, print_scores

# config:
config_path = "/home/flixpar/VisDa/config.yaml"
args = Namespace(**yaml.load(open(config_path, 'r')))
args.img_size = tuple(args.img_size)

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

		img = Variable(image.cuda())
		output = model(img)

		pred = np.squeeze(output.data.max(1)[1].cpu().numpy())
		gt = np.squeeze(Variable(truth).data.cpu().numpy())

		score, class_iou = scores(gt, pred, dataset.num_classes)
		iou += score['Mean IoU : \t']
		# print_scores(i+1, score, class_iou)

	iou /= samples
	print("Mean IOU: {}".format(iou))

if __name__ == "__main__":
	main()
