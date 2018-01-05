import torch
from torch import nn
from torch.utils import data
from torch.autograd import Variable
torch.backends.cudnn.benchmark = True
from gcn import GCN
import numpy as np

from dataloader import VisDaDataset
from metrics import scores

save_path = "/home/flixpar/VisDa/saves/gcn-8.pth"
samples = 50
out_path = "/home/flixpar/VisDa/pred/"

K = 7
scale_factor = 0.8
img_size = (int(scale_factor * 1052), int(scale_factor * 1914))

dataset = VisDaDataset(im_size=img_size)
dataloader = data.DataLoader(dataset, batch_size=1, shuffle=True)

model = GCN(dataset.num_classes, dataset.img_size, k=K).cuda()
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


def print_scores(i, score, class_iou):
	print("### Image {} ###".format(i))
	for key, val in score.items():
		print("{}{}".format(key, val))
	for key, val in class_iou.items():
		if not np.isnan(val):
			print("{}:\t{}".format(key, val))
	print()


if __name__ == "__main__":
	main()
