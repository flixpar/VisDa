import torch
from torch import nn
from torch.utils import data
from torch.autograd import Variable
torch.backends.cudnn.benchmark = True
import numpy as np

from dataloader import VisDaDataset

save_path = "/home/flixpar/VisDa/saves/gcn-2.pth"
samples = 20
out_path = "/home/flixpar/VisDa/pred/"

dataset = VisDaDataset()
dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)

model = torch.load(save_path).cuda()
model.eval()

def main():
	
	for image, truth in dataloader:

		img = Variable(image.cuda())
		output = model(img)

		pred = np.squeeze(output.data.max(0)[1].cpu().numpy())
		gt = Variable(truth).data.cpu().numpy()

		scores, class_iou = scores(gt, pred, dataset.num_classes)
		print(scores)
		print(class_iou)

if __name__ == "__main__":
	main()
