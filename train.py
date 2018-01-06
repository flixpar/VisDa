import torch
from torch import nn
from torch import optim
from torch import autograd
from torch.utils import data
torch.backends.cudnn.benchmark = True

from models.gcn import GCN
from loaders.dataloader import VisDaDataset
from util.loss import CrossEntropyLoss2d
from util.util import Namespace, poly_lr_scheduler

import os
from tqdm import tqdm
import yaml

# config:
config_path = "/home/flixpar/VisDa/config.yaml"
args = Namespace(**yaml.load(open(config_path, 'r')))

# logging:
save_path = os.path.join(args.base_path, "saves", "gcn-{}.pth")
logfile = open(os.path.join(args.base_path, "saves", "train.log"), 'w')

# data loading:
dataset = VisDaDataset(im_size=args.img_size)
dataloader = data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=6)
evalloader = data.DataLoader(VisDaDataset(im_size=args.img_size, mode="eval"), batch_size=1, shuffle=True)

# setup model:
model = GCN(dataset.num_classes, dataset.img_size, k=args.K).cuda()
model.train()

# setup loss and optimizer
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
criterion = CrossEntropyLoss2d(weight=dataset.class_weights).cuda()

def main():

	print("Starting training...")
	for epoch in range(args.max_epochs):

		for i, (image, label) in tqdm(enumerate(dataloader), total=int(len(dataset)/args.batch_size)):
			img = autograd.Variable(image.cuda())
			lbl = autograd.Variable(label.cuda())

			output = model(img)
			loss = criterion(output, lbl)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			if i % 100 == 0:
				tqdm.write("loss: {}".format(loss.data[0]))
				logfile.write(str(loss.data[0])+"\n")

			if i % args.eval_freq == 0:
				miou = evaluate()
				tqdm.write("Eval mIOU: {}".format(miou))
				logfile.write("Eval mIOU: {}\n".format(miou))

		print("Epoch {} completed.".format(epoch + 1))
		logfile.write("Epoch {} completed.\n".format(epoch + 1))
		torch.save(model.state_dict(), save_path.format(epoch + 1))

		miou = evaluate()
		tqdm.write("Eval mIOU: {}".format(miou))
		logfile.write("Eval mIOU: {}\n".format(miou))

		optimizer = poly_lr_scheduler(optimizer, args.lr, epoch,
			lr_decay_iter=args.lr_decay_freq, max_iter=args.max_epochs, power=args.lr_power)

	torch.save(model.state_dict(), save_path.format("final"))

def evaluate():
	model.eval()

	iou = 0
	samples = 0

	for (img, lbl) in evalloader:
		img = Variable(img.cuda())
		output = model(img)

		pred = np.squeeze(output.data.max(1)[1].cpu().numpy())
		gt = np.squeeze(lbl.cpu().numpy())

		iou += miou(gt, pred, dataset.num_classes)
		samples += 1

	model.train()
	miou /= samples
	return miou

if __name__ == "__init__":
	main()
