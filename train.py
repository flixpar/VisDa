import torch
from torch import nn
from torch import optim
from torch import autograd
from torch.utils import data
torch.backends.cudnn.benchmark = True

import os
from tqdm import tqdm
import yaml

import numpy as np
np.seterr(divide='ignore', invalid='ignore')

from models.gcn import GCN
from models.unet import UNet

from loaders.visda import VisDaDataset

from eval import Evaluator

from util.loss import CrossEntropyLoss2d
from util.util import Namespace, poly_lr_scheduler

# config:
args = Namespace(**yaml.load(open(os.path.join(os.getcwd(), "config.yaml"), 'r')))
args.img_size = (int(args.scale_factor*args.default_img_size[0]), int(args.scale_factor*args.default_img_size[1]))

paths = yaml.load(open(os.path.join(os.getcwd(), "paths.yaml"), 'r'))

args.print_dict()
print()

# logging:

save_path = os.path.join(paths["project_path"], "saves", args.model+"-{}.pth")

if not args.resume:
	assert not os.path.exists(os.path.join(paths["project_path"], "saves"))
	os.mkdir(os.path.join(paths["project_path"], "saves"))

logfile = open(os.path.join(paths["project_path"], "saves", "train.log"), 'w')
yaml.dump(args.dict(), open(os.path.join(paths["project_path"], "saves", "config.yaml"), 'w'))

# data loading:
dataset = VisDaDataset(im_size=args.img_size)
dataloader = data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)

# setup evaluator:
evaluator = Evaluator(mode="cityscapes", samples=25, metrics=["miou"], crf=False)

# setup model:
if args.model=="GCN":
	model = GCN(dataset.num_classes, dataset.img_size, k=args.K).cuda()
elif args.model=="UNet":
	model = UNet(dataset.num_classes).cuda()
else:
	raise ValueError("Invalid model arg.")
model.train()

if args.resume:
	assert os.path.exists(os.path.join(paths["project_path"], "saves"))
	resume_path = save_path.format(args.resume_epoch)
	model.load_state_dict(torch.load(resume_path))
	start_epoch = args.resume_epoch
else:
	start_epoch = 0

# setup loss and optimizer

if args.optimizer == "SGD":
	optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
elif args.optimizer == "Adam":
	optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
else:
	raise ValueError("Invalid optimizer arg.")

criterion = CrossEntropyLoss2d(weight=dataset.class_weights).cuda()

if args.resume and args.lr_decay:
	poly_lr_scheduler(optimizer, args.lr, args.resume_epoch, lr_decay_iter=args.lr_decay_freq,
		max_iter=args.max_epochs, power=args.lr_decay_power)

def main():

	print("Starting training...")
	global optimizer
	for epoch in range(start_epoch, args.max_epochs):

		iterations = int(len(dataset)/args.batch_size)
		for i, (image, label) in tqdm(enumerate(dataloader), total=iterations):
			img = autograd.Variable(image.cuda())
			lbl = autograd.Variable(label.cuda())

			output = model(img)
			loss = criterion(output, lbl)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			if i*args.batch_size % 500 == 0:
				tqdm.write("loss: {}".format(loss.data[0]))
				logfile.write(str(loss.data[0])+"\n")

			if i*args.batch_size % args.eval_freq == 0:
				iou = evaluator.eval(model)
				tqdm.write("Eval mIOU: {}".format(iou))
				logfile.write("Eval mIOU: {}\n".format(iou))

		print("Epoch {} completed.".format(epoch + 1))
		logfile.write("Epoch {} completed.\n".format(epoch + 1))
		torch.save(model.state_dict(), save_path.format(epoch + 1))

		iou = evaluator.eval(model)
		tqdm.write("Eval mIOU: {}\n".format(iou))
		logfile.write("Eval mIOU: {}\n\n".format(iou))

		if args.lr_decay:
			poly_lr_scheduler(optimizer, args.lr, epoch, lr_decay_iter=args.lr_decay_freq,
				max_iter=args.max_epochs, power=args.lr_decay_power)

	torch.save(model.state_dict(), save_path.format("final"))


if __name__ == "__main__":
	main()
