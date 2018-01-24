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
from models.gcn_densenet import GCN_DENSENET

from loaders.visda import VisDaDataset
from loaders.cityscapes_select import CityscapesSelectDataset

from eval import Evaluator

from util.loss import CrossEntropyLoss2d
from util.util import Namespace
from util.setup import *
from util.logger import Logger

# load config
args = load_args(os.getcwd())

# data loading
dataset = VisDaDataset(im_size=args.img_size)
dataloader = data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)

# setup evaluator
evaldataset = CityscapesSelectDataset(im_size=args.img_size, n_samples=args.eval_samples)
evaluator = Evaluator(evaldataset, samples=25, metrics=["miou"], crf=False)

# setup model
if args.model=="GCN":
	model = GCN(dataset.num_classes, dataset.img_size, k=args.K).cuda()
elif args.model=="UNet":
	model = UNet(dataset.num_classes).cuda()
elif args.model=="GCN_DENSENET":
	model = GCN_DENSENET(dataset.num_classes, dataset.img_size, k=args.K).cuda()
else: raise ValueError("Invalid model arg.")
model.train()

# resume previous training attempt
start_epoch = 0
if args.resume:
	load_save(model, args)
	start_epoch = args.resume_epoch

# setup loss and optimizer
optimizer = setup_optimizer(model, args)
scheduler = PolyLRScheduler(optimizer, args.lr, enable=args.lr_decay, lr_decay_iter=args.lr_decay_freq, max_iter=args.max_epochs, power=args.lr_decay_power)
loss_func = CrossEntropyLoss2d(weight=dataset.class_weights).cuda()

# setup logging
logger = Logger(args, evaluator)

# display config
logger.log_args()

def main():

	print("Starting training...")
	for epoch in range(start_epoch, args.max_epochs):
		scheduler.step(epoch)

		total_iterations = int(len(dataset)/args.batch_size)
		for i, (image, label) in tqdm(enumerate(dataloader), total=total_iterations):
			img = autograd.Variable(image.cuda())
			lbl = autograd.Variable(label.cuda())

			output = model(img)
			loss = loss_func(output, lbl)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			logger.log_iter(i, model, loss)

		logger.log_epoch(epoch, model)

	logger.save_final(model)

if __name__ == "__main__":
	main()
