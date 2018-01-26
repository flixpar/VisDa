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
from models.gcn_deconv import GCN_DECONV
from models.gcn_psp import GCN_PSP

from loaders.visda import VisDaDataset
from loaders.cityscapes_select import CityscapesSelectDataset

from eval import Evaluator

from util.loss import CrossEntropyLoss2d
from util.util import Namespace
from util.logger import Logger
from util import setup

# load config
args = setup.load_args(os.getcwd())

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
elif args.model=="GCN_DECONV":
	model = GCN_DECONV(dataset.num_classes, dataset.img_size, k=args.K).cuda()
elif args.model=="GCN_PSP":
	model = GCN_PSP(dataset.num_classes, dataset.img_size, k=args.K).cuda()
else:
	raise ValueError("Invalid model arg.")
model.train()

# resume previous training attempt
start_epoch = 0
if args.resume:
	setup.load_save(model, args)
	start_epoch = args.resume_epoch

# setup loss and optimizer
optimizer = setup.init_optimizer(model, args)
scheduler = setup.LRScheduler(optimizer, evaluator, args)
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

		logger.log_epoch(epoch, model, scheduler.get_lr())

	logger.save_final(model)

if __name__ == "__main__":
	main()
