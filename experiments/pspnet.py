import torch
from torch import nn
from torch import optim
from torch import autograd
from torch.utils import data
torch.backends.cudnn.benchmark = True

import numpy as np
np.seterr(divide='ignore', invalid='ignore')
from tqdm import tqdm

from models.pspnet import PSPNet

from loaders.visda import VisDaDataset
from loaders.cityscapes_select import CityscapesSelectDataset

from eval import Evaluator
from util.loss import CrossEntropyLoss2d
from util import setup
from util.logger import Logger


class Trainer:

	def __init__(self):
		self.args = self.get_config()
		self.dataset, self.dataloader = self.get_dataloader()
		self.model = self.get_model(self.dataset)
		self.optimizer = self.get_optimizer(self.model)
		self.evaluator = self.get_evaluator()
		self.scheduler = self.get_scheduler(self.optimizer, self.evaluator)
		self.loss_func = self.get_loss_func()

		# setup logging
		self.logger = Logger(self.args, self.evaluator)
		self.logger.log_args()

	def train(self):

		print("Starting training...")
		for epoch in range(self.args.start_epoch, self.args.max_epochs):
			self.scheduler.step(epoch)

			total_iterations = int(len(self.dataset)/self.args.batch_size)
			for i, (image, label) in tqdm(enumerate(self.dataloader), total=total_iterations):
				img = autograd.Variable(image.cuda())
				lbl = autograd.Variable(label.cuda())

				out, aux = self.model(img)
				loss_a = self.loss_func(out, lbl)
				loss_b = self.loss_func(aux, lbl)
				loss = (1.0 * loss_a) + (0.4 * loss_b)

				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()

				self.logger.log_iter(i, self.model, loss)

			self.logger.log_epoch(epoch, self.model, self.scheduler.get_lr())

		self.logger.save_final(self.model)


class TrainPSPNet(Trainer):

	def __init__(self, config):
		self.args = config
		super(TrainPSPNet, self).__init__()

	def get_model(self, dataset):
		model = PSPNet(dataset.num_classes, dataset.img_size).cuda()

		start_epoch = 0
		if self.args.resume:
			setup.load_save(model, self.args)
			start_epoch = self.args.resume_epoch
		self.args.start_epoch = start_epoch

		model.train()
		return model

	def get_optimizer(self, model):
		optimizer = setup.init_optimizer(model, self.args)
		return optimizer

	def get_scheduler(self, optimizer, evaluator):
		scheduler = setup.LRScheduler(optimizer, evaluator, self.args)
		return scheduler

	def get_dataloader(self):
		dataset = VisDaDataset(im_size=self.args.img_size, samples=None)
		dataloader = data.DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=8, drop_last=True)
		return dataset, dataloader

	def get_evaluator(self):
		evaldataset = CityscapesSelectDataset(im_size=self.args.img_size, n_samples=self.args.eval_samples)
		evaluator = Evaluator(evaldataset, samples=25, metrics=["miou"], crf=False)
		return evaluator

	def get_loss_func(self):
		loss_func = CrossEntropyLoss2d(weight=self.dataset.class_weights).cuda()
		return loss_func

	def get_config(self):
		return self.args
