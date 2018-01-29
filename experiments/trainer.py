import torch
from torch import nn
from torch import optim
from torch import autograd
from torch.utils import data
torch.backends.cudnn.benchmark = True

import numpy as np
np.seterr(divide='ignore', invalid='ignore')

from tqdm import tqdm

from util.logger import Logger

class Trainer:

	def __init__(self):
		self.dataloader = self.get_dataloader()
		self.model = self.get_model()
		self.optimizer = self.get_optimizer(model)
		self.evaluator = self.get_evaluator()
		self.scheduler = self.get_scheduler(optimizer, evaluator)
		self.loss_func = self.get_loss_func()

		# setup logging
		logger = Logger(args, evaluator)
		logger.log_args()

	def train(self):

		print("Starting training...")
		for epoch in range(start_epoch, args.max_epochs):
			self.scheduler.step(epoch)

			total_iterations = int(len(dataset)/args.batch_size)
			for i, (image, label) in tqdm(enumerate(dataloader), total=total_iterations):
				img = autograd.Variable(image.cuda())
				lbl = autograd.Variable(label.cuda())

				output = self.model(img)
				loss = loss_func(output, lbl)

				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()

				self.logger.log_iter(i, model, loss)

			self.logger.log_epoch(epoch, model, scheduler.get_lr())

		self.logger.save_final(model)
