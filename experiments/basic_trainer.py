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

				output = self.model(img)
				loss = self.loss_func(output, lbl)

				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()

				self.logger.log_iter(i, self.model, loss)

			self.logger.log_epoch(epoch, self.model, self.scheduler.get_lr())

		self.logger.save_final(self.model)
