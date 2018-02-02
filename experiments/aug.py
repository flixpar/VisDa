import torch
from torch import nn
from torch import optim
from torch import autograd
from torch.utils import data
torch.backends.cudnn.benchmark = True

from models.gcn import				GCN
from models.gcn_densenet import 	GCN_DENSENET
from models.gcn_deconv import 		GCN_DECONV
from models.gcn_psp import 			GCN_PSP
from models.gcn_comb import 		GCN_COMBINED
from models.unet import 			UNet

from loaders.visda_aug import VisDaAugDataset
from loaders.cityscapes_select import CityscapesSelectDataset

from tqdm import tqdm
import numpy as np
np.seterr(divide='ignore', invalid='ignore')

from eval import Evaluator
from util.loss import CrossEntropyLoss2d
from util import setup

from util.logger import Logger


class Trainer():

	def __init__(self, config):

		# setup
		self.args = config
		self.dataset, self.dataloader = self.get_dataloader()
		self.model = self.get_model(self.dataset)
		self.optimizer = self.get_optimizer(self.model)
		self.evaluator = self.get_evaluator()
		self.scheduler = self.get_scheduler(self.optimizer, self.evaluator)
		self.loss_func = self.get_loss_func()

		# setup logging
		self.logger = Logger(self.args, self.evaluator)
		self.logger.log_args()

	def get_model(self, dataset):
		if self.args.model=="GCN":
			model = GCN(dataset.num_classes, dataset.img_size, k=self.args.K).cuda()
		elif self.args.model=="UNet":
			model = UNet(dataset.num_classes).cuda()
		elif self.args.model=="GCN_DENSENET":
			model = GCN_DENSENET(dataset.num_classes, dataset.img_size, k=self.args.K).cuda()
		elif self.args.model=="GCN_DECONV":
			model = GCN_DECONV(dataset.num_classes, dataset.img_size, k=self.args.K).cuda()
		elif self.args.model=="GCN_PSP":
			model = GCN_PSP(dataset.num_classes, dataset.img_size, k=self.args.K).cuda()
		elif self.args.model=="GCN_COMB":
			model = GCN_COMBINED(dataset.num_classes, dataset.img_size, k=self.args.K).cuda()
		else:
			raise ValueError("Invalid model arg.")

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
		dataset = VisDaAugDataset(im_size=self.args.img_size, samples=None)
		dataloader = data.DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=8, drop_last=True)
		return dataset, dataloader

	def get_evaluator(self):
		evaldataset = CityscapesSelectDataset(im_size=self.args.img_size, n_samples=self.args.eval_samples)
		evaluator = Evaluator(evaldataset, samples=25, metrics=["miou"], crf=False)
		return evaluator

	def get_loss_func(self):
		loss_func = CrossEntropyLoss2d(weight=self.dataset.class_weights).cuda()
		return loss_func

	def train(self):

		print("Starting training...")
		for epoch in range(self.args.start_epoch, self.args.max_epochs):
			self.scheduler.step(epoch)

			total_iterations = int(len(self.dataset)/self.args.batch_size)
			for i, (image, label) in tqdm(enumerate(self.dataloader), total=total_iterations):
				bs, ncrops, c, h, w = image.size()
				image = image.view(-1, c, h, w)
				label = label.view(-1, h, w)

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
