import torch
from torch import nn
from torch import optim

import numpy as np
import yaml
import os

from util.util import Namespace

def poly_lr_scheduler(optimizer, init_lr, it, lr_decay_iter=1, max_iter=100, power=0.9):
	if it % lr_decay_iter or it > max_iter:
		return optimizer

	lr = init_lr*(1 - it/max_iter)**power
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr

class PolyLRScheduler:
	def __init__(self, optimizer, init_lr, enable=True, lr_decay_iter=1, max_iter=100, power=0.9):
		self.optimizer = optimizer
		self.init_lr = init_lr
		self.lr_decay_iter = lr_decay_iter
		self.max_iter = max_iter
		self.decay_power = power
		self.enable = enable
		self.lr = init_lr

	def step(self, it):
		if not self.enable:
			return

		if it % self.lr_decay_iter or it > self.max_iter:
			return self.optimizer

		self.lr = self.init_lr*(1 - it/self.max_iter)**self.decay_power
		for param_group in self.optimizer.param_groups:
			param_group['lr'] = self.lr

	def get_lr(self):
		return self.lr

def load_args(base_path, eval_path=None):

	config_file = open(os.path.join(base_path, "config.yaml"), 'r')

	if eval_path is not None:
		eval_file = open(os.path.join(eval_path, "eval.yaml"), 'r')
	else:
		eval_file = open(os.path.join(base_path, "eval.yaml"), 'r')

	args = Namespace(**yaml.load(config_file))
	args.eval = yaml.load(eval_file)

	# args.img_size = (int(args.scale_factor*args.default_img_size[0]), int(args.scale_factor*args.default_img_size[1]))

	height = args.scale_factor * args.default_img_size[0]
	width  = args.scale_factor * args.default_img_size[1]
	
	height = round(height / 16) * 16
	width  = round(width  / 16) * 16
	
	args.img_size = (int(height), int(width))

	return args

def init_optimizer(model, args):
	if args.optimizer == "SGD":
		optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
	elif args.optimizer == "Adam":
		optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
	else:
		raise ValueError("Invalid optimizer arg.")
	return optimizer

class LRScheduler:
	def __init__(self, optimizer, evaluator, args):
		self.optimizer = optimizer
		self.evaluator = evaluator
		self.args = args

		if args.scheduler == "poly":
			self.scheduler = PolyLRScheduler(optimizer, args.lr, enable=args.lr_decay, lr_decay_iter=args.lr_decay_freq, max_iter=args.max_epochs, power=args.lr_decay_power)
		elif args.scheduler == "plateau":
			self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.2, patience=1, verbose=True)
		elif args.scheduler == "step":
			self.scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
		elif args.scheduler == "exponential":
			self.scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.2)
		else:
			raise ValueError("Invalid scheduler option.")

	def step(self, it):
		if self.args.scheduler == "poly":
			self.scheduler.step(it)
		if self.args.scheduler == "plateau":
			acc = self.evaluator.eval()
			self.scheduler.step(acc)
		if self.args.scheduler == "step":
			self.scheduler.step()
		if self.args.scheduler == "exponential":
			self.scheduler.step()

def load_save(model, args):
	assert os.path.exists(os.path.join(args.paths["project_path"], "saves"))
	resume_path = os.path.join(args.paths["project_path"], "saves", "{}-{}.pth".format(args.model, args.resume_epoch))
	model.load_state_dict(torch.load(resume_path))
