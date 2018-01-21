import torch
from torch import nn
from torch import optim

import numpy as np

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

	def step(it):
		if not self.enable:
			return

		if it % self.lr_decay_iter or it > self.max_iter:
			return self.optimizer

		lr = self.init_lr*(1 - it/self.max_iter)**self.decay_power
		for param_group in self.optimizer.param_groups:
			param_group['lr'] = lr

def load_args(base_path):
	args = Namespace(**yaml.load(open(os.path.join(base_path, "config.yaml"), 'r')))
	args.img_size = tuple((np.asarray(args.default_img_size) * args.scale_factor).astype(np.int))
	return args

def setup_optimizer(model, args):
	if args.optimizer == "SGD":
		optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
	elif args.optimizer == "Adam":
		optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
	else:
		raise ValueError("Invalid optimizer arg.")
	return optimizer

def load_save(model, args):
	assert os.path.exists(os.path.join(paths["project_path"], "saves"))
	resume_path = save_path.format(args.resume_epoch)
	model.load_state_dict(torch.load(resume_path))
