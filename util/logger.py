import os
import yaml
from tqdm import tqdm

import torch

from util.util import Namespace


class Logger:

	def __init__(self, args, evaluator):

		# setup
		self.args = args
		self.evaluator = evaluator
		self.iterations = 0
		self.avgloss = RunningAvg()

		# create base save paths
		self.save_folder = os.path.join(args.paths["project_path"], "saves")
		self.save_path = os.path.join(self.save_folder, args.model+"-{}.pth")

		# create new save folder
		if not args.resume:
			assert not os.path.exists(self.save_folder)
			os.mkdir(self.save_folder)

		# open file for logging to
		self.logfile = open(os.path.join(self.save_folder, "train.log"), 'w')

		# dump config to save folder
		yaml.dump(args.dict(), open(os.path.join(self.save_folder, "config.yaml"), 'w'))

		# setup logging for best weights
		self.max_eval = 0
		self.max_eval_it = 0
		self.path_to_best = os.path.join(self.save_folder, "best-{}.pth")

	def log_args(self):
		self.args.print_dict()
		print()

	def log_epoch(self, epoch, model, lr):

		print("Epoch {} completed.".format(epoch + 1))
		self.logfile.write("Epoch {} completed.\n".format(epoch + 1))

		print("Epoch LR: {}".format(lr))
		self.logfile.write("Epoch {} LR: {}\n".format(epoch+1, lr))

		torch.save(model.state_dict(), self.save_path.format(epoch + 1))

		iou = self.evaluator.eval(model)
		tqdm.write("Eval mIOU: {}\n".format(iou))
		self.logfile.write("it: {}, miou: {}\n".format(self.iterations, iou))

	def log_iter(self, it, model, loss):

		self.avgloss.update(loss.data[0])

		self.iterations += self.args.batch_size
		img_num = (it+1) * self.args.batch_size

		if img_num % 100 < self.args.batch_size:
			self.logfile.write("it: {}, loss: {}\n".format(self.iterations, self.avgloss.get()))

		if img_num % 1000 < self.args.batch_size:
			tqdm.write("loss: {}".format(self.avgloss.get()))
			self.logfile.flush()
			self.avgloss.reset()

		if img_num % self.args.eval_freq < self.args.batch_size:
			iou = self.evaluator.eval(model)
			tqdm.write("Eval mIOU: {}".format(iou))
			self.logfile.write("it: {}, miou: {}\n".format(self.iterations, iou))
			if iou > self.max_eval:
				os.remove(self.path_to_best.format(self.max_eval_it))
				torch.save(model.state_dict(), self.path_to_best.format(it))
				self.max_eval = iou
				self.max_eval_it = it

	def save_final(self, model):
		torch.save(model.state_dict(), self.save_path.format("final"))

class RunningAvg:

	def __init__(self):
		self.sum = 0
		self.elements = 0

	def update(self, val):
		self.sum += val
		self.elements += 1

	def get(self):
		return (self.sum / self.elements) if self.elements != 0 else 0

	def reset(self):
		self.sum = 0
		self.elements = 0
