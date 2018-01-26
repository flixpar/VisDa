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

		self.iterations += self.args.batch_size
		img_num = (it+1) * self.args.batch_size

		if img_num % 1000 < self.args.batch_size:
			tqdm.write("loss: {}".format(loss.data[0]))
			
		if img_num % 100 < self.args.batch_size:
			self.logfile.write("it: {}, loss: {}\n".format(self.iterations, loss.data[0]))

		if img_num % self.args.eval_freq < self.args.batch_size:
			iou = self.evaluator.eval(model)
			tqdm.write("Eval mIOU: {}".format(iou))
			self.logfile.write("it: {}, miou: {}\n".format(self.iterations, iou))

	def save_final(self, model):
		torch.save(model.state_dict(), self.save_path.format("final"))
