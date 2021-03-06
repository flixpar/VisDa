import os
import sys
import argparse
from util import setup

from experiments import		basic
from experiments import 	aug
from experiments import 	pspnet
from experiments import		aug_rgb
from experiments import		cityscapes

def main(args):
	config = setup.load_args(os.getcwd())

	experiment = get_experiment(args.experiment, config)
	experiment.train()

def parse_args():
	parser = argparse.ArgumentParser(description="VisDA Project Experiments")
	parser.add_argument("-e", "--experiment", help="name of experiment to run", required=True)
	parser.add_argument("-c", "--config", help="path to config file")
	args = parser.parse_args()
	return args

def get_experiment(name, config):
	if name == "basic":
		experiment = basic.Basic(config)
	elif name == "aug":
		experiment = aug.Trainer(config)
	elif name == "pspnet":
		experiment = pspnet.TrainPSPNet(config)
	elif name == "aug_rgb":
		experiment = aug_rgb.Trainer(config)
	elif name == "cityscapes":
		experiment = cityscapes.Basic(config)
	else:
		raise ValueError("Invalid experiment name.")

	return experiment

if __name__ == "__main__":
	args = parse_args()
	main(args)
