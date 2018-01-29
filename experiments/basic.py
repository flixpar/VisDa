from experiments.basic_trainer import Trainer

from models.gcn import				GCN
from models.gcn_densenet import 	GCN_DENSENET
from models.gcn_deconv import 		GCN_DECONV
from models.gcn_psp import 			GCN_PSP
from models.gcn_comb import 		GCN_COMBINED
from models.unet import 			UNet

from loaders.visda import VisDaDataset
from loaders.cityscapes_select import CityscapesSelectDataset

from torch.utils import data

from eval import Evaluator
from util.loss import CrossEntropyLoss2d
from util import setup


class Basic(Trainer):

	def __init__(self, config):
		self.args = config
		super(Basic, self).__init__()

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
		dataset = VisDaDataset(im_size=self.args.img_size)
		dataloader = data.DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=8)
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
