
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import reduce
from torch.autograd import Variable

import os
from math import floor
import yaml

from util.setup import load_args
args = load_args(os.getcwd())
pretrained_dir = args.paths["pretrained_models_path"]

################## GCN Modules #####################

class _GlobalConvModule(nn.Module):
	def __init__(self, in_dim, out_dim, kernel_size):
		super(_GlobalConvModule, self).__init__()

		pad0 = floor((kernel_size[0] - 1) / 2)
		pad1 = floor((kernel_size[1] - 1) / 2)

		self.conv_l1 = nn.Conv2d(in_dim, out_dim, kernel_size=(kernel_size[0], 1), padding=(pad0, 0))
		self.conv_l2 = nn.Conv2d(out_dim, out_dim, kernel_size=(1, kernel_size[1]), padding=(0, pad1))
		self.conv_r1 = nn.Conv2d(in_dim, out_dim, kernel_size=(1, kernel_size[1]), padding=(0, pad1))
		self.conv_r2 = nn.Conv2d(out_dim, out_dim, kernel_size=(kernel_size[0], 1), padding=(pad0, 0))

	def forward(self, x):
		x_l = self.conv_l1(x)
		x_l = self.conv_l2(x_l)
		x_r = self.conv_r1(x)
		x_r = self.conv_r2(x_r)
		x = x_l + x_r
		return x

class _BoundaryRefineModule(nn.Module):
	def __init__(self, dim):
		super(_BoundaryRefineModule, self).__init__()
		self.relu = nn.ReLU(inplace=True)
		self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
		self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)

	def forward(self, x):
		residual = self.conv1(x)
		residual = self.relu(residual)
		residual = self.conv2(residual)
		out = x + residual
		return out

class _DeconvModule(nn.Module):
	def __init__(self, channels):
		super(_DeconvModule, self).__init__()
		self.deconv = nn.ConvTranspose2d(channels, channels, kernel_size=4, stride=2, padding=1)
		self.deconv.weight.data = self.make_bilinear_weights(4, channels)
		self.deconv.bias.data.zero_()

	def forward(self, x):
		out = self.deconv(x)
		return out

	def make_bilinear_weights(self, size, num_channels):
		factor = (size + 1) // 2
		if size % 2 == 1:
			center = factor - 1
		else:
			center = factor - 0.5
		og = np.ogrid[:size, :size]
		filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
		filt = torch.from_numpy(filt)
		w = torch.zeros(num_channels, num_channels, size, size)
		for i in range(num_channels):
			w[i, i] = filt
		return w

############################## GCN #################################

class GCN_RESNEXT(nn.Module):

	def __init__(self, num_classes, k=7):

		self.num_classes = num_classes
		self.K = k

		self.resnext = ResNeXt()
		self.resnext.load_state_dict(torch.load(os.path.join(pretrained_dir,'resnext.pth')))

		self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
		self.layer1 = nn.Sequential(resnet.maxpool, resnet.layer1)
		self.layer2 = resnet.layer2
		self.layer3 = resnet.layer3
		self.layer4 = resnet.layer4

		self.gcm1 = _GlobalConvModule(2048, num_classes, (self.K, self.K))
		self.gcm2 = _GlobalConvModule(1024, num_classes, (self.K, self.K))
		self.gcm3 = _GlobalConvModule(512, num_classes, (self.K, self.K))
		self.gcm4 = _GlobalConvModule(256, num_classes, (self.K, self.K))

		self.brm1 = _BoundaryRefineModule(num_classes)
		self.brm2 = _BoundaryRefineModule(num_classes)
		self.brm3 = _BoundaryRefineModule(num_classes)
		self.brm4 = _BoundaryRefineModule(num_classes)
		self.brm5 = _BoundaryRefineModule(num_classes)
		self.brm6 = _BoundaryRefineModule(num_classes)
		self.brm7 = _BoundaryRefineModule(num_classes)
		self.brm8 = _BoundaryRefineModule(num_classes)
		self.brm9 = _BoundaryRefineModule(num_classes)

		self.deconv = _DeconvModule(num_classes)

		self.initialize_weights(self.gcm1, self.gcm2, self.gcm3, self.gcm4, self.brm1, self.brm2, self.brm3, self.brm4, self.brm5, self.brm6, self.brm7, self.brm8, self.brm9)

	def forward(self, x):

		fm0 = self.resnext.layer0(x)
		fm1 = self.resnext.layer1(fm0)
		fm2 = self.resnext.layer2(fm1)
		fm3 = self.resnext.layer3(fm2)
		fm4 = self.resnext.layer4(fm3)

		gcfm1 = self.brm1(self.gcm1(fm4))
		gcfm2 = self.brm2(self.gcm2(fm3))
		gcfm3 = self.brm3(self.gcm3(fm2))
		gcfm4 = self.brm4(self.gcm4(fm1))

		fs1 = self.brm5(self.deconv(gcfm1) + gcfm2)
		fs2 = self.brm6(self.deconv(fs1) + gcfm3)
		fs3 = self.brm7(self.deconv(fs2) + gcfm4)
		fs4 = self.brm8(self.deconv(fs3))
		out = self.brm9(self.deconv(fs4))

		return out

	def initialize_weights(self, *models):
		for model in models:
			for module in model.modules():
				if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
					nn.init.kaiming_normal(module.weight)
					if module.bias is not None:
						module.bias.data.zero_()
				elif isinstance(module, nn.BatchNorm2d):
					module.weight.data.fill_(1)
					module.bias.data.zero_()


###################### ResNeXt Modules #########################

class LambdaBase(nn.Sequential):
	def __init__(self, fn, *args):
		super(LambdaBase, self).__init__(*args)
		self.lambda_func = fn

	def forward_prepare(self, input):
		output = []
		for module in self._modules.values():
			output.append(module(input))
		return output if output else input

class Lambda(LambdaBase):
	def forward(self, input):
		return self.lambda_func(self.forward_prepare(input))

class LambdaMap(LambdaBase):
	def forward(self, input):
		return list(map(self.lambda_func,self.forward_prepare(input)))

class LambdaReduce(LambdaBase):
	def forward(self, input):
		return reduce(self.lambda_func,self.forward_prepare(input))

########################### ResNeXt ###########################

class ResNeXt(nn.Module):

	def __init__(self):

		self.layer0 = nn.Sequential(
			self.resnext[0],
			self.resnext[1],
			self.resnext[2],
			self.resnext[3]
		)

		self.layer1 = self.rexnext[4]
		self.layer2 = self.resnext[5]
		self.layer3 = self.resnext[6]
		self.layer4 = self.resnext[7]

		self.layer5 = nn.Sequential(
			self.resnext[8],
			self.resnext[9],
			self.resnext[10],
		)

	def forward(self, x):

		x = self.layer0(x)
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		x = self.layer5(x)

		return x

	self.resnext = nn.Sequential( # Sequential,
		nn.Conv2d(3,64,(7, 7),(2, 2),(3, 3),1,1,bias=False),
		nn.BatchNorm2d(64),
		nn.ReLU(),
		nn.MaxPool2d((3, 3),(2, 2),(1, 1)),
		nn.Sequential( # Sequential,
			nn.Sequential( # Sequential,
				LambdaMap(lambda x: x, # ConcatTable,
					nn.Sequential( # Sequential,
						nn.Sequential( # Sequential,
							nn.Conv2d(64,256,(1, 1),(1, 1),(0, 0),1,1,bias=False),
							nn.BatchNorm2d(256),
							nn.ReLU(),
							nn.Conv2d(256,256,(3, 3),(1, 1),(1, 1),1,64,bias=False),
							nn.BatchNorm2d(256),
							nn.ReLU(),
						),
						nn.Conv2d(256,256,(1, 1),(1, 1),(0, 0),1,1,bias=False),
						nn.BatchNorm2d(256),
					),
					nn.Sequential( # Sequential,
						nn.Conv2d(64,256,(1, 1),(1, 1),(0, 0),1,1,bias=False),
						nn.BatchNorm2d(256),
					),
				),
				LambdaReduce(lambda x,y: x+y), # CAddTable,
				nn.ReLU(),
			),
			nn.Sequential( # Sequential,
				LambdaMap(lambda x: x, # ConcatTable,
					nn.Sequential( # Sequential,
						nn.Sequential( # Sequential,
							nn.Conv2d(256,256,(1, 1),(1, 1),(0, 0),1,1,bias=False),
							nn.BatchNorm2d(256),
							nn.ReLU(),
							nn.Conv2d(256,256,(3, 3),(1, 1),(1, 1),1,64,bias=False),
							nn.BatchNorm2d(256),
							nn.ReLU(),
						),
						nn.Conv2d(256,256,(1, 1),(1, 1),(0, 0),1,1,bias=False),
						nn.BatchNorm2d(256),
					),
					Lambda(lambda x: x), # Identity,
				),
				LambdaReduce(lambda x,y: x+y), # CAddTable,
				nn.ReLU(),
			),
			nn.Sequential( # Sequential,
				LambdaMap(lambda x: x, # ConcatTable,
					nn.Sequential( # Sequential,
						nn.Sequential( # Sequential,
							nn.Conv2d(256,256,(1, 1),(1, 1),(0, 0),1,1,bias=False),
							nn.BatchNorm2d(256),
							nn.ReLU(),
							nn.Conv2d(256,256,(3, 3),(1, 1),(1, 1),1,64,bias=False),
							nn.BatchNorm2d(256),
							nn.ReLU(),
						),
						nn.Conv2d(256,256,(1, 1),(1, 1),(0, 0),1,1,bias=False),
						nn.BatchNorm2d(256),
					),
					Lambda(lambda x: x), # Identity,
				),
				LambdaReduce(lambda x,y: x+y), # CAddTable,
				nn.ReLU(),
			),
		),
		nn.Sequential( # Sequential,
			nn.Sequential( # Sequential,
				LambdaMap(lambda x: x, # ConcatTable,
					nn.Sequential( # Sequential,
						nn.Sequential( # Sequential,
							nn.Conv2d(256,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
							nn.BatchNorm2d(512),
							nn.ReLU(),
							nn.Conv2d(512,512,(3, 3),(2, 2),(1, 1),1,64,bias=False),
							nn.BatchNorm2d(512),
							nn.ReLU(),
						),
						nn.Conv2d(512,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
						nn.BatchNorm2d(512),
					),
					nn.Sequential( # Sequential,
						nn.Conv2d(256,512,(1, 1),(2, 2),(0, 0),1,1,bias=False),
						nn.BatchNorm2d(512),
					),
				),
				LambdaReduce(lambda x,y: x+y), # CAddTable,
				nn.ReLU(),
			),
			nn.Sequential( # Sequential,
				LambdaMap(lambda x: x, # ConcatTable,
					nn.Sequential( # Sequential,
						nn.Sequential( # Sequential,
							nn.Conv2d(512,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
							nn.BatchNorm2d(512),
							nn.ReLU(),
							nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1),1,64,bias=False),
							nn.BatchNorm2d(512),
							nn.ReLU(),
						),
						nn.Conv2d(512,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
						nn.BatchNorm2d(512),
					),
					Lambda(lambda x: x), # Identity,
				),
				LambdaReduce(lambda x,y: x+y), # CAddTable,
				nn.ReLU(),
			),
			nn.Sequential( # Sequential,
				LambdaMap(lambda x: x, # ConcatTable,
					nn.Sequential( # Sequential,
						nn.Sequential( # Sequential,
							nn.Conv2d(512,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
							nn.BatchNorm2d(512),
							nn.ReLU(),
							nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1),1,64,bias=False),
							nn.BatchNorm2d(512),
							nn.ReLU(),
						),
						nn.Conv2d(512,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
						nn.BatchNorm2d(512),
					),
					Lambda(lambda x: x), # Identity,
				),
				LambdaReduce(lambda x,y: x+y), # CAddTable,
				nn.ReLU(),
			),
			nn.Sequential( # Sequential,
				LambdaMap(lambda x: x, # ConcatTable,
					nn.Sequential( # Sequential,
						nn.Sequential( # Sequential,
							nn.Conv2d(512,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
							nn.BatchNorm2d(512),
							nn.ReLU(),
							nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1),1,64,bias=False),
							nn.BatchNorm2d(512),
							nn.ReLU(),
						),
						nn.Conv2d(512,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
						nn.BatchNorm2d(512),
					),
					Lambda(lambda x: x), # Identity,
				),
				LambdaReduce(lambda x,y: x+y), # CAddTable,
				nn.ReLU(),
			),
		),
		nn.Sequential( # Sequential,
			nn.Sequential( # Sequential,
				LambdaMap(lambda x: x, # ConcatTable,
					nn.Sequential( # Sequential,
						nn.Sequential( # Sequential,
							nn.Conv2d(512,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
							nn.BatchNorm2d(1024),
							nn.ReLU(),
							nn.Conv2d(1024,1024,(3, 3),(2, 2),(1, 1),1,64,bias=False),
							nn.BatchNorm2d(1024),
							nn.ReLU(),
						),
						nn.Conv2d(1024,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
						nn.BatchNorm2d(1024),
					),
					nn.Sequential( # Sequential,
						nn.Conv2d(512,1024,(1, 1),(2, 2),(0, 0),1,1,bias=False),
						nn.BatchNorm2d(1024),
					),
				),
				LambdaReduce(lambda x,y: x+y), # CAddTable,
				nn.ReLU(),
			),
			nn.Sequential( # Sequential,
				LambdaMap(lambda x: x, # ConcatTable,
					nn.Sequential( # Sequential,
						nn.Sequential( # Sequential,
							nn.Conv2d(1024,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
							nn.BatchNorm2d(1024),
							nn.ReLU(),
							nn.Conv2d(1024,1024,(3, 3),(1, 1),(1, 1),1,64,bias=False),
							nn.BatchNorm2d(1024),
							nn.ReLU(),
						),
						nn.Conv2d(1024,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
						nn.BatchNorm2d(1024),
					),
					Lambda(lambda x: x), # Identity,
				),
				LambdaReduce(lambda x,y: x+y), # CAddTable,
				nn.ReLU(),
			),
			nn.Sequential( # Sequential,
				LambdaMap(lambda x: x, # ConcatTable,
					nn.Sequential( # Sequential,
						nn.Sequential( # Sequential,
							nn.Conv2d(1024,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
							nn.BatchNorm2d(1024),
							nn.ReLU(),
							nn.Conv2d(1024,1024,(3, 3),(1, 1),(1, 1),1,64,bias=False),
							nn.BatchNorm2d(1024),
							nn.ReLU(),
						),
						nn.Conv2d(1024,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
						nn.BatchNorm2d(1024),
					),
					Lambda(lambda x: x), # Identity,
				),
				LambdaReduce(lambda x,y: x+y), # CAddTable,
				nn.ReLU(),
			),
			nn.Sequential( # Sequential,
				LambdaMap(lambda x: x, # ConcatTable,
					nn.Sequential( # Sequential,
						nn.Sequential( # Sequential,
							nn.Conv2d(1024,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
							nn.BatchNorm2d(1024),
							nn.ReLU(),
							nn.Conv2d(1024,1024,(3, 3),(1, 1),(1, 1),1,64,bias=False),
							nn.BatchNorm2d(1024),
							nn.ReLU(),
						),
						nn.Conv2d(1024,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
						nn.BatchNorm2d(1024),
					),
					Lambda(lambda x: x), # Identity,
				),
				LambdaReduce(lambda x,y: x+y), # CAddTable,
				nn.ReLU(),
			),
			nn.Sequential( # Sequential,
				LambdaMap(lambda x: x, # ConcatTable,
					nn.Sequential( # Sequential,
						nn.Sequential( # Sequential,
							nn.Conv2d(1024,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
							nn.BatchNorm2d(1024),
							nn.ReLU(),
							nn.Conv2d(1024,1024,(3, 3),(1, 1),(1, 1),1,64,bias=False),
							nn.BatchNorm2d(1024),
							nn.ReLU(),
						),
						nn.Conv2d(1024,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
						nn.BatchNorm2d(1024),
					),
					Lambda(lambda x: x), # Identity,
				),
				LambdaReduce(lambda x,y: x+y), # CAddTable,
				nn.ReLU(),
			),
			nn.Sequential( # Sequential,
				LambdaMap(lambda x: x, # ConcatTable,
					nn.Sequential( # Sequential,
						nn.Sequential( # Sequential,
							nn.Conv2d(1024,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
							nn.BatchNorm2d(1024),
							nn.ReLU(),
							nn.Conv2d(1024,1024,(3, 3),(1, 1),(1, 1),1,64,bias=False),
							nn.BatchNorm2d(1024),
							nn.ReLU(),
						),
						nn.Conv2d(1024,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
						nn.BatchNorm2d(1024),
					),
					Lambda(lambda x: x), # Identity,
				),
				LambdaReduce(lambda x,y: x+y), # CAddTable,
				nn.ReLU(),
			),
			nn.Sequential( # Sequential,
				LambdaMap(lambda x: x, # ConcatTable,
					nn.Sequential( # Sequential,
						nn.Sequential( # Sequential,
							nn.Conv2d(1024,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
							nn.BatchNorm2d(1024),
							nn.ReLU(),
							nn.Conv2d(1024,1024,(3, 3),(1, 1),(1, 1),1,64,bias=False),
							nn.BatchNorm2d(1024),
							nn.ReLU(),
						),
						nn.Conv2d(1024,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
						nn.BatchNorm2d(1024),
					),
					Lambda(lambda x: x), # Identity,
				),
				LambdaReduce(lambda x,y: x+y), # CAddTable,
				nn.ReLU(),
			),
			nn.Sequential( # Sequential,
				LambdaMap(lambda x: x, # ConcatTable,
					nn.Sequential( # Sequential,
						nn.Sequential( # Sequential,
							nn.Conv2d(1024,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
							nn.BatchNorm2d(1024),
							nn.ReLU(),
							nn.Conv2d(1024,1024,(3, 3),(1, 1),(1, 1),1,64,bias=False),
							nn.BatchNorm2d(1024),
							nn.ReLU(),
						),
						nn.Conv2d(1024,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
						nn.BatchNorm2d(1024),
					),
					Lambda(lambda x: x), # Identity,
				),
				LambdaReduce(lambda x,y: x+y), # CAddTable,
				nn.ReLU(),
			),
			nn.Sequential( # Sequential,
				LambdaMap(lambda x: x, # ConcatTable,
					nn.Sequential( # Sequential,
						nn.Sequential( # Sequential,
							nn.Conv2d(1024,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
							nn.BatchNorm2d(1024),
							nn.ReLU(),
							nn.Conv2d(1024,1024,(3, 3),(1, 1),(1, 1),1,64,bias=False),
							nn.BatchNorm2d(1024),
							nn.ReLU(),
						),
						nn.Conv2d(1024,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
						nn.BatchNorm2d(1024),
					),
					Lambda(lambda x: x), # Identity,
				),
				LambdaReduce(lambda x,y: x+y), # CAddTable,
				nn.ReLU(),
			),
			nn.Sequential( # Sequential,
				LambdaMap(lambda x: x, # ConcatTable,
					nn.Sequential( # Sequential,
						nn.Sequential( # Sequential,
							nn.Conv2d(1024,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
							nn.BatchNorm2d(1024),
							nn.ReLU(),
							nn.Conv2d(1024,1024,(3, 3),(1, 1),(1, 1),1,64,bias=False),
							nn.BatchNorm2d(1024),
							nn.ReLU(),
						),
						nn.Conv2d(1024,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
						nn.BatchNorm2d(1024),
					),
					Lambda(lambda x: x), # Identity,
				),
				LambdaReduce(lambda x,y: x+y), # CAddTable,
				nn.ReLU(),
			),
			nn.Sequential( # Sequential,
				LambdaMap(lambda x: x, # ConcatTable,
					nn.Sequential( # Sequential,
						nn.Sequential( # Sequential,
							nn.Conv2d(1024,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
							nn.BatchNorm2d(1024),
							nn.ReLU(),
							nn.Conv2d(1024,1024,(3, 3),(1, 1),(1, 1),1,64,bias=False),
							nn.BatchNorm2d(1024),
							nn.ReLU(),
						),
						nn.Conv2d(1024,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
						nn.BatchNorm2d(1024),
					),
					Lambda(lambda x: x), # Identity,
				),
				LambdaReduce(lambda x,y: x+y), # CAddTable,
				nn.ReLU(),
			),
			nn.Sequential( # Sequential,
				LambdaMap(lambda x: x, # ConcatTable,
					nn.Sequential( # Sequential,
						nn.Sequential( # Sequential,
							nn.Conv2d(1024,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
							nn.BatchNorm2d(1024),
							nn.ReLU(),
							nn.Conv2d(1024,1024,(3, 3),(1, 1),(1, 1),1,64,bias=False),
							nn.BatchNorm2d(1024),
							nn.ReLU(),
						),
						nn.Conv2d(1024,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
						nn.BatchNorm2d(1024),
					),
					Lambda(lambda x: x), # Identity,
				),
				LambdaReduce(lambda x,y: x+y), # CAddTable,
				nn.ReLU(),
			),
			nn.Sequential( # Sequential,
				LambdaMap(lambda x: x, # ConcatTable,
					nn.Sequential( # Sequential,
						nn.Sequential( # Sequential,
							nn.Conv2d(1024,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
							nn.BatchNorm2d(1024),
							nn.ReLU(),
							nn.Conv2d(1024,1024,(3, 3),(1, 1),(1, 1),1,64,bias=False),
							nn.BatchNorm2d(1024),
							nn.ReLU(),
						),
						nn.Conv2d(1024,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
						nn.BatchNorm2d(1024),
					),
					Lambda(lambda x: x), # Identity,
				),
				LambdaReduce(lambda x,y: x+y), # CAddTable,
				nn.ReLU(),
			),
			nn.Sequential( # Sequential,
				LambdaMap(lambda x: x, # ConcatTable,
					nn.Sequential( # Sequential,
						nn.Sequential( # Sequential,
							nn.Conv2d(1024,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
							nn.BatchNorm2d(1024),
							nn.ReLU(),
							nn.Conv2d(1024,1024,(3, 3),(1, 1),(1, 1),1,64,bias=False),
							nn.BatchNorm2d(1024),
							nn.ReLU(),
						),
						nn.Conv2d(1024,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
						nn.BatchNorm2d(1024),
					),
					Lambda(lambda x: x), # Identity,
				),
				LambdaReduce(lambda x,y: x+y), # CAddTable,
				nn.ReLU(),
			),
			nn.Sequential( # Sequential,
				LambdaMap(lambda x: x, # ConcatTable,
					nn.Sequential( # Sequential,
						nn.Sequential( # Sequential,
							nn.Conv2d(1024,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
							nn.BatchNorm2d(1024),
							nn.ReLU(),
							nn.Conv2d(1024,1024,(3, 3),(1, 1),(1, 1),1,64,bias=False),
							nn.BatchNorm2d(1024),
							nn.ReLU(),
						),
						nn.Conv2d(1024,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
						nn.BatchNorm2d(1024),
					),
					Lambda(lambda x: x), # Identity,
				),
				LambdaReduce(lambda x,y: x+y), # CAddTable,
				nn.ReLU(),
			),
			nn.Sequential( # Sequential,
				LambdaMap(lambda x: x, # ConcatTable,
					nn.Sequential( # Sequential,
						nn.Sequential( # Sequential,
							nn.Conv2d(1024,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
							nn.BatchNorm2d(1024),
							nn.ReLU(),
							nn.Conv2d(1024,1024,(3, 3),(1, 1),(1, 1),1,64,bias=False),
							nn.BatchNorm2d(1024),
							nn.ReLU(),
						),
						nn.Conv2d(1024,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
						nn.BatchNorm2d(1024),
					),
					Lambda(lambda x: x), # Identity,
				),
				LambdaReduce(lambda x,y: x+y), # CAddTable,
				nn.ReLU(),
			),
			nn.Sequential( # Sequential,
				LambdaMap(lambda x: x, # ConcatTable,
					nn.Sequential( # Sequential,
						nn.Sequential( # Sequential,
							nn.Conv2d(1024,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
							nn.BatchNorm2d(1024),
							nn.ReLU(),
							nn.Conv2d(1024,1024,(3, 3),(1, 1),(1, 1),1,64,bias=False),
							nn.BatchNorm2d(1024),
							nn.ReLU(),
						),
						nn.Conv2d(1024,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
						nn.BatchNorm2d(1024),
					),
					Lambda(lambda x: x), # Identity,
				),
				LambdaReduce(lambda x,y: x+y), # CAddTable,
				nn.ReLU(),
			),
			nn.Sequential( # Sequential,
				LambdaMap(lambda x: x, # ConcatTable,
					nn.Sequential( # Sequential,
						nn.Sequential( # Sequential,
							nn.Conv2d(1024,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
							nn.BatchNorm2d(1024),
							nn.ReLU(),
							nn.Conv2d(1024,1024,(3, 3),(1, 1),(1, 1),1,64,bias=False),
							nn.BatchNorm2d(1024),
							nn.ReLU(),
						),
						nn.Conv2d(1024,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
						nn.BatchNorm2d(1024),
					),
					Lambda(lambda x: x), # Identity,
				),
				LambdaReduce(lambda x,y: x+y), # CAddTable,
				nn.ReLU(),
			),
			nn.Sequential( # Sequential,
				LambdaMap(lambda x: x, # ConcatTable,
					nn.Sequential( # Sequential,
						nn.Sequential( # Sequential,
							nn.Conv2d(1024,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
							nn.BatchNorm2d(1024),
							nn.ReLU(),
							nn.Conv2d(1024,1024,(3, 3),(1, 1),(1, 1),1,64,bias=False),
							nn.BatchNorm2d(1024),
							nn.ReLU(),
						),
						nn.Conv2d(1024,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
						nn.BatchNorm2d(1024),
					),
					Lambda(lambda x: x), # Identity,
				),
				LambdaReduce(lambda x,y: x+y), # CAddTable,
				nn.ReLU(),
			),
			nn.Sequential( # Sequential,
				LambdaMap(lambda x: x, # ConcatTable,
					nn.Sequential( # Sequential,
						nn.Sequential( # Sequential,
							nn.Conv2d(1024,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
							nn.BatchNorm2d(1024),
							nn.ReLU(),
							nn.Conv2d(1024,1024,(3, 3),(1, 1),(1, 1),1,64,bias=False),
							nn.BatchNorm2d(1024),
							nn.ReLU(),
						),
						nn.Conv2d(1024,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
						nn.BatchNorm2d(1024),
					),
					Lambda(lambda x: x), # Identity,
				),
				LambdaReduce(lambda x,y: x+y), # CAddTable,
				nn.ReLU(),
			),
			nn.Sequential( # Sequential,
				LambdaMap(lambda x: x, # ConcatTable,
					nn.Sequential( # Sequential,
						nn.Sequential( # Sequential,
							nn.Conv2d(1024,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
							nn.BatchNorm2d(1024),
							nn.ReLU(),
							nn.Conv2d(1024,1024,(3, 3),(1, 1),(1, 1),1,64,bias=False),
							nn.BatchNorm2d(1024),
							nn.ReLU(),
						),
						nn.Conv2d(1024,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
						nn.BatchNorm2d(1024),
					),
					Lambda(lambda x: x), # Identity,
				),
				LambdaReduce(lambda x,y: x+y), # CAddTable,
				nn.ReLU(),
			),
			nn.Sequential( # Sequential,
				LambdaMap(lambda x: x, # ConcatTable,
					nn.Sequential( # Sequential,
						nn.Sequential( # Sequential,
							nn.Conv2d(1024,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
							nn.BatchNorm2d(1024),
							nn.ReLU(),
							nn.Conv2d(1024,1024,(3, 3),(1, 1),(1, 1),1,64,bias=False),
							nn.BatchNorm2d(1024),
							nn.ReLU(),
						),
						nn.Conv2d(1024,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
						nn.BatchNorm2d(1024),
					),
					Lambda(lambda x: x), # Identity,
				),
				LambdaReduce(lambda x,y: x+y), # CAddTable,
				nn.ReLU(),
			),
			nn.Sequential( # Sequential,
				LambdaMap(lambda x: x, # ConcatTable,
					nn.Sequential( # Sequential,
						nn.Sequential( # Sequential,
							nn.Conv2d(1024,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
							nn.BatchNorm2d(1024),
							nn.ReLU(),
							nn.Conv2d(1024,1024,(3, 3),(1, 1),(1, 1),1,64,bias=False),
							nn.BatchNorm2d(1024),
							nn.ReLU(),
						),
						nn.Conv2d(1024,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
						nn.BatchNorm2d(1024),
					),
					Lambda(lambda x: x), # Identity,
				),
				LambdaReduce(lambda x,y: x+y), # CAddTable,
				nn.ReLU(),
			),
		),
		nn.Sequential( # Sequential,
			nn.Sequential( # Sequential,
				LambdaMap(lambda x: x, # ConcatTable,
					nn.Sequential( # Sequential,
						nn.Sequential( # Sequential,
							nn.Conv2d(1024,2048,(1, 1),(1, 1),(0, 0),1,1,bias=False),
							nn.BatchNorm2d(2048),
							nn.ReLU(),
							nn.Conv2d(2048,2048,(3, 3),(2, 2),(1, 1),1,64,bias=False),
							nn.BatchNorm2d(2048),
							nn.ReLU(),
						),
						nn.Conv2d(2048,2048,(1, 1),(1, 1),(0, 0),1,1,bias=False),
						nn.BatchNorm2d(2048),
					),
					nn.Sequential( # Sequential,
						nn.Conv2d(1024,2048,(1, 1),(2, 2),(0, 0),1,1,bias=False),
						nn.BatchNorm2d(2048),
					),
				),
				LambdaReduce(lambda x,y: x+y), # CAddTable,
				nn.ReLU(),
			),
			nn.Sequential( # Sequential,
				LambdaMap(lambda x: x, # ConcatTable,
					nn.Sequential( # Sequential,
						nn.Sequential( # Sequential,
							nn.Conv2d(2048,2048,(1, 1),(1, 1),(0, 0),1,1,bias=False),
							nn.BatchNorm2d(2048),
							nn.ReLU(),
							nn.Conv2d(2048,2048,(3, 3),(1, 1),(1, 1),1,64,bias=False),
							nn.BatchNorm2d(2048),
							nn.ReLU(),
						),
						nn.Conv2d(2048,2048,(1, 1),(1, 1),(0, 0),1,1,bias=False),
						nn.BatchNorm2d(2048),
					),
					Lambda(lambda x: x), # Identity,
				),
				LambdaReduce(lambda x,y: x+y), # CAddTable,
				nn.ReLU(),
			),
			nn.Sequential( # Sequential,
				LambdaMap(lambda x: x, # ConcatTable,
					nn.Sequential( # Sequential,
						nn.Sequential( # Sequential,
							nn.Conv2d(2048,2048,(1, 1),(1, 1),(0, 0),1,1,bias=False),
							nn.BatchNorm2d(2048),
							nn.ReLU(),
							nn.Conv2d(2048,2048,(3, 3),(1, 1),(1, 1),1,64,bias=False),
							nn.BatchNorm2d(2048),
							nn.ReLU(),
						),
						nn.Conv2d(2048,2048,(1, 1),(1, 1),(0, 0),1,1,bias=False),
						nn.BatchNorm2d(2048),
					),
					Lambda(lambda x: x), # Identity,
				),
				LambdaReduce(lambda x,y: x+y), # CAddTable,
				nn.ReLU(),
			),
		),
		nn.AvgPool2d((7, 7),(1, 1)),
		Lambda(lambda x: x.view(x.size(0),-1)), # View,
		nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(2048,1000)), # Linear,
	)