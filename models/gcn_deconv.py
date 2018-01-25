import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models
import os
from math import floor
import yaml

from util.setup import load_args
args = load_args(os.getcwd())
pretrained_dir = args.paths["pretrained_models_path"]


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

class _UpsampleModule(nn.Module):
	def __init(self, channels):
		super(_UpsampleModule, self).__init__()
		self.pre = nn.Sequential(
			nn.Conv2d(channels, channels),
			nn.BatchNorm2d(channels),
			nn.ReLU(inplace=True)
		)
		self.deconv = nn.ConvTranspose2d(channels, channels, 2, stride=2, padding=1)
		self.upsample = nn.Upsample(scale_factor=2, mode="bilinear")

	def forward(self, x):
		p = self.pre(x)
		de = self.deconv(p)
		up = self.upsample(x)
		# out = torch.cat([de, up], 1)
		out = de + up
		return out

class GCN_DECONV(nn.Module):
	def __init__(self, num_classes, input_size, k=7):
		super(GCN_DECONV, self).__init__()

		self.K = k
		self.input_size = input_size

		# https://download.pytorch.org/models/resnet152-b121ed2d.pth
		res152_path = os.path.join(pretrained_dir, 'resnet152.pth')

		resnet = models.resnet152()
		resnet.load_state_dict(torch.load(res152_path))

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

		self.up1 = nn.ConvTranspose2d(num_classes, num_classes, 4, stride=2, padding=1)
		self.up2 = nn.ConvTranspose2d(num_classes, num_classes, 4, stride=2, padding=1)
		self.up3 = nn.ConvTranspose2d(num_classes, num_classes, 4, stride=2, padding=1)
		self.up4 = nn.ConvTranspose2d(num_classes, num_classes, 4, stride=2, padding=1)
		self.up5 = nn.ConvTranspose2d(num_classes, num_classes, 4, stride=2, padding=1)

		self.deconv1 = _UpsampleModule(num_classes)
		self.deconv2 = _UpsampleModule(num_classes)
		self.deconv3 = _UpsampleModule(num_classes)
		self.deconv4 = _UpsampleModule(num_classes)
		self.deconv5 = _UpsampleModule(num_classes)

		initialize_weights(self.gcm1, self.gcm2, self.gcm3, self.gcm4, self.brm1, self.brm2, self.brm3,
					self.brm4, self.brm5, self.brm6, self.brm7, self.brm8, self.brm9,
					self.up1, self.up2, self.up3, self.up4, self.up5)
					# self.deconv1, self.deconv2, self.deconv3, self.deconv4, self.deconv5)

	def forward(self, x):
		fm0 = self.layer0(x)
		fm1 = self.layer1(fm0)
		fm2 = self.layer2(fm1)
		fm3 = self.layer3(fm2)
		fm4 = self.layer4(fm3)

		gcfm1 = self.brm1(self.gcm1(fm4))
		gcfm2 = self.brm2(self.gcm2(fm3))
		gcfm3 = self.brm3(self.gcm3(fm2))
		gcfm4 = self.brm4(self.gcm4(fm1))

		# fs1 = self.brm5(F.upsample(gcfm1, fm3.size()[2:], mode='bilinear') + gcfm2)
		# fs2 = self.brm6(F.upsample(fs1, fm2.size()[2:], mode='bilinear') + gcfm3)
		# fs3 = self.brm7(F.upsample(fs2, fm1.size()[2:], mode='bilinear') + gcfm4)
		# fs4 = self.brm8(F.upsample(fs3, fm0.size()[2:], mode='bilinear'))
		# out = self.brm9(F.upsample(fs4, self.input_size, mode='bilinear'))

		# fs1 = self.brm5(self.deconv1(gcfm1) + gcfm2)
		# fs2 = self.brm6(self.deconv2(fs1) + gcfm3)
		# fs3 = self.brm7(self.deconv3(fs2) + gcfm4)
		# fs4 = self.brm8(self.deconv4(fs3))
		# out = self.brm9(self.deconv5(fs4))

		fs1 = self.brm5(self.up1(gcfm1) + gcfm2)
		fs2 = self.brm6(self.up2(fs1) + gcfm3)
		fs3 = self.brm7(self.up3(fs2) + gcfm4)
		fs4 = self.brm8(self.up4(fs3))
		out = self.brm9(self.up5(fs4))

		return out


def initialize_weights(*models):
	for model in models:
		for module in model.modules():
			if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
				nn.init.kaiming_normal(module.weight)
				if module.bias is not None:
					module.bias.data.zero_()
			elif isinstance(module, nn.BatchNorm2d):
				module.weight.data.fill_(1)
				module.bias.data.zero_()
			elif isinstance(module, nn.ConvTranspose2d):
				nn.init.kaiming_normal(module.weight)
				if module.bias is not None:
					module.bias.data.zero_()