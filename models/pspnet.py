import torch
from torch import nn
from torchvision import models

from util.setup import load_args
args = load_args(os.getcwd())
pretrained_dir = args.paths["pretrained_models_path"]


class _PyramidPoolingModule(nn.Module):
	def __init__(self, in_channels, out_channels, out_size, levels=(1, 2, 3, 6)):
		super(_PyramidPoolingModule, self).__init__()

		self.num_features = len(levels) * out_channels
		self.layers = nn.ModuleList()
		for level in levels:
			self.layers.append(nn.Sequential(
				nn.AdaptiveAvgPool2d(level),
				nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False),
				nn.BatchNorm2d(out_channels),
				nn.ReLU(inplace=True),
				nn.Upsample(size=out_size, mode='bilinear')
			))

	def forward(self, x):
		
		features = [layer(x) for layer in self.layers]
		out = torch.cat(features, 1)

		return out

class PSPNet(nn.Module):

	def __init__(self, num_classes, img_size):
		super(PSPNet, self).__init__()

		# load resnet
		resnet = models.resnet152()
		res152_path = os.path.join(pretrained_dir, 'resnet152.pth')
		resnet.load_state_dict(torch.load(res152_path))

		# start resnet
		self.resblock0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)

		# resnet blocks
		self.resblock1 = resnet.layer1
		self.resblock2 = resnet.layer2
		self.resblock3 = resnet.layer3
		self.resblock4 = resnet.layer4

		# pyramid spatial pooling module
		self.ppm = _PyramidPoolingModule(512, 32, img_size)
		self.final = nn.Sequential(
			nn.Conv2d(512 + self.ppm.num_features, num_classes, kernel_size=3, padding=1),
			nn.BatchNorm2d(num_classes),
			nn.ReLU(inplace=True)
		)

		# define upsampler
		self.up = nn.Upsample(size=img_size, mode='bilinear')

		# initialize weights for non-resnet layers
		initialize_weights(self.first, self.ppm, self.final)

	def forward(self, x):
		
		x = self.resblock0(x)

		x = self.resblock1(x)
		x = self.resblock2(x)
		aux = self.resblock3(x)
		x = self.resblock4(aux)

		x = torch.cat([self.ppm(x), x], 1)
		x = self.final(x)

		x = self.up(x)
		aux = self.up(aux)

		return x, aux


def initialize_weights(*blocks):
	for block in blocks:
		for module in block.modules():
			if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
				nn.init.kaiming_normal(module.weight)
				if module.bias is not None:
					module.bias.data.zero_()
			elif isinstance(module, nn.BatchNorm2d):
				module.weight.data.fill_(1)
				module.bias.data.zero_()
