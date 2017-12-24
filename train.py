import torch
from torch import nn
from torch import optim
from torch import autograd
from torch.utils import data

from dataloader import VisDaDataset
from gcn import GCN
from util import CrossEntropyLoss2d

import os
import tqdm

# config:
epochs = 25  # 100
batch_size = 1
lr = 1e-4
weight_decay = 2e-5
momentum = 0.9

base_path = "/home/flixpar/VisDa"
save_path = os.path.join(base_path, "saves", "gcn-%d.pth")

torch.backends.cudnn.benchmark = True
# print("GPUs: {}".format(torch.cuda.device_count()))

dataset = VisDaDataset()
dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)

model = GCN(dataset.num_classes, dataset.img_size).cuda()
model.train()

optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
criterion = CrossEntropyLoss2d(weight=dataset.class_weights.cuda())

print("Starting training...")

for epoch in range(epochs):

	for i, (image, label) in tqdm.tqdm(enumerate(dataloader)):
		img = autograd.Variable(image.cuda())
		lbl = autograd.Variable(label.cuda())
		#img = autograd.Variable(image)
		#lbl = autograd.Variable(label)

		output = model(img)
		loss = criterion(output, lbl)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

	print("Epoch {} completed.".format(epoch + 1))

	if (epoch + 1) % 25 == 0:
		lr /= 5
		optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
		torch.save(model.state_dict(), save_path % (epoch + 1))

torch.save(model.state_dict(), os.path.join(base_path, "saves", "gcn-final.pth"))
