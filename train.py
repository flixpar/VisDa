import torch
from torch import nn
from torch import optim
from torch import autograd
from torch.utils import data

from dataloader import VisDaDataset
from gcn import GCN
from util import CrossEntropyLoss2d

import os
from tqdm import tqdm

# config:
epochs = 25
batch_size = 4
lr = 2e-6
weight_decay = 1e-6
momentum = 0.95

base_path = "/home/flixpar/VisDa"
save_path = os.path.join(base_path, "saves", "gcn-%d.pth")

log_file_path = "/home/flixpar/VisDa/train.log"
log_file = open(log_file_path, 'w')

torch.backends.cudnn.benchmark = True
# print("GPUs: {}".format(torch.cuda.device_count()))

dataset = VisDaDataset()
dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=6)

model = GCN(dataset.num_classes, dataset.img_size).cuda()
model.train()

optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=True)
criterion = CrossEntropyLoss2d(weight=dataset.class_weights).cuda()

print("Starting training...")

for epoch in range(epochs):

	for i, (image, label) in tqdm(enumerate(dataloader), total=int(len(dataset)/batch_size)):
		img = autograd.Variable(image.cuda())
		lbl = autograd.Variable(label.cuda())

		output = model(img)
		loss = criterion(output, lbl)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if i % 100 == 0:
			tqdm.write("loss: {}".format(loss.data[0]))
			log_file.write(str(loss.data[0]))

	print("Epoch {} completed.".format(epoch + 1))
	log_file.write("Epoch {} completed.".format(epoch + 1))

	#if (epoch + 1) % 25 == 0:
	if True:
		lr /= 5
		optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
		torch.save(model.state_dict(), save_path % (epoch + 1))

torch.save(model.state_dict(), os.path.join(base_path, "saves", "gcn-final.pth"))
