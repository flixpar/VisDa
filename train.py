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
max_epochs = 20
batch_size = 2
lr = 2e-3
weight_decay = 0.0005
momentum = 0.99
K = 7
img_size = (1052, 1914)
lr_decay_freq = 2
lr_decay_rate = 2

base_path = "/home/flixpar/VisDa"
save_path = os.path.join(base_path, "saves", "gcn-{}.pth")

log_file_path = os.path.join(base_path, "saves", "train.log")
log_file = open(log_file_path, 'w')

torch.backends.cudnn.benchmark = True
# print("GPUs: {}".format(torch.cuda.device_count()))

dataset = VisDaDataset(img_size=img_size)
dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=6)

model = GCN(dataset.num_classes, dataset.img_size, k=K).cuda()
model.train()

optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=True)
criterion = CrossEntropyLoss2d(weight=dataset.class_weights).cuda()

print("Starting training...")
for epoch in range(max_epochs):

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
			log_file.write(str(loss.data[0])+"\n")

	print("Epoch {} completed.".format(epoch + 1))
	log_file.write("Epoch {} completed.\n".format(epoch + 1))
	torch.save(model.state_dict(), save_path.format(epoch + 1))

	if (epoch + 1) % lr_decay_freq == 0:
		lr /= lr_decay_rate
		optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)


torch.save(model.state_dict(), save_path.format("final"))
