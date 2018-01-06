import torch
from torch import nn
from torch import optim
from torch import autograd
from torch.utils import data
torch.backends.cudnn.benchmark = True

from dataloader import VisDaDataset
from gcn import GCN
from util import CrossEntropyLoss2d

import os
from tqdm import tqdm
import yaml

# config:
config_path = "/home/flixpar/VisDa/config.yaml"
args = Namespace(**yaml.load(open(config_path, 'r')))
args.img_size = tuple(args.img_size)

save_path = os.path.join(args.base_path, "saves", "gcn-{}.pth")

log_file_path = os.path.join(args.base_path, "saves", "train.log")
log_file = open(log_file_path, 'w')

dataset = VisDaDataset(im_size=args.img_size)
dataloader = data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=6)

model = GCN(dataset.num_classes, dataset.img_size, k=args.K).cuda()
model.train()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
criterion = CrossEntropyLoss2d(weight=dataset.class_weights).cuda()

print("Starting training...")
for epoch in range(args.max_epochs):

	for i, (image, label) in tqdm(enumerate(dataloader), total=int(len(dataset)/args.batch_size)):
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

	if (epoch + 1) % args.lr_decay_freq == 0:
		args.lr /= args.lr_decay_rate
		optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)


torch.save(model.state_dict(), save_path.format("final"))
