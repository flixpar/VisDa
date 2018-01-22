import random
from torch.utils import data

class EvalDataloader(data.Dataset):

	def __init__(self, dataset, samples, rand=True):

		self.dataset = dataset
		self.num_samples = samples
		self.img_size = self.dataset.img_size

		self.chosen = random.sample(range(len(self.dataset)), self.num_samples)

	def __getitem__(self, index):
		idx = self.chosen[index]
		processed = self.dataset.__getitem__(idx)
		unprocessed = self.dataset.get_original(idx)
		return processed, unprocessed

	def __len__(self):
		return self.num_samples
